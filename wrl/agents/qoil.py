"""Q-OIL agent — residual SAC with DECOUPLED critics for learning from
interventions (Q-Optimistic Intervention Learning).

Two critics (the paper's key trick to avoid catastrophic over-optimism):
  * Q_TD  (task critic): standard Bellman target on the task reward, bootstraps
    itself -> unbiased value.  Q_TD = r_task + gamma * Q_TD(s',a').
  * Q_Opt (optimistic critic): adds the optimism BONUS on intervention
    transitions but bootstraps from Q_TD's target, NOT itself:
        Q_Opt = r_task + r_bonus + gamma * Q_TD_target(s', a').
    This confines the bonus to intervention states (it can't propagate back to
    the predecessor/failing states).

The actor maximizes Q_Opt and is BC-regularized (lambda ~= 0.1) on intervention
actions. r_bonus = b * is_intervention, read from the buffer's is_intervention
flag (PAM-gated expert takeovers are logged with is_intervention=True).

Subclasses ResidualSACAgent: the learner is a residual on a frozen base policy
(the suboptimal pixel DP); interventions execute the expert (pi_h) action.
"""

from functools import partial
from typing import ClassVar, FrozenSet, Iterable, Optional

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from wrl.agents.residual_sac import ResidualPolicy, ResidualSACAgent
from wrl.common.common import JaxRLTrainState, ModuleDict
from wrl.common.encoding import EncodingWrapper
from wrl.common.optimizers import make_optimizer
from wrl.common.typing import Data, Params, PRNGKey
from wrl.networks.actor_critic_nets import Critic, ensemblize
from wrl.networks.lagrange import GeqLagrangeMultiplier
from wrl.networks.mlp import MLP


class QOILAgent(ResidualSACAgent):
    CRITIC_NETWORKS: ClassVar[FrozenSet[str]] = frozenset({"critic", "opt_critic"})
    ALL_NETWORKS: ClassVar[FrozenSet[str]] = frozenset(
        {"critic", "opt_critic", "actor", "temperature"}
    )

    # ---- optimistic critic forward ---------------------------------------
    def forward_opt_critic(self, observations, actions, rng, *, grad_params=None):
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations, actions, name="opt_critic",
            rngs={"dropout": rng} if grad_params is not None else {},
            train=grad_params is not None,
        )

    # ---- Q_Opt loss: bonus + bootstrap from Q_TD target ------------------
    def opt_critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        rng, next_rng, td_rng, q_rng = jax.random.split(rng, 4)
        next_full, _ = self._compute_next_actions(batch, next_rng)

        # TASK-critic target (NOT the optimistic critic's own target)
        target_td = self.forward_target_critic(batch["next_observations"], next_full, rng=td_rng)
        if self.config["critic_subsample_size"] is not None:
            rng, sub = jax.random.split(rng)
            idx = jax.random.randint(sub, (self.config["critic_subsample_size"],),
                                     0, self.config["critic_ensemble_size"])
            target_td = target_td[idx]
        target_min = target_td.min(axis=0)

        bonus = self.config["intervention_bonus"] * batch["is_intervention"].astype(jnp.float32)
        target_q = batch["rewards"] + bonus + self.config["discount"] * batch["masks"] * target_min
        target_q = jax.lax.stop_gradient(target_q)

        predicted = self.forward_opt_critic(
            batch["observations"], batch["actions"], rng=q_rng, grad_params=params)
        chex.assert_shape(predicted, (self.config["critic_ensemble_size"], batch_size))
        opt_loss = jnp.mean((predicted - target_q[None]) ** 2)
        return opt_loss, {"opt_critic_loss": opt_loss, "opt_q": predicted.mean(),
                          "bonus_frac": batch["is_intervention"].mean()}

    # ---- actor maximizes Q_Opt (+ BC reg on interventions) --------------
    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()
        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        base = batch["base_actions"]
        dist = self.forward_policy(batch["observations"], base, rng=policy_rng, grad_params=params)
        residual, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        full = self._compose_action(residual, base)

        # OPTIMISTIC critic drives policy improvement
        predicted_qs = self.forward_opt_critic(batch["observations"], full, rng=critic_rng)
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))

        actor_loss = -jnp.mean(predicted_q - temperature * log_probs)
        info = {"actor_loss": actor_loss, "temperature": temperature,
                "entropy": -log_probs.mean(), "residual_abs_mean": jnp.abs(residual).mean()}

        # BC regularization on intervention (expert) actions -> residual target
        bc_weight = self.config.get("bc_weight", 0.0)
        if bc_weight > 0 and "is_intervention" in batch:
            scale = self._current_edit_scale() * self._residual_mask()
            residual_target = jnp.clip(
                (batch["actions"] - base) / jnp.maximum(scale, 1e-6), -0.999, 0.999)
            demo_logp = dist.log_prob(residual_target)
            mask = batch["is_intervention"].astype(jnp.float32)
            bc_loss = -(mask * demo_logp).sum() / (mask.sum() + 1e-6)
            actor_loss = actor_loss + bc_weight * bc_loss
            info["bc_loss"] = bc_loss
        return actor_loss, info

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss_fn, batch),         # Q_TD (task only)
            "opt_critic": partial(self.opt_critic_loss_fn, batch),  # Q_Opt (+ bonus)
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    # ---- construction (mirrors ResidualSACAgent.create_pixels + opt critic) --
    @classmethod
    def create_pixels(
        cls, rng, observations, sample_full_action, sample_base_action, *,
        action_dim, horizon, edit_scale=0.25, residual_xyzg=False,
        encoder_type="resnet", use_proprio=True, image_keys=("image",),
        critic_network_kwargs={"hidden_dims": [256, 256], "activations": nn.tanh,
                               "use_layer_norm": True},
        policy_network_kwargs={"hidden_dims": [256, 256], "activations": nn.tanh,
                               "use_layer_norm": True},
        policy_kwargs={"tanh_squash_distribution": True, "std_parameterization": "exp",
                       "std_min": 1e-5, "std_max": 5},
        critic_ensemble_size=2, critic_subsample_size=None, temperature_init=1e-2,
        intervention_bonus=0.1, bc_weight=0.1, discount_per_step=0.97,
        edit_scale_end=None, edit_scale_incr=0.0, edit_scale_steps=2500,
        target_entropy=None, augmentation_function=None, **kwargs,
    ):
        from wrl.vision.resnet_v1 import resnetv1_configs
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}
        encoders = {k: resnetv1_configs["resnetv1-10"](
            pooling_method="spatial_learned_embeddings", num_spatial_blocks=8,
            bottleneck_dim=256, pre_pooling=False, name=f"encoder_{k}") for k in image_keys}
        encoder_def = EncodingWrapper(encoder=encoders, use_proprio=use_proprio,
                                      enable_stacking=True, image_keys=image_keys)

        def critic():
            backbone = ensemblize(partial(MLP, **critic_network_kwargs),
                                  critic_ensemble_size)(name="critic_ensemble")
            return Critic(encoder=encoder_def, network=backbone)

        actor_def = ResidualPolicy(encoder=encoder_def, network=MLP(**policy_network_kwargs),
                                   action_dim=sample_full_action.shape[-1], name="actor", **policy_kwargs)
        critic_def = critic()
        opt_critic_def = critic()
        temperature_def = GeqLagrangeMultiplier(init_value=temperature_init,
                                                constraint_shape=(), name="temperature")

        model_def = ModuleDict({"actor": actor_def, "critic": critic_def,
                                "opt_critic": opt_critic_def, "temperature": temperature_def})
        txs = {k: make_optimizer(learning_rate=3e-4)
               for k in ("actor", "critic", "opt_critic", "temperature")}
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng,
                                actor=[observations, sample_base_action],
                                critic=[observations, sample_full_action],
                                opt_critic=[observations, sample_full_action],
                                temperature=[])["params"]
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(apply_fn=model_def.apply, params=params, txs=txs,
                                       target_params=params, rng=create_rng)

        chunk_dim = sample_full_action.shape[-1]
        if target_entropy is None:
            target_entropy = -chunk_dim / 2
        return cls(state=state, config=dict(
            critic_ensemble_size=critic_ensemble_size, critic_subsample_size=critic_subsample_size,
            discount=discount_per_step ** horizon, discount_per_step=discount_per_step,
            soft_target_update_rate=0.005, target_entropy=target_entropy, backup_entropy=False,
            edit_scale_start=float(edit_scale),
            edit_scale_end=float(edit_scale if edit_scale_end is None else edit_scale_end),
            edit_scale_incr=float(edit_scale_incr), edit_scale_steps=int(edit_scale_steps),
            residual_xyzg=residual_xyzg, action_dim=action_dim, horizon=horizon,
            replan_steps=horizon, chunk_dim=chunk_dim, image_keys=image_keys, reward_bias=0.0,
            intervention_bonus=intervention_bonus, bc_weight=bc_weight,
            augmentation_function=augmentation_function, **kwargs))
