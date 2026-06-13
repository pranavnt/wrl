"""Residual SAC — the EXPO-FT path.

A residual policy rides on top of a frozen base policy (served separately). The
base policy emits an `H`-step action chunk `a_base` (flattened to dim `A*H`);
the learned residual edits it:

    a_full = a_base + edit_scale * mask * residual          # executed chunk

The critic scores the full chunk `Q(obs, a_full)`; the actor outputs the
residual (conditioned on the base chunk) and is improved against `Q`. SAC runs
at the chunk-MDP level: the stored reward is the discounted sum over the chunk
and the bootstrap discount is `per_step_discount ** H` (folded into
`config["discount"]`, so the inherited critic loss is unchanged).

`a_full = a_base + s * r` is an affine map of the residual `r` (base = constant
shift, `s = edit_scale` = constant scale), so `log pi(a_full) = log pi_r(r) -
A*H*log(s)` differs from the residual log-prob only by a constant. That constant
drops out of both the actor gradient and the temperature update, so we use the
residual log-prob directly with `target_entropy = -A*H/2` — mathematically
equivalent to expo-ft's explicit edit-scale entropy correction, and cleaner.

A zero base policy reduces this to chunked SAC from scratch.
"""

from functools import partial
from typing import ClassVar, FrozenSet, Iterable, Optional

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from wrl.agents.sac import SACAgent
from wrl.common.common import JaxRLTrainState, ModuleDict, default_init
from wrl.common.encoding import EncodingWrapper
from wrl.common.optimizers import make_optimizer
from wrl.common.typing import Data, Params, PRNGKey
from wrl.networks.actor_critic_nets import (
    Critic,
    TanhMultivariateNormalDiag,
    ensemblize,
)
from wrl.networks.lagrange import GeqLagrangeMultiplier
from wrl.networks.mlp import MLP


# ---------------------------------------------------------------------------
# residual policy network (base-action conditioned)
# ---------------------------------------------------------------------------


class ResidualPolicy(nn.Module):
    """Like `Policy`, but conditioned on the base action chunk: the base chunk
    is concatenated into the MLP input after encoding the observation."""

    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int  # flattened chunk dim A*H
    init_final: Optional[float] = None
    std_parameterization: str = "exp"
    std_min: float = 1e-5
    std_max: float = 10.0
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(
        self,
        observations,
        base_action: jnp.ndarray,
        temperature: float = 1.0,
        train: bool = False,
    ) -> distrax.Distribution:
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)

        inputs = jnp.concatenate([obs_enc, base_action], axis=-1)
        outputs = self.network(inputs, train=train)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        if self.std_parameterization == "exp":
            log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
            stds = jnp.exp(log_stds)
        elif self.std_parameterization == "uniform":
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            stds = jnp.exp(log_stds)
        else:
            raise ValueError(f"Invalid std_parameterization: {self.std_parameterization}")

        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        if self.tanh_squash_distribution:
            return TanhMultivariateNormalDiag(loc=means, scale_diag=stds)
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)


# ---------------------------------------------------------------------------
# residual SAC agent
# ---------------------------------------------------------------------------


class ResidualSACAgent(SACAgent):
    """Chunked residual SAC. Inherits the SAC update/critic/temperature
    machinery; overrides only the base-action-aware pieces."""

    CRITIC_NETWORKS: ClassVar[FrozenSet[str]] = frozenset({"critic"})
    ALL_NETWORKS: ClassVar[FrozenSet[str]] = frozenset({"critic", "actor", "temperature"})

    # ---- residual composition -------------------------------------------

    def _residual_mask(self):
        """Per-chunk multiplicative mask. When `residual_xyzg`, zero the
        rotation dims (3,4,5) of each per-step action so the residual only
        edits translation (0,1,2) + gripper (last)."""
        if not self.config["residual_xyzg"]:
            return 1.0
        A = self.config["action_dim"]
        H = self.config["horizon"]
        per_step = jnp.ones(A).at[3:6].set(0.0)
        return jnp.tile(per_step, H)

    def _current_edit_scale(self):
        """edit_scale ramped from start to end as a curriculum: start tiny so the
        base keeps succeeding during collection (reward signal flows), then give
        the residual more authority. Computed from the train step (traced -> no
        JIT recompiles). Constant when start == end."""
        c = self.config
        k = jnp.floor(self.state.step / c["edit_scale_steps"])
        return jnp.minimum(
            c["edit_scale_end"], c["edit_scale_start"] + c["edit_scale_incr"] * k
        )

    def _compose_action(self, residual: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
        scaled = residual * self._current_edit_scale() * self._residual_mask()
        return base + scaled

    # ---- base-action-conditioned policy ---------------------------------

    def forward_policy(
        self,
        observations: Data,
        base_action: jnp.ndarray,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            base_action,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def _compute_next_actions(self, batch, rng):
        """Next full action = base(s') + edit_scale * residual(s', base(s'))."""
        batch_size = batch["rewards"].shape[0]
        next_base = batch["next_base_actions"]
        dist = self.forward_policy(batch["next_observations"], next_base, rng=rng)
        next_residual, next_log_probs = dist.sample_and_log_prob(seed=rng)
        next_full = self._compose_action(next_residual, next_base)
        chex.assert_equal_shape([batch["actions"], next_full])
        chex.assert_shape(next_log_probs, (batch_size,))
        return next_full, next_log_probs

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        base = batch["base_actions"]
        dist = self.forward_policy(
            batch["observations"], base, rng=policy_rng, grad_params=params
        )
        residual, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        full = self._compose_action(residual, base)

        predicted_qs = self.forward_critic(batch["observations"], full, rng=critic_rng)
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        actor_loss = -jnp.mean(predicted_q - temperature * log_probs)
        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),
            "residual_abs_mean": jnp.abs(residual).mean(),
        }

        # BC regularization: anchor the residual toward the demo correction
        # r* = (a_demo - a_base) / edit_scale on demo transitions only. On a
        # competent base r* is small, so this pins the residual near 0 and
        # prevents the zero-reward drift that collapses the policy. Masked by
        # is_intervention so online transitions don't contribute a BC term.
        bc_weight = self.config.get("bc_weight", 0.0)
        bc_only = self.config.get("bc_only", False)
        if bc_weight > 0 and "is_intervention" in batch:
            scale = self._current_edit_scale() * self._residual_mask()
            residual_target = jnp.clip(
                (batch["actions"] - base) / jnp.maximum(scale, 1e-6), -0.999, 0.999
            )
            demo_logp = dist.log_prob(residual_target)
            mask = batch["is_intervention"].astype(jnp.float32)
            bc_loss = -(mask * demo_logp).sum() / (mask.sum() + 1e-6)
            # bc_only: drop the Q/entropy term entirely -> pure behavior cloning
            # of the residual (the critic still trains but the actor ignores it).
            actor_loss = bc_loss if bc_only else actor_loss + bc_weight * bc_loss
            info["bc_loss"] = bc_loss
        return actor_loss, info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        base_action: jnp.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """Return the full executed chunk `a_base + edit_scale * residual`."""
        dist = self.forward_policy(observations, base_action, rng=seed, train=False)
        residual = dist.mode() if argmax else dist.sample(seed=seed)
        return self._compose_action(residual, base_action)

    # ---- construction ----------------------------------------------------

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        sample_full_action: jnp.ndarray,
        sample_base_action: jnp.ndarray,
        *,
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # residual config (edit_scale may be a curriculum: start -> end)
        edit_scale: float,
        residual_xyzg: bool,
        action_dim: int,
        horizon: int,
        edit_scale_end: Optional[float] = None,   # None -> fixed edit_scale
        edit_scale_incr: float = 0.0,
        edit_scale_steps: int = 2500,
        # algorithm config
        discount_per_step: float = 0.97,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        actor_optimizer_kwargs={"learning_rate": 3e-4},
        critic_optimizer_kwargs={"learning_rate": 3e-4},
        temperature_optimizer_kwargs={"learning_rate": 3e-4},
        **kwargs,
    ):
        model_def = ModuleDict(
            {"actor": actor_def, "critic": critic_def, "temperature": temperature_def}
        )
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations, sample_base_action],
            critic=[observations, sample_full_action],
            temperature=[],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        chunk_dim = sample_full_action.shape[-1]
        assert chunk_dim == action_dim * horizon, (
            f"full action dim {chunk_dim} != action_dim*horizon "
            f"({action_dim}*{horizon})"
        )
        if target_entropy is None:
            target_entropy = -chunk_dim / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                # chunk-MDP discount: per-step discount over the H-step chunk
                discount=discount_per_step ** horizon,
                discount_per_step=discount_per_step,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                # residual (edit_scale curriculum: start -> end)
                edit_scale_start=float(edit_scale),
                edit_scale_end=float(edit_scale if edit_scale_end is None else edit_scale_end),
                edit_scale_incr=float(edit_scale_incr),
                edit_scale_steps=int(edit_scale_steps),
                residual_xyzg=residual_xyzg,
                action_dim=action_dim,
                horizon=horizon,
                replan_steps=horizon,
                chunk_dim=chunk_dim,
                # misc
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        sample_full_action: jnp.ndarray,
        sample_base_action: jnp.ndarray,
        *,
        action_dim: int,
        horizon: int,
        edit_scale: float = 1.0,
        residual_xyzg: bool = False,
        encoder_type: str = "resnet",
        use_proprio: bool = True,
        image_keys: Iterable[str] = ("image",),
        critic_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1e-2,
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        policy_network_kwargs = {**policy_network_kwargs, "activate_final": True}
        critic_network_kwargs = {**critic_network_kwargs, "activate_final": True}

        if encoder_type == "resnet":
            from wrl.vision.resnet_v1 import resnetv1_configs

            encoders = {
                k: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pre_pooling=False,
                    name=f"encoder_{k}",
                )
                for k in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            from wrl.vision.resnet_v1 import PreTrainedResNetEncoder, resnetv1_configs

            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True, name="pretrained_encoder",
            )
            encoders = {
                k: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{k}",
                )
                for k in image_keys
            }
        else:
            raise NotImplementedError(f"Unknown encoder type: {encoder_type}")

        # Shared encoder instance for actor + critic (Flax shares the params;
        # the actor stop-grads it, so only the critic trains the encoder).
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,
            enable_stacking=True,
            image_keys=image_keys,
        )

        critic_backbone = ensemblize(
            partial(MLP, **critic_network_kwargs), critic_ensemble_size
        )(name="critic_ensemble")
        critic_def = Critic(
            encoder=encoder_def, network=critic_backbone, name="critic"
        )

        actor_def = ResidualPolicy(
            encoder=encoder_def,
            network=MLP(**policy_network_kwargs),
            action_dim=sample_full_action.shape[-1],
            name="actor",
            **policy_kwargs,
        )

        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,
            constraint_shape=(),
            constraint_type="geq",
            name="temperature",
        )

        agent = cls.create(
            rng,
            observations,
            sample_full_action,
            sample_base_action,
            actor_def=actor_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            edit_scale=edit_scale,
            residual_xyzg=residual_xyzg,
            action_dim=action_dim,
            horizon=horizon,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        if "pretrained" in encoder_type:
            from wrl.utils.train_utils import load_resnet10_params

            agent = load_resnet10_params(agent, image_keys)
        return agent


def make_residual_sac_pixel_agent(
    seed,
    sample_obs,
    sample_full_action,
    sample_base_action,
    *,
    action_dim,
    horizon,
    image_keys=("image",),
    encoder_type="resnet",
    edit_scale=1.0,
    edit_scale_end=None,
    edit_scale_incr=0.0,
    edit_scale_steps=2500,
    residual_xyzg=False,
    discount_per_step=0.97,
    critic_ensemble_size=2,
    critic_subsample_size=None,
    target_entropy=None,
    reward_bias=0.0,
    bc_weight=0.0,
    bc_only=False,
):
    from wrl.utils.launcher import make_batch_augmentation_func

    return ResidualSACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_full_action,
        sample_base_action,
        action_dim=action_dim,
        horizon=horizon,
        edit_scale=edit_scale,
        edit_scale_end=edit_scale_end,
        edit_scale_incr=edit_scale_incr,
        edit_scale_steps=edit_scale_steps,
        residual_xyzg=residual_xyzg,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        discount_per_step=discount_per_step,
        target_entropy=target_entropy,
        reward_bias=reward_bias,
        bc_weight=bc_weight,
        bc_only=bc_only,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
