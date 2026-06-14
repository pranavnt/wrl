"""Faithful DSRL agent (https://arxiv.org/abs/2506.15799), adapted to a FROZEN,
EXTERNAL, pixel-conditioned base diffusion policy with a STATE-conditioned
steering policy.

Two critics (the reference's key design):
  * action critic  Q_a(s, a)  over the DECODED action chunk a = f_DP(s, w).
    Trained by a Bellman backup. Because our decoder is an external pixel model
    we cannot decode inside the jitted loss, so we use the SARSA target with the
    stored next decoded action `a'` (one decode per env step, carried forward —
    same trick as the residual base_actions). Pessimistic backup: mean - rho*std.
  * latent critic  Q_z(s, w)  over the noise w. DISTILLED from the action critic
    on the collected (w, a) pairs:  Q_z(s, w) <- stopgrad Q_a(s, a).  This gives a
    well-calibrated latent value so best-of-n / the actor steer correctly (the
    direct latent-Bellman version was miscalibrated and went *below* base).

The actor (TanhNormal over w) maximizes Q_z; best-of-n scores candidate latents
with Q_z (no decode). The base DP weights never change.

Buffer batch keys: observations, actions (= latent w), decoded_actions (= a),
next_observations, next_decoded_actions (= a'), rewards, masks.
"""

from functools import partial
from typing import ClassVar, FrozenSet, Optional

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from wrl.common.common import ModuleDict, nonpytree_field
from wrl.common.optimizers import make_optimizer
from wrl.common.typing import Data, Params, PRNGKey
from wrl.networks.actor_critic_nets import Critic, Policy, ensemblize
from wrl.networks.lagrange import GeqLagrangeMultiplier
from wrl.networks.mlp import MLP


class DSRLAgent(flax.struct.PyTreeNode):
    state: TrainState
    target_params: Params
    rng: PRNGKey
    config: dict = nonpytree_field()

    # Empty CRITIC set -> Session does ONE combined update per train_step (our
    # total_loss already covers all four networks). ALL_NETWORKS is informational.
    CRITIC_NETWORKS: ClassVar[FrozenSet[str]] = frozenset()
    ALL_NETWORKS: ClassVar[FrozenSet[str]] = frozenset(
        {"actor", "critic", "z_critic", "temperature"}
    )

    # ---- forward helpers -------------------------------------------------
    def _apply(self, params, *args, name, **kw):
        return self.state.apply_fn({"params": params}, *args, name=name, **kw)

    def actor_dist(self, obs, params=None) -> distrax.Distribution:
        return self._apply(params or self.state.params, obs, name="actor")

    def q_action(self, obs, a, params=None):
        return self._apply(params or self.state.params, obs, a, name="critic")

    def q_action_target(self, obs, a):
        return self._apply(self.target_params, obs, a, name="critic")

    def q_latent(self, obs, w, params=None):
        return self._apply(params or self.state.params, obs, w, name="z_critic")

    def temperature(self, params=None):
        return self._apply(params or self.state.params, name="temperature")

    # ---- combined loss ---------------------------------------------------
    def total_loss(self, batch, params, rng):
        info = {}
        rho = self.config["rho"]
        ns = self.config["noise_scale"]
        disc = self.config["discount"]

        # --- action critic: SARSA backup with stored next decoded action ---
        next_qs = self.q_action_target(batch["next_observations"], batch["next_decoded_actions"])
        next_q = next_qs.mean(axis=0) - rho * next_qs.std(axis=0)   # pessimistic
        target_q = batch["rewards"] + disc * batch["masks"] * next_q
        target_q = jax.lax.stop_gradient(target_q)
        q = self.q_action(batch["observations"], batch["decoded_actions"], params)
        critic_loss = jnp.mean((q - target_q[None]) ** 2)

        # --- latent critic: distill from action critic on collected (w, a) ---
        q_a_target = jax.lax.stop_gradient(
            self.q_action(batch["observations"], batch["decoded_actions"]).mean(axis=0)
        )
        qz = self.q_latent(batch["observations"], batch["actions"], params)  # (ens, B)
        distill_loss = jnp.mean((qz - q_a_target[None]) ** 2)

        # --- actor: maximize Q_z on fresh actor noise ---
        rng, akey = jax.random.split(rng)
        dist = self.actor_dist(batch["observations"], params)
        w, log_probs = dist.sample_and_log_prob(seed=akey)
        w = w * ns
        qz_pi = self.q_latent(batch["observations"], w).mean(axis=0)
        alpha = jax.lax.stop_gradient(self.temperature(params))
        actor_loss = jnp.mean(alpha * log_probs - qz_pi)

        # --- temperature (auto entropy tuning via the Geq lagrange penalty) ---
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        temp_loss = self._apply(
            params, lhs=entropy, rhs=self.config["target_entropy"], name="temperature"
        )

        total = critic_loss + distill_loss + actor_loss + temp_loss
        info.update(
            critic_loss=critic_loss, distill_loss=distill_loss, actor_loss=actor_loss,
            temp_loss=temp_loss, q_action=q.mean(), target_q=target_q.mean(),
            q_latent=qz.mean(), entropy=-log_probs.mean(), alpha=alpha,
        )
        return total, info

    @partial(jax.jit, static_argnames=("networks_to_update",))
    def update(self, batch, networks_to_update=None):
        # networks_to_update is accepted for Session compatibility but ignored:
        # total_loss already updates all four networks in one combined step.
        new_rng, rng = jax.random.split(self.rng)
        grads, info = jax.grad(
            lambda p: self.total_loss(batch, p, rng), has_aux=True
        )(self.state.params)
        new_state = self.state.apply_gradients(grads=grads)
        tau = self.config["soft_target_update_rate"]
        new_target = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_state.params["modules_critic"], self.target_params["modules_critic"],
        )
        target_params = {**self.target_params, "modules_critic": new_target}
        return self.replace(state=new_state, target_params=target_params, rng=new_rng), info

    # ---- action selection ------------------------------------------------
    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(self, observations, *, seed=None, argmax=False, **kw):
        dist = self.actor_dist(observations)
        a = dist.mode() if argmax else dist.sample(seed=seed)
        return a * self.config["noise_scale"]

    @partial(jax.jit, static_argnames=("n",))
    def sample_best_of_n(self, observations, n: int, seed: PRNGKey):
        """Sample n latents, score with the (distilled, calibrated) z_critic,
        return the best. observations is a single unbatched state vector."""
        obs_n = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None], (n,) + x.shape), observations
        )
        dist = self.actor_dist(obs_n)
        w = dist.sample(seed=seed) * self.config["noise_scale"]
        qz = self.q_latent(obs_n, w)                       # (ens, n)
        score = qz.mean(axis=0) - self.config["rho"] * qz.std(axis=0)
        return w[jnp.argmax(score)]

    # ---- construction ----------------------------------------------------
    @classmethod
    def create(
        cls,
        seed: int,
        sample_obs,                # state vector (state_dim,)
        latent_dim: int,           # w dim (Tp*d_a)
        decoded_dim: int,          # a dim (Ta*d_a)
        *,
        hidden_dims=(256, 256, 256),
        discount: float = 0.99,
        noise_scale: float = 1.0,
        rho: float = 0.5,
        soft_target_update_rate: float = 0.005,
        num_qs: int = 10,
        target_entropy: Optional[float] = None,
        lr: float = 3e-4,
    ):
        rng = jax.random.PRNGKey(seed)
        ob = jnp.asarray(sample_obs)[None]
        w0 = jnp.zeros((1, latent_dim))
        a0 = jnp.zeros((1, decoded_dim))

        def mlp_critic():
            net = ensemblize(
                partial(MLP, hidden_dims=list(hidden_dims), activations=nn.relu,
                        use_layer_norm=True, activate_final=True),
                num_qs,
            )(name="ens")
            return Critic(encoder=None, network=net)

        actor_def = Policy(
            encoder=None,
            network=MLP(hidden_dims=list(hidden_dims), activations=nn.relu,
                        use_layer_norm=False, activate_final=True),
            action_dim=latent_dim, tanh_squash_distribution=True,
            std_parameterization="exp", std_min=1e-5, std_max=5.0, name="actor",
        )
        networks = {
            "actor": actor_def,
            "critic": mlp_critic(),     # Q_a(s, a)
            "z_critic": mlp_critic(),   # Q_z(s, w)
            "temperature": GeqLagrangeMultiplier(
                init_value=1.0, constraint_shape=(), name="temperature"),
        }
        model_def = ModuleDict(networks)
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng, actor=[ob], critic=[ob, a0], z_critic=[ob, w0], temperature=[],
        )["params"]
        state = TrainState.create(
            apply_fn=model_def.apply, params=params, tx=make_optimizer(learning_rate=lr),
        )
        if target_entropy is None:
            target_entropy = -latent_dim / 2
        rng, agent_rng = jax.random.split(rng)
        return cls(
            state=state,
            target_params=params,
            rng=agent_rng,
            config=dict(
                discount=discount, noise_scale=noise_scale, rho=rho,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=float(target_entropy), latent_dim=latent_dim,
                decoded_dim=decoded_dim,
            ),
        )
