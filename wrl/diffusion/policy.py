"""A small JAX/Flax diffusion policy (DDPM train, DDIM sample).

Conditioned on a single observation (images + proprio, via the same
EncodingWrapper the RL agents use), predicts a flattened `H`-step action chunk
`(H*A,)`. Trained by behavior cloning with the standard noise-prediction loss;
served as the frozen base policy for EXPO-FT residual RL.

Actions are assumed pre-normalized to ~[-1, 1] (robosuite OSC actions already
are), so no separate action normalization is applied; samples are clipped to
[-1, 1].
"""

from typing import ClassVar

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from wrl.common.common import nonpytree_field
from wrl.common.encoding import EncodingWrapper
from wrl.vision.resnet_v1 import resnetv1_configs


def _sinusoidal_embedding(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / (half - 1))
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class DiffusionPolicyNet(nn.Module):
    encoder: nn.Module
    action_dim: int
    horizon: int
    hidden_dims: tuple = (512, 512, 512)
    time_dim: int = 64

    @nn.compact
    def __call__(self, observations, noisy_actions, time, train: bool = False):
        obs_emb = self.encoder(observations, train=train, stop_gradient=False)
        t_emb = _sinusoidal_embedding(time, self.time_dim)
        t_emb = nn.Dense(self.time_dim)(nn.swish(nn.Dense(self.time_dim)(t_emb)))

        x = jnp.concatenate([noisy_actions, obs_emb, t_emb], axis=-1)
        for h in self.hidden_dims:
            x = nn.swish(nn.Dense(h)(x))
            x = nn.LayerNorm()(x)
        return nn.Dense(self.horizon * self.action_dim)(x)


def _linear_beta_schedule(num_timesteps: int) -> np.ndarray:
    return np.linspace(1e-4, 0.02, num_timesteps, dtype=np.float32)


class DiffusionPolicy(flax.struct.PyTreeNode):
    state: TrainState
    alphas_cumprod: jnp.ndarray  # (T,)
    rng: jax.Array
    config: dict = nonpytree_field()

    ACTION_LOW: ClassVar[float] = -1.0
    ACTION_HIGH: ClassVar[float] = 1.0

    # ---- training -------------------------------------------------------

    def _loss_fn(self, params, batch, rng):
        obs = batch["observations"]
        x0 = batch["actions"]  # (B, H*A)
        bs = x0.shape[0]
        T = self.config["num_train_timesteps"]

        t_rng, noise_rng, drop_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (bs,), 0, T)
        noise = jax.random.normal(noise_rng, x0.shape)
        abar = self.alphas_cumprod[t][:, None]
        x_t = jnp.sqrt(abar) * x0 + jnp.sqrt(1.0 - abar) * noise

        pred = self.state.apply_fn(
            {"params": params}, obs, x_t, t, train=True, rngs={"dropout": drop_rng},
        )
        loss = jnp.mean((pred - noise) ** 2)
        return loss, {"diffusion_loss": loss}

    @jax.jit
    def update(self, batch):
        rng, step_rng = jax.random.split(self.rng)
        grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
        (loss, info), grads = grad_fn(self.state.params, batch, step_rng)
        new_state = self.state.apply_gradients(grads=grads)
        return self.replace(state=new_state, rng=rng), info

    # ---- sampling (DDIM, deterministic) ---------------------------------

    @jax.jit
    def sample(self, observations, rng):
        """Return a flattened action chunk `(B, H*A)` (or `(H*A,)` if obs is
        unbatched). DDIM with eta=0."""
        cfg = self.config
        A, H = cfg["action_dim"], cfg["horizon"]
        dim = A * H

        img = observations[cfg["image_keys"][0]]
        unbatched = img.ndim == 4  # (T,H,W,C) -> single obs
        if unbatched:
            observations = jax.tree_util.tree_map(lambda x: x[None], observations)
            bs = 1
        else:
            bs = img.shape[0]

        seq = cfg["infer_timesteps"]  # tuple, descending ints
        x = jax.random.normal(rng, (bs, dim))
        for i, t in enumerate(seq):
            t_arr = jnp.full((bs,), t, jnp.int32)
            eps = self.state.apply_fn(
                {"params": self.state.params}, observations, x, t_arr, train=False
            )
            abar_t = self.alphas_cumprod[t]
            x0 = (x - jnp.sqrt(1.0 - abar_t) * eps) / jnp.sqrt(abar_t)
            x0 = jnp.clip(x0, self.ACTION_LOW, self.ACTION_HIGH)
            t_prev = seq[i + 1] if i + 1 < len(seq) else -1
            abar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else jnp.float32(1.0)
            x = jnp.sqrt(abar_prev) * x0 + jnp.sqrt(1.0 - abar_prev) * eps

        x = jnp.clip(x, self.ACTION_LOW, self.ACTION_HIGH)
        return x[0] if unbatched else x

    # ---- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        rng,
        sample_obs,
        action_dim: int,
        horizon: int,
        *,
        image_keys=("agentview_image",),
        use_proprio: bool = True,
        hidden_dims=(512, 512, 512),
        num_train_timesteps: int = 100,
        num_infer_steps: int = 16,
        learning_rate: float = 1e-4,
    ):
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
        encoder_def = EncodingWrapper(
            encoder=encoders, use_proprio=use_proprio,
            enable_stacking=True, image_keys=image_keys,
        )
        net = DiffusionPolicyNet(
            encoder=encoder_def, action_dim=action_dim, horizon=horizon,
            hidden_dims=tuple(hidden_dims),
        )

        rng, init_rng, state_rng = jax.random.split(rng, 3)
        sample_chunk = jnp.zeros((1, action_dim * horizon), jnp.float32)
        sample_t = jnp.zeros((1,), jnp.int32)
        sample_obs_b = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None], sample_obs)
        params = net.init(init_rng, sample_obs_b, sample_chunk, sample_t, train=False)["params"]

        state = TrainState.create(
            apply_fn=net.apply, params=params, tx=optax.adam(learning_rate),
        )

        betas = _linear_beta_schedule(num_train_timesteps)
        alphas_cumprod = jnp.asarray(np.cumprod(1.0 - betas))
        infer = np.round(np.linspace(num_train_timesteps - 1, 0, num_infer_steps))
        infer_timesteps = tuple(int(x) for x in infer.astype(int))

        img0 = np.asarray(sample_obs[image_keys[0]])
        return cls(
            state=state,
            alphas_cumprod=alphas_cumprod,
            rng=state_rng,
            config=dict(
                action_dim=action_dim,
                horizon=horizon,
                image_keys=tuple(image_keys),
                use_proprio=use_proprio,
                hidden_dims=tuple(hidden_dims),
                num_train_timesteps=num_train_timesteps,
                num_infer_steps=num_infer_steps,
                infer_timesteps=infer_timesteps,
                # obs shapes for checkpoint reconstruction
                image_shape=tuple(int(s) for s in img0.shape),  # (1,H,W,C)
                proprio_dim=int(np.asarray(sample_obs["state"]).shape[-1]),
            ),
        )

    # ---- checkpointing --------------------------------------------------

    def _sample_obs_from_config(self):
        cfg = self.config
        obs = {k: np.zeros(cfg["image_shape"], np.uint8) for k in cfg["image_keys"]}
        obs["state"] = np.zeros((1, cfg["proprio_dim"]), np.float32)
        return obs

    def save(self, path: str):
        import pickle

        params = jax.tree_util.tree_map(np.asarray, self.state.params)
        with open(path, "wb") as f:
            pickle.dump({"params": params, "config": self.config}, f)

    @classmethod
    def load(cls, path: str):
        import pickle

        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = blob["config"]
        obs = {k: np.zeros(cfg["image_shape"], np.uint8) for k in cfg["image_keys"]}
        obs["state"] = np.zeros((1, cfg["proprio_dim"]), np.float32)
        dp = cls.create(
            jax.random.PRNGKey(0), obs, cfg["action_dim"], cfg["horizon"],
            image_keys=cfg["image_keys"], use_proprio=cfg["use_proprio"],
            hidden_dims=cfg["hidden_dims"], num_train_timesteps=cfg["num_train_timesteps"],
            num_infer_steps=cfg["num_infer_steps"],
        )
        params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
        return dp.replace(state=dp.state.replace(params=params))
