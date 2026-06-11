"""JAX/Flax flow-matching ("Diffusion Forcing") policy for pixels.

Ported from plantok's PyTorch DP (which gets ~85% on tool-hang low-dim), adapted
to pixel observations:

  * Transformer denoiser with PER-POSITION flow-time: each of the Tp action
    positions carries its own time t_i in [0,1]; a non-causal encoder lets every
    position attend to the observation token and to each other.
  * Rectified flow: x_i(t) = (1-t) a_i + t eps_i, velocity target u = eps - a.
  * Trainable (UNFROZEN) GroupNorm ResNet image encoder per camera, applied to
    each frame of an obs history, end-to-end with the denoiser. GroupNorm keeps
    rollout (batch 1) behavior identical to training.
  * Per-dim mean/std action normalization (fit on data).
  * Cold sampler: re-noise the whole horizon, Euler-integrate Tp positions to 0,
    execute the first Ta -> the EXPO-FT base chunk.

Images normalized in the encoder (/255); no separate obs normalization buffer.
"""

import math
import pickle
from functools import partial
from typing import ClassVar

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from wrl.common.common import nonpytree_field
from wrl.vision.data_augmentations import batched_random_crop


# ---------------------------------------------------------------------------
# trainable GroupNorm ResNet frame encoder (port of plantok FrameEncoder)
# ---------------------------------------------------------------------------


def _gn(c):
    return nn.GroupNorm(num_groups=min(8, c), epsilon=1e-5)


class _BasicBlock(nn.Module):
    cout: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        cin = x.shape[-1]
        h = nn.Conv(self.cout, (3, 3), strides=self.stride, padding=1, use_bias=False)(x)
        h = nn.relu(_gn(self.cout)(h))
        h = nn.Conv(self.cout, (3, 3), padding=1, use_bias=False)(h)
        h = _gn(self.cout)(h)
        s = x
        if self.stride != 1 or cin != self.cout:
            s = nn.Conv(self.cout, (1, 1), strides=self.stride, use_bias=False)(x)
            s = _gn(self.cout)(s)
        return nn.relu(h + s)


class FrameEncoder(nn.Module):
    """(N, H, W, 3) uint8/float frame -> (N, d_model)."""

    d_model: int
    channels: tuple = (32, 64, 128, 256)

    @nn.compact
    def __call__(self, frames):
        x = frames.astype(jnp.float32) / 255.0
        x = nn.Conv(self.channels[0], (5, 5), strides=2, padding=2, use_bias=False)(x)
        x = nn.relu(_gn(self.channels[0])(x))
        for i, c in enumerate(self.channels):
            x = _BasicBlock(c, stride=1 if i == 0 else 2)(x)
            x = _BasicBlock(c)(x)
        x = jnp.mean(x, axis=(-3, -2))  # global average pool -> (N, C)
        return nn.Dense(self.d_model)(x)


# ---------------------------------------------------------------------------
# transformer denoiser
# ---------------------------------------------------------------------------


def _sinusoidal_time(t, d):
    """t: (...) in [0,1] -> (..., d)."""
    half = d // 2
    freqs = jnp.exp(-math.log(10000.0) * jnp.arange(half) / (half - 1))
    args = t[..., None] * freqs * 1000.0
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class _EncoderLayer(nn.Module):
    d: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(num_heads=self.n_heads, qkv_features=self.d)(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.d)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d)(h)
        return x + h


class FlowDenoiser(nn.Module):
    d_a: int
    Tp: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    image_keys: tuple = ("agentview_image",)
    use_proprio: bool = True

    @nn.compact
    def __call__(self, x, t, observations, train: bool = False):
        """x: (B,Tp,d_a) noised normalized actions; t: (B,Tp) per-position time;
        observations: dict with each image key (B, T_hist, H, W, C) and optional
        'state' (B, T_hist, proprio). Returns velocity (B,Tp,d_a)."""
        d = self.d_model

        # ---- encode observation -> single conditioning token --------------
        parts = []
        for k in self.image_keys:
            imgs = observations[k]  # (B, T, H, W, C)
            B, T = imgs.shape[0], imgs.shape[1]
            flat = imgs.reshape((B * T,) + imgs.shape[2:])
            emb = FrameEncoder(d_model=d, name=f"enc_{k}")(flat)  # (B*T, d)
            parts.append(emb.reshape(B, T * d))
        if self.use_proprio and "state" in observations:
            s = observations["state"]
            parts.append(s.reshape(s.shape[0], -1))
        obs_cat = jnp.concatenate(parts, axis=-1)
        obs_emb = nn.Dense(d)(nn.silu(nn.Dense(d)(obs_cat)))  # obs_proj
        cond_marker = self.param("cond_marker", nn.initializers.normal(0.02), (d,))
        cond = (obs_emb + cond_marker)[:, None, :]  # (B,1,d)

        # ---- action tokens ------------------------------------------------
        pos = self.param("pos_embed", nn.initializers.normal(0.02), (self.Tp, d))
        t_emb = nn.Dense(d)(nn.silu(nn.Dense(d)(_sinusoidal_time(t, d))))  # (B,Tp,d)
        h = nn.Dense(d)(x) + pos[None] + t_emb  # (B,Tp,d)

        seq = jnp.concatenate([cond, h], axis=1)  # (B,1+Tp,d)
        for _ in range(self.n_layers):
            seq = _EncoderLayer(d, self.n_heads)(seq)
        return nn.Dense(self.d_a)(seq[:, 1:])  # (B,Tp,d_a)


# ---------------------------------------------------------------------------
# flow policy container
# ---------------------------------------------------------------------------


class FlowPolicy(flax.struct.PyTreeNode):
    state: TrainState
    ema_params: any
    a_mean: jnp.ndarray
    a_std: jnp.ndarray
    rng: jax.Array
    config: dict = nonpytree_field()

    # ---- normalization --------------------------------------------------
    def _norm(self, a):
        return (a - self.a_mean) / self.a_std

    def _denorm(self, a):
        return a * self.a_std + self.a_mean

    def with_action_stats(self, mean, std):
        return self.replace(
            a_mean=jnp.asarray(mean, jnp.float32),
            a_std=jnp.maximum(jnp.asarray(std, jnp.float32), 1e-3),
        )

    # ---- training -------------------------------------------------------
    def _loss_fn(self, params, batch, rng):
        a = self._norm(batch["actions"])  # (B,Tp,d_a)
        B, Tp = a.shape[0], a.shape[1]
        eps_rng, t_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, a.shape)
        t = jax.random.uniform(t_rng, (B, Tp))
        x = (1 - t)[..., None] * a + t[..., None] * eps
        target = eps - a
        pred = self.state.apply_fn({"params": params}, x, t, batch["observations"], train=True)
        loss = jnp.mean((pred - target) ** 2)
        return loss, {"flow_loss": loss}

    def _augment(self, observations, rng):
        """Random-crop (pad+crop shift) each image key — the key pixel-DP
        regularizer. Frames are (B, T, H, W, C); crop with num_batch_dims=2."""
        obs = dict(observations)
        for k in self.config["image_keys"]:
            rng, ck = jax.random.split(rng)
            obs[k] = batched_random_crop(obs[k], ck, padding=4, num_batch_dims=2)
        return obs

    @jax.jit
    def update(self, batch):
        rng, step_rng, aug_rng = jax.random.split(self.rng, 3)
        obs = self._augment(batch["observations"], aug_rng)
        batch = {**batch, "observations": obs}
        (loss, info), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(
            self.state.params, batch, step_rng
        )
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        new_state = self.state.apply_gradients(grads=grads)
        decay = self.config["ema_decay"]
        new_ema = jax.tree_util.tree_map(
            lambda e, p: decay * e + (1.0 - decay) * p, self.ema_params, new_state.params
        )
        return self.replace(state=new_state, ema_params=new_ema, rng=rng), info

    # ---- cold sampling --------------------------------------------------
    @partial(jax.jit, static_argnames=())
    def sample_chunk(self, observations, rng):
        """Cold Euler sampling -> first Ta actions flattened to (Ta*d_a,).
        observations: unbatched dict (image (T,H,W,C), state (T,proprio))."""
        cfg = self.config
        Tp, Ta, d_a = cfg["Tp"], cfg["Ta"], cfg["d_a"]
        obs = jax.tree_util.tree_map(lambda v: v[None], observations)  # add batch
        n = cfg["n_sample_steps"]
        x = jax.random.normal(rng, (1, Tp, d_a))
        t0 = jnp.ones((1, Tp))
        dt = t0 / n
        for k in range(n):
            t = t0 * (1 - k / n)
            # sample with the EMA weights (standard for diffusion policies)
            v = self.state.apply_fn({"params": self.ema_params}, x, t, obs, train=False)
            x = x - v * dt[..., None]
        a = self._denorm(x)[0]  # (Tp, d_a)
        return a[:Ta].reshape(Ta * d_a)

    # ---- construction ---------------------------------------------------
    @classmethod
    def create(
        cls,
        rng,
        sample_obs,
        d_a: int,
        *,
        Tp: int = 16,
        Ta: int = 8,
        image_keys=("agentview_image",),
        use_proprio: bool = True,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_sample_steps: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.9999,
    ):
        net = FlowDenoiser(
            d_a=d_a, Tp=Tp, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            image_keys=tuple(image_keys), use_proprio=use_proprio,
        )
        rng, init_rng, state_rng = jax.random.split(rng, 3)
        obs_b = jax.tree_util.tree_map(lambda v: jnp.asarray(v)[None], sample_obs)
        x0 = jnp.zeros((1, Tp, d_a))
        t0 = jnp.zeros((1, Tp))
        params = net.init(init_rng, x0, t0, obs_b, train=False)["params"]
        state = TrainState.create(
            apply_fn=net.apply, params=params,
            tx=optax.adamw(learning_rate, weight_decay=weight_decay),
        )
        img0 = np.asarray(sample_obs[image_keys[0]])
        return cls(
            state=state,
            ema_params=jax.tree_util.tree_map(jnp.array, params),
            a_mean=jnp.zeros(d_a),
            a_std=jnp.ones(d_a),
            rng=state_rng,
            config=dict(
                d_a=d_a, Tp=Tp, Ta=Ta, image_keys=tuple(image_keys),
                use_proprio=use_proprio, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, n_sample_steps=n_sample_steps, ema_decay=ema_decay,
                image_shape=tuple(int(s) for s in img0.shape),
                proprio_dim=int(np.asarray(sample_obs["state"]).shape[-1]) if use_proprio else 0,
            ),
        )

    # ---- checkpointing --------------------------------------------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "params": jax.tree_util.tree_map(np.asarray, self.state.params),
                "ema_params": jax.tree_util.tree_map(np.asarray, self.ema_params),
                "a_mean": np.asarray(self.a_mean), "a_std": np.asarray(self.a_std),
                "config": self.config,
            }, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        cfg = blob["config"]
        T = cfg["image_shape"][0]
        obs = {k: np.zeros(cfg["image_shape"], np.uint8) for k in cfg["image_keys"]}
        if cfg["use_proprio"]:
            obs["state"] = np.zeros((T, cfg["proprio_dim"]), np.float32)
        fp = cls.create(
            jax.random.PRNGKey(0), obs, cfg["d_a"], Tp=cfg["Tp"], Ta=cfg["Ta"],
            image_keys=cfg["image_keys"], use_proprio=cfg["use_proprio"],
            d_model=cfg["d_model"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
            n_sample_steps=cfg["n_sample_steps"],
        )
        params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
        ema = jax.tree_util.tree_map(jnp.asarray, blob.get("ema_params", blob["params"]))
        return fp.replace(
            state=fp.state.replace(params=params), ema_params=ema,
            a_mean=jnp.asarray(blob["a_mean"]), a_std=jnp.asarray(blob["a_std"]),
        )
