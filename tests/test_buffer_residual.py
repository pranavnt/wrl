"""Verify residual-RL fields (base_actions / next_base_actions) round-trip
through both buffers, including the pixel buffer's frame-stacking path.

The key invariant: base/next-base chunks are ordinary per-index columns and
must NOT be perturbed by pixel frame-stacking. We construct each transition so
that `next_base_actions == base_actions + 1`; any validly sampled row must
preserve that relation regardless of stacking.
"""

import gymnasium as gym
import numpy as np

from wrl.data import (
    MemoryEfficientReplayBufferDataStore,
    ReplayBufferDataStore,
    residual_extra_fields,
)

A, H = 2, 3
CHUNK = A * H  # flattened base-chunk dim
NUM_STACK = 2


def _state_obs_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)


def _pixel_obs_space():
    return gym.spaces.Dict(
        {
            "pix": gym.spaces.Box(0, 255, shape=(NUM_STACK, 4, 4, 3), dtype=np.uint8),
            "state": gym.spaces.Box(-1.0, 1.0, shape=(NUM_STACK, 4), dtype=np.float32),
        }
    )


def _action_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(CHUNK,), dtype=np.float32)


def _transition(t, obs, next_obs, done=False):
    """base = t, next_base = t+1, full action = base + 0.5 (all entries)."""
    return dict(
        observations=obs,
        next_observations=next_obs,
        actions=np.full(CHUNK, t + 0.5, np.float32),
        rewards=np.float32(0.1 * t),
        masks=np.float32(0.0 if done else 1.0),
        dones=bool(done),
        is_intervention=False,
        base_actions=np.full(CHUNK, float(t), np.float32),
        next_base_actions=np.full(CHUNK, float(t + 1), np.float32),
    )


def test_state_buffer_residual_roundtrip():
    buf = ReplayBufferDataStore(
        _state_obs_space(), _action_space(), capacity=200,
        extra_fields=residual_extra_fields(CHUNK),
    )
    for t in range(60):
        obs = np.full(4, 0.01 * t, np.float32)
        nobs = np.full(4, 0.01 * (t + 1), np.float32)
        buf.insert(_transition(t, obs, nobs))

    batch = buf.sample(32)
    base = np.asarray(batch["base_actions"])
    nbase = np.asarray(batch["next_base_actions"])
    acts = np.asarray(batch["actions"])
    assert base.shape == (32, CHUNK)
    assert np.allclose(nbase, base + 1.0)
    assert np.allclose(acts, base + 0.5)


def test_pixel_buffer_residual_survives_framestacking():
    buf = MemoryEfficientReplayBufferDataStore(
        _pixel_obs_space(), _action_space(), capacity=300, image_keys=("pix",),
        extra_fields=residual_extra_fields(CHUNK),
    )
    rng = np.random.default_rng(0)
    for t in range(80):
        obs = {
            "pix": rng.integers(0, 256, (NUM_STACK, 4, 4, 3), dtype=np.uint8),
            "state": np.full((NUM_STACK, 4), 0.01 * t, np.float32),
        }
        nobs = {
            "pix": rng.integers(0, 256, (NUM_STACK, 4, 4, 3), dtype=np.uint8),
            "state": np.full((NUM_STACK, 4), 0.01 * (t + 1), np.float32),
        }
        buf.insert(_transition(t, obs, nobs, done=(t % 20 == 19)))

    batch = buf.sample(24, pack_obs_and_next_obs=False)
    base = np.asarray(batch["base_actions"])
    nbase = np.asarray(batch["next_base_actions"])
    acts = np.asarray(batch["actions"])

    # residual fields preserved per-row, untouched by stacking
    assert base.shape == (24, CHUNK)
    assert np.allclose(nbase, base + 1.0), "next_base misaligned under stacking"
    assert np.allclose(acts, base + 0.5)

    # pixels are frame-stacked into (B, T, H, W, C)
    assert batch["observations"]["pix"].shape == (24, NUM_STACK, 4, 4, 3)
    assert batch["next_observations"]["pix"].shape == (24, NUM_STACK, 4, 4, 3)
