"""Unit test for ResidualSACAgent (no env, no base server).

Checks: construction on pixel obs, the residual composition contract
(`edit_scale=0` -> full action == base), and that a full update and a
critic-only update produce finite losses with the right keys.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from wrl.agents.residual_sac import ResidualSACAgent, make_residual_sac_pixel_agent

A, H = 7, 2            # 7-dof action, 2-step chunk
CHUNK = A * H          # 14
# Images are single-frame (T=1): the ResNet normalizes for 3-channel RGB, so we
# do NOT stack frames into channels. Temporal context comes from action chunking.
T, IMG, C, PROPRIO = 1, 8, 3, 5
IMAGE_KEYS = ("agentview",)


def _sample_obs():
    return {
        "agentview": np.zeros((T, IMG, IMG, C), np.uint8),
        "state": np.zeros((T, PROPRIO), np.float32),
    }


def _make_agent(seed=0, edit_scale=1.0):
    return make_residual_sac_pixel_agent(
        seed,
        _sample_obs(),
        np.zeros(CHUNK, np.float32),   # sample full action
        np.zeros(CHUNK, np.float32),   # sample base action
        action_dim=A,
        horizon=H,
        image_keys=IMAGE_KEYS,
        encoder_type="resnet",
        edit_scale=edit_scale,
    )


def _random_batch(rng, batch_size=8):
    rng = np.random.default_rng(rng)

    def obs():
        return {
            "agentview": rng.integers(0, 256, (batch_size, T, IMG, IMG, C), np.uint8),
            "state": rng.standard_normal((batch_size, T, PROPRIO)).astype(np.float32),
        }

    base = rng.standard_normal((batch_size, CHUNK)).astype(np.float32)
    next_base = rng.standard_normal((batch_size, CHUNK)).astype(np.float32)
    return frozen_dict.freeze(
        {
            "observations": obs(),
            "next_observations": obs(),  # present -> update skips _unpack
            "actions": base + 0.1,        # full executed chunk
            "base_actions": base,
            "next_base_actions": next_base,
            "rewards": rng.standard_normal(batch_size).astype(np.float32),
            "masks": np.ones(batch_size, np.float32),
            "dones": np.zeros(batch_size, bool),
        }
    )


def test_build_and_sample():
    agent = _make_agent()
    obs = jax.device_put(_sample_obs())
    base = jnp.ones(CHUNK, jnp.float32)
    full = agent.sample_actions(obs, base, seed=jax.random.PRNGKey(0))
    assert full.shape == (CHUNK,)
    assert np.all(np.isfinite(np.asarray(full)))


def test_compose_zero_edit_scale_is_identity():
    agent = _make_agent(edit_scale=0.0)
    obs = jax.device_put(_sample_obs())
    base = jnp.arange(CHUNK, dtype=jnp.float32)
    full = agent.sample_actions(obs, base, seed=jax.random.PRNGKey(1), argmax=True)
    # edit_scale=0 -> residual contributes nothing -> full == base
    assert np.allclose(np.asarray(full), np.asarray(base), atol=1e-5)


def test_full_and_critic_only_update():
    agent = _make_agent()
    batch = _random_batch(0)

    agent2, info = agent.update(batch, networks_to_update=ResidualSACAgent.ALL_NETWORKS)
    # info is nested per network: info['critic']['critic_loss'], etc.
    for net, key in [
        ("critic", "critic_loss"),
        ("actor", "actor_loss"),
        ("actor", "residual_abs_mean"),
        ("temperature", "temperature_loss"),
    ]:
        assert key in info[net], info[net].keys()
        assert np.isfinite(float(info[net][key])), (net, key, info[net][key])

    # critic-only update path (used for cta_ratio > 1)
    agent3, cinfo = agent2.update(
        _random_batch(1), networks_to_update=ResidualSACAgent.CRITIC_NETWORKS
    )
    assert np.isfinite(float(cinfo["critic"]["critic_loss"]))
