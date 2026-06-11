"""End-to-end fleet smoke for the EXPO-FT residual pipeline, no robomimic.

A tiny pixel env + the action-chunk wrapper + a mock (zeros) base-policy server
+ the Session learner + the HTTP server. Exercises: base-policy querying, the
residual buffer schema, chunked residual updates, learner throughput, and
msgpack param sync over HTTP. Proves the whole pipeline without mujoco.
"""

import time

import gymnasium as gym
import httpx
import jax
import jax.numpy as jnp
import numpy as np

import wrl
from wrl.agents.residual_sac import make_residual_sac_pixel_agent
from wrl.base_policy import BasePolicyClient
from wrl.base_policy.mock_server import run_mock_in_thread
from wrl.data import residual_extra_fields
from envs.chunk_wrapper import ActionChunkWrapper

A, H = 4, 2
CHUNK = A * H
IMG, PROPRIO = 8, 5
DISCOUNT = 0.97
MOCK_PORT, HTTP_PORT = 8251, 5599


class DummyPixelEnv(gym.Env):
    """Random image + state obs (single-frame, leading dim 1); reward = -||a||^2."""

    def __init__(self, ep_len=8):
        self.observation_space = gym.spaces.Dict(
            {
                "agentview": gym.spaces.Box(0, 255, (1, IMG, IMG, 3), np.uint8),
                "state": gym.spaces.Box(-10.0, 10.0, (1, PROPRIO), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (A,), np.float32)
        self.ep_len = ep_len
        self._t = 0
        self._rng = np.random.default_rng(0)

    def _obs(self):
        return {
            "agentview": self._rng.integers(0, 256, (1, IMG, IMG, 3), np.uint8),
            "state": self._rng.standard_normal((1, PROPRIO)).astype(np.float32),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        reward = -float(np.mean(np.square(action)))
        done = self._t >= self.ep_len
        return self._obs(), reward, done, False, {}


def test_residual_fleet_pipeline():
    server, _ = run_mock_in_thread("zeros", A, H, port=MOCK_PORT)
    time.sleep(0.2)
    base = BasePolicyClient(port=MOCK_PORT)

    env = ActionChunkWrapper(DummyPixelEnv(), A, H, discount=DISCOUNT)
    sample_obs = env.observation_space.sample()

    agent = make_residual_sac_pixel_agent(
        0, sample_obs, np.zeros(CHUNK, np.float32), np.zeros(CHUNK, np.float32),
        action_dim=A, horizon=H, image_keys=("agentview",), encoder_type="resnet",
        edit_scale=1.0, discount_per_step=DISCOUNT,
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=32, cta_ratio=1, training_starts=20,
        replay_buffer_capacity=5_000, demo_buffer_capacity=5_000,
        max_steps=30, image_keys=("agentview",),
        extra_fields=residual_extra_fields(CHUNK),
    )
    session = wrl.Session(agent, env, cfg, rng_seed=0)
    session.start_learner()
    session.start_server(port=HTTP_PORT)

    try:
        obs, _ = env.reset()
        a_base = base.query(obs)
        for chunk_step in range(120):
            if chunk_step < 12:
                full = env.action_space.sample()       # random-chunk warmup
            else:
                full = session.policy.sample(obs, a_base)
            next_obs, r, done, trunc, info = env.step(full)
            next_a_base = base.query(next_obs)
            session.buffer.add(
                obs, full, next_obs, r, done,
                base_actions=a_base, next_base_actions=next_a_base,
            )
            if done or trunc:
                obs, _ = env.reset()
                a_base = base.query(obs)
            else:
                obs, a_base = next_obs, next_a_base

        # let the learner finish its max_steps updates
        deadline = time.monotonic() + 240.0
        while session.status()["learner_running"] and time.monotonic() < deadline:
            time.sleep(0.1)

        st = session.status()
        assert st["learner_step"] >= cfg.max_steps, st
        assert st["params_version"] >= cfg.max_steps, st
        assert session.buffer.online_size > 0, st

        # HTTP fleet surface round-trips
        time.sleep(0.3)
        r_status = httpx.get(f"http://localhost:{HTTP_PORT}/status", timeout=5.0)
        assert r_status.status_code == 200
        assert r_status.json()["learner_step"] >= cfg.max_steps

        r_params = httpx.get(f"http://localhost:{HTTP_PORT}/params", timeout=10.0)
        assert r_params.status_code == 200
        assert int(r_params.headers["X-Params-Version"]) >= 1
        assert len(r_params.content) > 0
    finally:
        session.stop_learner()
        base.close()
        server.shutdown()
