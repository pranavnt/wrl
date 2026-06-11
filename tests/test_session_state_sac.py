"""Smoke test: the Session core trains a state-only SAC on Pendulum.

Validates the learner thread, buffer routing, UTD pacing, and param publishing
without any pixels / base policy / fleet server. Fast, CPU-only.
"""

import time

import gymnasium as gym
import jax
import jax.numpy as jnp

import wrl
from wrl.utils.launcher import make_sac_state_agent


def test_state_sac_trains():
    env = gym.make("Pendulum-v1")
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()

    agent = make_sac_state_agent(
        seed=0, sample_obs=sample_obs, sample_action=sample_action, discount=0.99
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=64,
        cta_ratio=1,
        training_starts=80,
        replay_buffer_capacity=10_000,
        demo_buffer_capacity=10_000,
        max_steps=200,
    )
    session = wrl.Session(agent, env, cfg, rng_seed=0)
    session.start_learner()

    # Collect enough data to clear `training_starts`, then let the background
    # learner run to `max_steps` (it keeps training on the buffer regardless of
    # new data). The actor naturally outruns the JIT-compiling learner, so we
    # don't bound on env steps — we wait for the learner to finish.
    obs, _ = env.reset(seed=0)
    random_steps = 100
    for env_step in range(400):
        if env_step < random_steps:
            action = env.action_space.sample()
        else:
            action = session.policy.sample(obs)
        next_obs, reward, done, trunc, info = env.step(action)
        session.buffer.add(obs, action, next_obs, reward, done)
        if done or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    deadline = time.monotonic() + 60.0
    while session.status()["learner_running"] and time.monotonic() < deadline:
        time.sleep(0.05)
    session.stop_learner()
    st = session.status()
    assert st["learner_step"] >= cfg.max_steps, st
    assert st["params_version"] >= cfg.max_steps, st
    assert session.buffer.online_size > 0, st
    env.close()
