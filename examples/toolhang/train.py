"""Tool-hang training: EXPO-FT residual RL or chunked RL-from-scratch.

The two modes share everything but where the base action chunk comes from:

  --base dp     EXPO-FT: query a running DP base-policy server (start it with
                `python -m wrl.base_policy.dp_server --checkpoint ...`).
  --base zeros  chunked RL from scratch: base chunk is zero, so the residual
                policy *is* the policy.

Example (EXPO-FT):
    python -m wrl.base_policy.dp_server --checkpoint checkpoints/dp_toolhang.pkl --port 8200 &
    python examples/toolhang/train.py --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --base dp --dp-port 8200 --horizon 8 --warmstart-demos
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wrl
from envs.chunk_wrapper import ActionChunkWrapper
from envs.dataset_loader import load_robomimic_pixels
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.agents.residual_sac import make_residual_sac_pixel_agent
from wrl.base_policy import BasePolicyClient
from wrl.data import residual_extra_fields


def main(
    dataset_path: str,
    base: str = "dp",               # "dp" | "zeros"
    dp_host: str = "localhost",
    dp_port: int = 8200,
    horizon: int = 8,
    edit_scale: float = 0.25,
    discount: float = 0.99,
    image_size: int = 84,
    max_episode_steps: int = 700,
    batch_size: int = 256,
    cta_ratio: int = 4,
    training_starts: int = 1_000,
    random_chunks: int = 50,
    max_steps: int = 200_000,
    min_utd: float = 0.0,
    warmstart_demos: bool = False,
    warmstart_max: int = 5_000,
    http_port: int = 5588,
    encoder_type: str = "resnet",
    seed: int = 0,
    smoke: bool = False,
):
    if smoke:
        base, max_steps, training_starts, random_chunks = "zeros", 30, 20, 8
        cta_ratio, max_episode_steps = 1, 60

    env = make_robomimic_pixel_env(
        dataset_path, image_size=image_size, max_episode_steps=max_episode_steps
    )
    A = env.action_space.shape[0]
    chunk_dim = A * horizon
    cenv = ActionChunkWrapper(env, A, horizon, discount=discount)

    sample_obs = env.observation_space.sample()
    agent = make_residual_sac_pixel_agent(
        seed, sample_obs, np.zeros(chunk_dim, np.float32), np.zeros(chunk_dim, np.float32),
        action_dim=A, horizon=horizon, image_keys=env.image_keys,
        encoder_type=encoder_type, edit_scale=edit_scale, discount_per_step=discount,
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=200_000, demo_buffer_capacity=200_000,
        max_steps=max_steps, image_keys=env.image_keys,
        extra_fields=residual_extra_fields(chunk_dim),
    )
    session = wrl.Session(agent, cenv, cfg, rng_seed=seed)

    if base == "dp":
        client = BasePolicyClient(host=dp_host, port=dp_port)
        query_base = client.query
    elif base == "zeros":
        zeros = np.zeros(chunk_dim, np.float32)
        query_base = lambda _obs: zeros  # noqa: E731
    else:
        raise ValueError(f"--base must be 'dp' or 'zeros', got {base!r}")

    if warmstart_demos:
        data = load_robomimic_pixels(dataset_path, env.image_keys, env.proprio_keys, horizon)
        demos = data.residual_transitions(discount, max_transitions=warmstart_max, seed=seed)
        n = session.preload_demos(demos)
        print(f"[actor] warm-started demo buffer with {n} residual transitions")

    session.start_learner()
    session.start_server(port=http_port)
    print(f"[actor] base={base} edit_scale={edit_scale} chunk_dim={chunk_dim} "
          f"action_dim={A} horizon={horizon}")

    obs, _ = cenv.reset(seed=seed)
    a_base = np.asarray(query_base(obs), np.float32)
    assert a_base.shape == (chunk_dim,), (a_base.shape, chunk_dim)

    chunk_step = 0
    ep_ret, ep_success = 0.0, 0.0
    last_log = time.time()
    try:
        while session.status()["learner_running"]:
            if chunk_step < random_chunks:
                full = cenv.action_space.sample()
            else:
                full = session.policy.sample(obs, a_base)
            next_obs, r, done, trunc, info = cenv.step(full)
            next_a_base = np.asarray(query_base(next_obs), np.float32)
            session.buffer.add(
                obs, full, next_obs, r, done,
                base_actions=a_base, next_base_actions=next_a_base,
            )
            ep_ret += r
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            chunk_step += 1

            if done or trunc:
                session.record_episode(ep_ret)
                st = session.status()
                print(f"[actor] ep_ret={ep_ret:.2f} success={ep_success:.0f} "
                      f"chunks={chunk_step} learner_step={st['learner_step']} "
                      f"utd={st['effective_utd']:.2f} online={st['online_buffer']}")
                if min_utd > 0 and st["learner_step"] > 0:
                    session.wait_for_utd(min_utd)
                obs, _ = cenv.reset()
                a_base = np.asarray(query_base(obs), np.float32)
                ep_ret, ep_success = 0.0, 0.0
            else:
                obs, a_base = next_obs, next_a_base

            if time.time() - last_log > 30:
                last_log = time.time()
                print(f"[actor] heartbeat {session.status()}")
    finally:
        session.stop_learner()
        cenv.close()


if __name__ == "__main__":
    tyro.cli(main)
