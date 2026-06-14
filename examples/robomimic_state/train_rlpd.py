"""RLPD from scratch on a low-dim robomimic task (the Q-OIL expert recipe).

SAC + 50/50 demo/online sampling (RLPD), high UTD, sparse reward. Trains an
expert policy + critic to drive Q-OIL interventions/gating.

    uv run python examples/robomimic_state/train_rlpd.py \
        --dataset-path data/robomimic/tool_hang/ph/low_dim_v141.hdf5 \
        --cta-ratio 20 --max-steps 200000 --wandb-project wrl-rlpd
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wrl
from envs.robomimic_state import RoboMimicStateEnv
from wrl.utils.launcher import make_sac_state_agent


def save_agent(agent, path, sample_obs, sample_action, discount):
    """Pickle the RLPD expert (policy + critic + target critic params + shapes
    so it can be rebuilt with make_sac_state_agent + load)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    blob = {
        "params": jax.tree_util.tree_map(np.asarray, agent.state.params),
        "target_params": jax.tree_util.tree_map(np.asarray, agent.state.target_params),
        "obs_dim": int(np.asarray(sample_obs).shape[-1]),
        "action_dim": int(np.asarray(sample_action).shape[-1]),
        "discount": discount,
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    os.replace(tmp, path)  # atomic


def main(
    dataset_path: str,
    discount: float = 0.99,
    batch_size: int = 256,
    bc_weight: float = 0.0,         # BC reg on demo actions (bootstraps hard sparse tasks)
    cta_ratio: int = 20,            # UTD ratio (RLPD uses high UTD)
    training_starts: int = 1000,
    random_steps: int = 1000,
    max_steps: int = 200_000,
    max_episode_steps: int = 700,
    lean_obs: bool = True,          # canonical low_dim state (match the state DP / its demos)
    extra_demos: str = "",          # npz of state-DP-collected demos (examples/collect_state_dp_demos.py)
    demo_buffer_capacity: int = 1_000_000,
    reset_to_demo_prob: float = 0.5,   # train resets from demo states (eval uses full dist)
    min_utd: float = 0.0,           # actor pacing (0 = off); set ~cta_ratio to keep UTD high
    eval_every: int = 10_000,       # env steps between evals
    eval_episodes: int = 25,
    http_port: int = 5588,
    out_path: str = "checkpoints/rlpd_expert.pkl",
    wandb_project: str = "",
    seed: int = 0,
):
    env = RoboMimicStateEnv(dataset_path, max_episode_steps=max_episode_steps,
                            reset_to_demo_prob=reset_to_demo_prob, lean_obs=lean_obs)
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    print(f"[rlpd] obs_dim={sample_obs.shape} action_dim={sample_action.shape}")

    agent = make_sac_state_agent(seed, sample_obs, sample_action, discount=discount,
                                 bc_weight=bc_weight)
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=1_000_000, demo_buffer_capacity=demo_buffer_capacity,
        max_steps=max_steps,
    )
    session = wrl.Session(agent, env, cfg, rng_seed=seed)

    demos = env.demo_transitions()
    n_human = len(demos)
    if extra_demos:
        from examples.collect_state_dp_demos import load_state_demos
        dp_demos = load_state_demos(extra_demos)
        assert dp_demos[0]["observations"].shape == demos[0]["observations"].shape, \
            (dp_demos[0]["observations"].shape, demos[0]["observations"].shape)
        demos = demos + dp_demos
        print(f"[rlpd] + {len(dp_demos)} state-DP demo transitions from {extra_demos}")
    n = session.preload_demos(demos)
    print(f"[rlpd] preloaded {n} demo transitions ({n_human} human + {n - n_human} DP)")

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            task=dataset_path.split("/")[-3], cta_ratio=cta_ratio, discount=discount,
            batch_size=batch_size))

    def evaluate(k):
        succ, rets = 0, []
        for ep in range(k):
            o, _ = env.reset(seed=20000 + ep, options={"normal": True})
            ret, s, d, t = 0.0, 0.0, False, False
            while not (d or t):
                o, r, d, t, info = env.step(session.policy.sample(o, argmax=True))
                ret += r
                s = max(s, float(info.get("success", 0.0)))
            succ += int(s > 0)
            rets.append(ret)
        return succ / k, float(np.mean(rets))

    session.start_learner()
    session.start_server(port=http_port)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_succ, last_eval, best_sr = 0.0, 0.0, 0, 0.0
    t0 = time.time()
    try:
        for step in range(1, 10_000_000):
            if not session.status()["learner_running"]:
                break
            if step < random_steps:
                action = env.action_space.sample()
            else:
                action = session.policy.sample(obs)
            next_obs, r, done, trunc, info = env.step(action)
            session.buffer.add(obs, action, next_obs, r, done)
            ep_ret += r
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))

            if done or trunc:
                session.record_episode(ep_ret)
                if min_utd > 0 and session.status()["learner_step"] > 0:
                    session.wait_for_utd(min_utd)
                obs, _ = env.reset()
                ep_ret, ep_succ = 0.0, 0.0
            else:
                obs = next_obs

            if step - last_eval >= eval_every:
                last_eval = step
                sr, mret = evaluate(eval_episodes)
                st = session.status()
                print(f"[eval] env_step={step} learner_step={st['learner_step']} "
                      f"utd={st['effective_utd']:.1f} success={sr:.1%} return={mret:.2f} "
                      f"({step/(time.time()-t0):.0f} env_it/s)")
                save_agent(session.snapshot_agent(), out_path, sample_obs, sample_action, discount)
                if sr >= best_sr:
                    best_sr = sr
                    save_agent(session.snapshot_agent(), out_path.replace(".pkl", "_best.pkl"),
                               sample_obs, sample_action, discount)
                if wandb_project:
                    import wandb
                    wandb.log({"eval/success": sr, "eval/return": mret, "eval/best": best_sr,
                               "env_step": step, "learner_step": st["learner_step"]})
    finally:
        session.stop_learner()
        save_agent(session.snapshot_agent(), out_path, sample_obs, sample_action, discount)
        print(f"[rlpd] saved final checkpoint to {out_path} (best success {best_sr:.1%})")
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
