"""Roll out the STATE diffusion policy and collect successful trajectories as
RLPD demo transitions (lean low-dim state, matching the DP and the RLPD env).

Each kept episode is truncated at first success (clean goal-reaching demo). Saves
incrementally to an npz of per-step transitions; load with `load_state_demos`.

    uv run python examples/collect_state_dp_demos.py \
        --dp-checkpoint checkpoints/flowdp_state_toolhang_best.pkl \
        --dataset-path data/robomimic/tool_hang/ph/low_dim_v141.hdf5 \
        --target-successes 1500 --out-path data/dp_demos/toolhang_state.npz
"""

import os
import time
from collections import deque

import jax
import numpy as np
import tyro

from envs.robomimic_state import RoboMimicStateEnv
from wrl.diffusion.flow_policy import FlowPolicy


def load_state_demos(path):
    """npz -> list of RLPD transition dicts (is_intervention=True)."""
    d = np.load(path)
    out = []
    for i in range(len(d["actions"])):
        out.append(dict(
            observations=d["observations"][i], actions=d["actions"][i],
            next_observations=d["next_observations"][i], rewards=float(d["rewards"][i]),
            masks=float(d["masks"][i]), dones=bool(d["dones"][i]), is_intervention=True,
        ))
    return out


def main(
    dp_checkpoint: str,
    dataset_path: str,
    out_path: str,
    target_successes: int = 1500,
    n_sample_steps: int = 32,
    use_ema: bool = True,
    max_episode_steps: int = 700,
    save_every: int = 50,
    seed: int = 0,
):
    fp = FlowPolicy.load(dp_checkpoint)
    fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Ta, d_a = fp.config["Ta"], fp.config["d_a"]
    obs_hist = fp.config["obs_history"]
    env = RoboMimicStateEnv(dataset_path, max_episode_steps=max_episode_steps, lean_obs=True)
    assert env.observation_space.shape[0] == fp.config["proprio_dim"], \
        (env.observation_space.shape, fp.config["proprio_dim"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[collect] state_dim={env.observation_space.shape[0]} Ta={Ta} d_a={d_a} "
          f"obs_hist={obs_hist} use_ema={use_ema} target={target_successes}")

    obs_buf, act_buf, nobs_buf, rew_buf, mask_buf, done_buf = [], [], [], [], [], []
    successes, episodes, rng = 0, 0, jax.random.PRNGKey(seed + 1)
    t0 = time.time()

    def save():
        tmp = out_path + ".tmp"
        np.savez_compressed(
            tmp, observations=np.asarray(obs_buf, np.float32),
            actions=np.asarray(act_buf, np.float32),
            next_observations=np.asarray(nobs_buf, np.float32),
            rewards=np.asarray(rew_buf, np.float32), masks=np.asarray(mask_buf, np.float32),
            dones=np.asarray(done_buf, bool), n_successes=successes, n_episodes=episodes)
        os.replace(tmp + ".npz", out_path)

    while successes < target_successes:
        o, _ = env.reset(seed=10_000_000 + episodes, options={"normal": True})
        hist = deque([o] * obs_hist, maxlen=obs_hist)
        ep, done, trunc, succeeded = [], False, False, False
        while not (done or trunc):
            obs_in = {"state": np.stack(hist)}
            rng, k = jax.random.split(rng)
            chunk = np.asarray(jax.device_get(
                fp.sample_chunk(jax.device_put(obs_in), k, use_ema=use_ema))).reshape(Ta, d_a)
            for i in range(Ta):
                s = o
                o, r, done, trunc, info = env.step(chunk[i])
                hist.append(o)
                succ_now = float(info.get("success", 0.0)) > 0
                ep.append((s, chunk[i], o, float(r), 0.0 if succ_now else 1.0, succ_now))
                if succ_now:
                    succeeded = True
                    break
                if trunc:
                    break
            if succeeded:
                break
        episodes += 1
        if succeeded:
            for s, a, ns, r, m, dn in ep:
                obs_buf.append(s); act_buf.append(a); nobs_buf.append(ns)
                rew_buf.append(r); mask_buf.append(m); done_buf.append(dn)
            successes += 1
            if successes % save_every == 0:
                save()
                print(f"[collect] {successes}/{target_successes} successes "
                      f"({episodes} eps, {successes/max(1,episodes):.0%} rate, "
                      f"{len(act_buf)} transitions, {successes/(time.time()-t0)*60:.1f} succ/min)")
    save()
    print(f"[collect] DONE {successes} successes / {episodes} eps, "
          f"{len(act_buf)} transitions -> {out_path}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
