"""Tight standalone eval of a trained flow DP checkpoint, with two inference
levers that need no retraining:

  --n-sample-steps : Euler integration steps for cold sampling (train used 16)
  --n-exec         : how many of the Ta sampled actions to execute before
                     replanning (receding horizon; <Ta replans more often,
                     usually more precise on hard tasks). Default = Ta.

    uv run python -m wrl.diffusion.eval_flow \
        --checkpoint checkpoints/flowdp_toolhang.pkl --episodes 50 \
        --n-sample-steps 32 --n-exec 4
"""

import time
from collections import deque

import jax
import numpy as np
import tyro

from envs.robomimic_pixels import RoboMimicPixelEnv
from wrl.diffusion.flow_policy import FlowPolicy


def _rollout(fp, env, obs_history, n_exec, rng, seed):
    obs, _ = env.reset(seed=seed)
    himg = {k: deque([obs[k][0]] * obs_history, maxlen=obs_history) for k in env.image_keys}
    hstate = deque([obs["state"][0]] * obs_history, maxlen=obs_history)
    Ta, d_a = fp.config["Ta"], fp.config["d_a"]
    success, done, trunc = 0.0, False, False
    while not (done or trunc):
        obs_in = {k: np.stack(himg[k]) for k in env.image_keys}
        obs_in["state"] = np.stack(hstate)
        rng, k = jax.random.split(rng)
        chunk = np.asarray(jax.device_get(fp.sample_chunk(jax.device_put(obs_in), k)))
        chunk = chunk.reshape(Ta, d_a)
        for i in range(min(n_exec, Ta)):
            obs, _r, done, trunc, info = env.step(chunk[i])
            for cam in env.image_keys:
                himg[cam].append(obs[cam][0])
            hstate.append(obs["state"][0])
            success = max(success, float(info.get("success", 0.0)))
            if done or trunc:
                break
    return success


def main(
    checkpoint: str,
    episodes: int = 50,
    n_sample_steps: int = 0,   # 0 = use checkpoint's value
    n_exec: int = 0,           # 0 = execute full Ta
    obs_history: int = 0,      # 0 = infer from checkpoint image_shape[0]
    image_size: int = 84,
    max_episode_steps: int = 700,
    seed: int = 0,
):
    fp = FlowPolicy.load(checkpoint)
    if n_sample_steps > 0:
        fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Ta = fp.config["Ta"]
    if n_exec <= 0:
        n_exec = Ta
    if obs_history <= 0:
        obs_history = fp.config["image_shape"][0]

    # derive dataset path from checkpoint name is brittle; require env build from a
    # dataset with matching task. Reuse the standard tool-hang/transport image hdf5.
    import os
    task = "tool_hang" if "toolhang" in os.path.basename(checkpoint) else "transport"
    dataset_path = f"data/robomimic/{task}/ph/image_84.hdf5"
    env = RoboMimicPixelEnv(dataset_path, image_size=image_size,
                            max_episode_steps=max_episode_steps)

    print(f"[eval] ckpt={checkpoint} Ta={Ta} n_exec={n_exec} "
          f"n_sample_steps={fp.config['n_sample_steps']} obs_hist={obs_history} "
          f"episodes={episodes}")
    rng = jax.random.PRNGKey(seed + 1)
    succ, t0 = [], time.time()
    for ep in range(episodes):
        rng, k = jax.random.split(rng)
        s = _rollout(fp, env, obs_history, n_exec, k, seed=20000 + ep)
        succ.append(s)
        if (ep + 1) % 5 == 0:
            print(f"[eval] {ep+1}/{episodes} running success={np.mean(succ):.1%} "
                  f"({(ep+1)/(time.time()-t0):.2f} ep/s)")
    sr = float(np.mean(succ))
    # Wilson 95% CI
    n = len(succ); p = sr; z = 1.96
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    print(f"[eval] FINAL success={sr:.1%}  ({int(sr*n)}/{n})  "
          f"Wilson95=[{max(0,centre-half):.1%},{min(1,centre+half):.1%}]  "
          f"n_exec={n_exec} n_sample_steps={fp.config['n_sample_steps']}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
