"""Train the JAX flow-matching pixel DP on a robomimic image dataset, with
cold receding-horizon rollout eval.

    uv run python -m wrl.diffusion.train_flow \
        --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --tp 16 --ta 8 --obs-history 2 --train-steps 60000 \
        --out-path checkpoints/flowdp_toolhang.pkl
"""

import os
import time
from collections import deque

import jax
import numpy as np
import tyro

from envs.robomimic_pixels import RoboMimicPixelEnv
from envs.dataset_loader import load_robomimic_pixels
from wrl.diffusion.flow_policy import FlowPolicy


def _cold_rollout(fp, env, obs_history, rng, seed):
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
        for i in range(Ta):
            obs, _r, done, trunc, info = env.step(chunk[i])
            for cam in env.image_keys:
                himg[cam].append(obs[cam][0])
            hstate.append(obs["state"][0])
            success = max(success, float(info.get("success", 0.0)))
            if done or trunc:
                break
    return success


def main(
    dataset_path: str,
    out_path: str,
    tp: int = 16,
    ta: int = 8,
    obs_history: int = 2,
    image_size: int = 84,
    batch_size: int = 128,
    train_steps: int = 60000,
    lr: float = 1e-4,
    d_model: int = 256,
    n_layers: int = 6,
    n_sample_steps: int = 16,
    eval_every: int = 5000,
    eval_episodes: int = 20,
    max_episode_steps: int = 700,
    log_every: int = 500,
    seed: int = 0,
    wandb_project: str = "",
):
    env = RoboMimicPixelEnv(dataset_path, image_size=image_size,
                            max_episode_steps=max_episode_steps)
    image_keys, proprio_keys = env.image_keys, env.proprio_keys
    d_a = env.action_space.shape[0]
    print(f"[flow] image_keys={image_keys} d_a={d_a} Tp={tp} Ta={ta} obs_hist={obs_history}")

    data = load_robomimic_pixels(dataset_path, image_keys, proprio_keys, tp)
    a_mean, a_std = data.action_stats()
    print(f"[flow] {data.N} steps; action std range "
          f"[{a_std.min():.3f},{a_std.max():.3f}]")

    sample_obs = {k: np.zeros((obs_history, image_size, image_size, 3), np.uint8)
                  for k in image_keys}
    sample_obs["state"] = np.zeros((obs_history, data.proprio.shape[1]), np.float32)

    fp = FlowPolicy.create(
        jax.random.PRNGKey(seed), sample_obs, d_a, Tp=tp, Ta=ta, image_keys=image_keys,
        d_model=d_model, n_layers=n_layers, n_sample_steps=n_sample_steps, learning_rate=lr,
    ).with_action_stats(a_mean, a_std)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            task=os.path.basename(dataset_path), tp=tp, ta=ta, obs_history=obs_history,
            d_model=d_model, n_layers=n_layers, batch_size=batch_size, lr=lr))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rng = np.random.default_rng(seed)
    eval_rng = jax.random.PRNGKey(seed + 1)
    t0, ema = time.time(), None
    for step in range(1, train_steps + 1):
        batch = jax.device_put(data.flow_sample(batch_size, tp, obs_history, rng))
        fp, info = fp.update(batch)
        loss = float(info["flow_loss"])
        ema = loss if ema is None else 0.99 * ema + 0.01 * loss
        if step % log_every == 0:
            sps = step / (time.time() - t0)
            print(f"[flow] step {step}/{train_steps} loss {loss:.4f} ema {ema:.4f} ({sps:.1f} it/s)")
            if wandb_project:
                import wandb
                wandb.log({"flow_loss": loss, "ema": ema, "step": step})
        if eval_every and step % eval_every == 0:
            fp.save(out_path)
            succ = []
            for ep in range(eval_episodes):
                eval_rng, k = jax.random.split(eval_rng)
                succ.append(_cold_rollout(fp, env, obs_history, k, seed=20000 + ep))
            sr = float(np.mean(succ))
            print(f"[flow] step {step} EVAL success={sr:.1%} ({eval_episodes} ep)")
            if wandb_project:
                import wandb
                wandb.log({"eval/success": sr, "step": step})

    fp.save(out_path)
    print(f"[flow] saved {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
