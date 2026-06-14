"""Train a STATE (low-dim) flow-matching diffusion policy on a robomimic low_dim
dataset, with cold receding-horizon rollout eval on RoboMimicStateEnv.

State DPs avoid the pixel perception bottleneck (tool-hang pixel DP capped ~62%;
plantok's low-dim DP gets ~85%), so this gives a much stronger base/expert.

    uv run python -m wrl.diffusion.train_flow_state \
        --dataset-path data/robomimic/tool_hang/ph/low_dim_v141.hdf5 \
        --tp 16 --ta 8 --obs-history 2 --train-steps 100000 \
        --out-path checkpoints/flowdp_state_toolhang.pkl
"""

import os
import time
from collections import deque

import jax
import numpy as np
import tyro

from envs.dataset_loader import load_robomimic_state
from envs.robomimic_state import RoboMimicStateEnv
from wrl.diffusion.flow_policy import FlowPolicy


def _cold_rollout(fp, env, obs_history, rng, seed, use_ema=True):
    obs, _ = env.reset(seed=seed, options={"normal": True})
    hist = deque([obs] * obs_history, maxlen=obs_history)
    Ta, d_a = fp.config["Ta"], fp.config["d_a"]
    success, done, trunc = 0.0, False, False
    while not (done or trunc):
        obs_in = {"state": np.stack(hist)}  # (obs_history, state_dim)
        rng, k = jax.random.split(rng)
        chunk = np.asarray(jax.device_get(
            fp.sample_chunk(jax.device_put(obs_in), k, use_ema=use_ema)))
        chunk = chunk.reshape(Ta, d_a)
        for i in range(Ta):
            obs, _r, done, trunc, info = env.step(chunk[i])
            hist.append(obs)
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
    batch_size: int = 256,
    train_steps: int = 100000,
    lr: float = 1e-4,
    d_model: int = 256,
    n_layers: int = 6,
    n_sample_steps: int = 32,
    ema_decay: float = 0.9999,
    eval_every: int = 10000,
    eval_episodes: int = 25,
    max_episode_steps: int = 700,
    log_every: int = 500,
    seed: int = 0,
    wandb_project: str = "",
):
    env = RoboMimicStateEnv(dataset_path, max_episode_steps=max_episode_steps)
    d_a = env.action_space.shape[0]
    data = load_robomimic_state(dataset_path, tp)
    a_mean, a_std = data.action_stats()
    print(f"[flow-s] state_dim={data.state_dim} d_a={d_a} Tp={tp} Ta={ta} "
          f"obs_hist={obs_history} N={data.N} demos_state_keys={data.state_keys}")
    assert data.state_dim == env.observation_space.shape[0], \
        (data.state_dim, env.observation_space.shape)

    sample_obs = {"state": np.zeros((obs_history, data.state_dim), np.float32)}
    fp = FlowPolicy.create(
        jax.random.PRNGKey(seed), sample_obs, d_a, Tp=tp, Ta=ta, image_keys=(),
        use_proprio=True, d_model=d_model, n_layers=n_layers,
        n_sample_steps=n_sample_steps, learning_rate=lr, ema_decay=ema_decay,
    ).with_action_stats(a_mean, a_std)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            task=os.path.basename(dataset_path), variant="state", tp=tp, ta=ta,
            obs_history=obs_history, d_model=d_model, n_layers=n_layers,
            batch_size=batch_size, lr=lr))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rng = np.random.default_rng(seed)
    eval_rng = jax.random.PRNGKey(seed + 1)
    t0, ema, best = time.time(), None, -1.0
    for step in range(1, train_steps + 1):
        batch = jax.device_put(data.flow_sample(batch_size, tp, obs_history, rng))
        fp, info = fp.update(batch)
        loss = float(info["flow_loss"])
        ema = loss if ema is None else 0.99 * ema + 0.01 * loss
        if step % log_every == 0:
            print(f"[flow-s] step {step}/{train_steps} loss {loss:.4f} ema {ema:.4f} "
                  f"({step/(time.time()-t0):.1f} it/s)")
            if wandb_project:
                import wandb
                wandb.log({"flow_loss": loss, "ema": ema, "step": step})
        if eval_every and step % eval_every == 0:
            fp.save(out_path)
            succ, succ_o = [], []
            for ep in range(eval_episodes):
                eval_rng, k1, k2 = jax.random.split(eval_rng, 3)
                succ.append(_cold_rollout(fp, env, obs_history, k1, 20000 + ep, use_ema=True))
                succ_o.append(_cold_rollout(fp, env, obs_history, k2, 20000 + ep, use_ema=False))
            sr, sr_o = float(np.mean(succ)), float(np.mean(succ_o))
            print(f"[flow-s] step {step} EVAL ema={sr:.1%} online={sr_o:.1%} ({eval_episodes} ep)")
            if sr >= best:
                best = sr
                fp.save(out_path.replace(".pkl", "_best.pkl"))
            if wandb_project:
                import wandb
                wandb.log({"eval/success": sr, "eval/success_online": sr_o, "step": step})

    fp.save(out_path)
    print(f"[flow-s] saved {out_path} (best ema eval {best:.1%})")


if __name__ == "__main__":
    tyro.cli(main)
