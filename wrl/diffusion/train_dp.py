"""Train the base Diffusion Policy on a rendered robomimic image dataset.

    uv run python -m wrl.diffusion.train_dp \
        --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --horizon 8 --train-steps 100000 \
        --out-path checkpoints/dp_toolhang.pkl
"""

import os
import time

import jax
import numpy as np
import tyro

# import the env wrapper for canonical image/proprio keys (also sets EGL/stub)
from envs.robomimic_pixels import RoboMimicPixelEnv
from envs.dataset_loader import load_robomimic_pixels
from wrl.diffusion.policy import DiffusionPolicy


def main(
    dataset_path: str,
    out_path: str,
    horizon: int = 8,
    image_size: int = 84,
    batch_size: int = 128,
    train_steps: int = 100_000,
    learning_rate: float = 1e-4,
    num_train_timesteps: int = 100,
    num_infer_steps: int = 16,
    log_every: int = 500,
    save_every: int = 10_000,
    seed: int = 0,
):
    # Build the env once only to read canonical keys + action_dim (consistency).
    env = RoboMimicPixelEnv(dataset_path, image_size=image_size)
    image_keys, proprio_keys = env.image_keys, env.proprio_keys
    action_dim = env.action_space.shape[0]
    env.close()
    print(f"[dp] image_keys={image_keys} proprio_keys={proprio_keys} action_dim={action_dim}")

    data = load_robomimic_pixels(dataset_path, image_keys, proprio_keys, horizon)
    print(f"[dp] {data.N} transitions, {len(data.valid_starts)} valid chunk starts")

    sample_obs = {k: np.zeros((1, image_size, image_size, 3), np.uint8) for k in image_keys}
    sample_obs["state"] = np.zeros((1, data.proprio.shape[1]), np.float32)

    dp = DiffusionPolicy.create(
        jax.random.PRNGKey(seed), sample_obs, action_dim, horizon,
        image_keys=image_keys, num_train_timesteps=num_train_timesteps,
        num_infer_steps=num_infer_steps, learning_rate=learning_rate,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rng = np.random.default_rng(seed)
    t0, ema = time.time(), None
    for step in range(1, train_steps + 1):
        batch = data.dp_sample(batch_size, rng)
        batch = jax.device_put(batch)
        dp, info = dp.update(batch)
        loss = float(info["diffusion_loss"])
        ema = loss if ema is None else 0.99 * ema + 0.01 * loss
        if step % log_every == 0:
            sps = step / (time.time() - t0)
            print(f"[dp] step {step}/{train_steps} loss {loss:.4f} ema {ema:.4f} ({sps:.1f} it/s)")
        if step % save_every == 0:
            dp.save(out_path)
    dp.save(out_path)
    print(f"[dp] saved checkpoint to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
