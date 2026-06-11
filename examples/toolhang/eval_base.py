"""Evaluate a trained base Diffusion Policy alone (no residual) on tool-hang.

Reports success rate over N episodes — the baseline EXPO-FT residual RL should
beat. Runs the DP in-process (no server needed).

    uv run python examples/toolhang/eval_base.py \
        --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --checkpoint checkpoints/dp_toolhang.pkl --episodes 20
"""

import jax
import numpy as np
import tyro

from envs.chunk_wrapper import ActionChunkWrapper
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.diffusion.policy import DiffusionPolicy


def main(
    dataset_path: str,
    checkpoint: str,
    episodes: int = 20,
    image_size: int = 84,
    max_episode_steps: int = 700,
    seed: int = 0,
):
    dp = DiffusionPolicy.load(checkpoint)
    horizon = dp.config["horizon"]

    env = make_robomimic_pixel_env(
        dataset_path, image_size=image_size, max_episode_steps=max_episode_steps
    )
    A = env.action_space.shape[0]
    cenv = ActionChunkWrapper(env, A, horizon, discount=0.99)

    rng = jax.random.PRNGKey(seed)
    successes, returns = 0, []
    for ep in range(episodes):
        obs, _ = cenv.reset(seed=seed + ep)
        ep_ret, ep_succ, done, trunc = 0.0, 0.0, False, False
        while not (done or trunc):
            rng, k = jax.random.split(rng)
            chunk = np.asarray(jax.device_get(dp.sample(obs, k)), np.float32)
            obs, r, done, trunc, info = cenv.step(chunk)
            ep_ret += r
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
        successes += int(ep_succ > 0)
        returns.append(ep_ret)
        print(f"[eval] ep {ep+1}/{episodes} success={ep_succ:.0f} return={ep_ret:.2f}")

    cenv.close()
    print(f"[eval] base DP success rate: {successes}/{episodes} = {successes/episodes:.1%}")


if __name__ == "__main__":
    tyro.cli(main)
