"""Tight standalone eval of a saved DSRL checkpoint (best-of-n latent steering)
vs the frozen base DP, paired McNemar over n reset seeds.

    uv run python examples/toolhang/eval_dsrl.py \
        --dsrl-checkpoint checkpoints/dsrl_toolhang_best.pkl --episodes 50
"""

import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from envs.chunk_wrapper import ActionChunkWrapper
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.diffusion.flow_policy import FlowPolicy
from wrl.utils.launcher import make_sac_pixel_agent
from examples.toolhang.train_dsrl import LatentDecodeEnv


def _mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _wilson(p, n):
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def main(
    dsrl_checkpoint: str = "checkpoints/dsrl_toolhang_best.pkl",
    dataset_path: str = "data/robomimic/tool_hang/ph/image_84.hdf5",
    episodes: int = 50,
    n_sample_steps: int = 32,
    max_episode_steps: int = 700,
    discount: float = 0.99,
    seed: int = 0,
):
    blob = pickle.load(open(dsrl_checkpoint, "rb"))
    latent_dim, latent_scale = blob["latent_dim"], blob["latent_scale"]
    best_of_n = blob["best_of_n"]
    fp = FlowPolicy.load(blob["base_checkpoint"])
    fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Tp, Ta, d_a = fp.config["Tp"], fp.config["Ta"], fp.config["d_a"]
    obs_hist = fp.config["image_shape"][0]
    print(f"[eval-dsrl] ckpt={dsrl_checkpoint} latent_scale={latent_scale} "
          f"best_of_n={best_of_n} saved_eval={blob.get('eval_success')}")

    env = make_robomimic_pixel_env(dataset_path, image_size=84,
                                   max_episode_steps=max_episode_steps)
    chunk_env = ActionChunkWrapper(env, d_a, Ta, discount=discount, frame_history=obs_hist)
    cenv = LatentDecodeEnv(chunk_env, fp, latent_scale, Tp, Ta, d_a, obs_hist, norm_clip_ratio=5.0)

    agent = make_sac_pixel_agent(seed, env.observation_space.sample(),
                                 np.zeros(latent_dim, np.float32), image_keys=env.image_keys,
                                 discount=discount ** Ta, critic_ensemble_size=10,
                                 critic_subsample_size=2)
    agent = agent.replace(state=agent.state.replace(
        params=jax.tree_util.tree_map(jnp.asarray, blob["params"]),
        target_params=jax.tree_util.tree_map(jnp.asarray, blob["target_params"])))

    rng = np.random.default_rng(seed)

    def base_latent():
        return np.clip(rng.normal(0, 1.0 / latent_scale, latent_dim), -1, 1).astype(np.float32)

    def rollout(mode, key, reset_seed):
        o, _ = cenv.reset(seed=reset_seed)
        s, done, trunc = 0.0, False, False
        while not (done or trunc):
            if mode == "base":
                a = base_latent()
            else:
                key, k = jax.random.split(key)
                a = np.asarray(jax.device_get(
                    agent.sample_best_of_n(jax.device_put(o), best_of_n, k)), np.float32)
            o, r, done, trunc, info = cenv.step(a)
            s = max(s, float(info.get("success", 0.0)))
        return int(s > 0)

    key = jax.random.PRNGKey(seed + 1)
    base_s, dsrl_s = [], []
    for i in range(episodes):
        base_s.append(rollout("base", key, 1000 + i))
        key, k = jax.random.split(key)
        dsrl_s.append(rollout("bon", k, 1000 + i))
        if (i + 1) % 5 == 0:
            print(f"[eval-dsrl] {i+1}/{episodes} base={np.mean(base_s):.1%} dsrl={np.mean(dsrl_s):.1%}")
    base_s, dsrl_s = np.array(base_s), np.array(dsrl_s)
    b = int(((base_s == 1) & (dsrl_s == 0)).sum())
    c = int(((base_s == 0) & (dsrl_s == 1)).sum())
    p = _mcnemar_exact_p(b, c)
    bl, bh = _wilson(base_s.mean(), episodes)
    dl, dh = _wilson(dsrl_s.mean(), episodes)
    print(f"[eval-dsrl] base={base_s.mean():.1%} [{bl:.0%},{bh:.0%}]  "
          f"DSRL={dsrl_s.mean():.1%} [{dl:.0%},{dh:.0%}]  "
          f"delta={dsrl_s.mean()-base_s.mean():+.1%}  "
          f"discordant(base+/dsrl-={b}, base-/dsrl+={c}) McNemar p={p:.4f} n={episodes}")
    env.close()


if __name__ == "__main__":
    tyro.cli(main)
