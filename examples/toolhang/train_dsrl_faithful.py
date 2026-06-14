"""Faithful DSRL: two-critic (action critic + distilled latent critic) steering
of a frozen pixel base DP, with a STATE-conditioned steering policy.

Uses wrl.agents.dsrl.DSRLAgent. The action critic Q_a(s,a) is a SARSA backup on
the stored decoded action chunk (one decode/step, carried forward like the
residual base_actions); the latent critic Q_z(s,w) is distilled from it; the
actor + best-of-n steer on Q_z. See wrl/agents/dsrl.py.

    python examples/toolhang/train_dsrl_faithful.py \
        --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --checkpoint checkpoints/flowdp_toolhang_v2.pkl --wandb-project wrl-dsrl
"""

import math
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wrl
from envs.chunk_wrapper import ActionChunkWrapper
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.agents.dsrl import DSRLAgent
from wrl.diffusion.flow_policy import FlowPolicy
from examples.toolhang.train_dsrl import LatentDecodeEnv


def _mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def main(
    dataset_path: str,
    checkpoint: str,
    latent_scale: float = 1.0,
    rho: float = 0.5,
    best_of_n: int = 16,
    discount: float = 0.99,
    batch_size: int = 256,
    cta_ratio: int = 1,
    max_utd: float = 8.0,
    training_starts: int = 1_000,
    random_chunks: int = 1_000,
    max_steps: int = 60_000,
    num_qs: int = 10,
    n_sample_steps: int = 32,
    image_size: int = 84,
    max_episode_steps: int = 700,
    eval_every: int = 2_000,
    eval_episodes: int = 10,
    final_eval_episodes: int = 50,
    out_path: str = "checkpoints/dsrl_faithful_toolhang.pkl",
    http_port: int = 5601,
    wandb_project: str = "",
    seed: int = 0,
):
    fp = FlowPolicy.load(checkpoint)
    fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Tp, Ta, d_a = fp.config["Tp"], fp.config["Ta"], fp.config["d_a"]
    obs_hist = fp.config["image_shape"][0]
    latent_dim, decoded_dim = Tp * d_a, Ta * d_a
    print(f"[dsrl-f] Tp={Tp} Ta={Ta} d_a={d_a} latent={latent_dim} decoded={decoded_dim}")

    env = make_robomimic_pixel_env(dataset_path, image_size=image_size,
                                   max_episode_steps=max_episode_steps, include_lowdim=True)
    chunk_env = ActionChunkWrapper(env, d_a, Ta, discount=discount, frame_history=obs_hist)
    cenv = LatentDecodeEnv(chunk_env, fp, latent_scale, Tp, Ta, d_a, obs_hist,
                           norm_clip_ratio=5.0, state_obs=True)
    sample_state = cenv.observation_space.sample()
    print(f"[dsrl-f] state_dim={sample_state.shape}")

    agent = DSRLAgent.create(seed, sample_state, latent_dim, decoded_dim,
                             discount=discount ** Ta, noise_scale=latent_scale, rho=rho,
                             num_qs=num_qs)
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=200_000, demo_buffer_capacity=1, max_steps=max_steps,
        max_utd=max_utd, image_keys=None,
        extra_fields=(("decoded_actions", np.float32, (decoded_dim,)),
                      ("next_decoded_actions", np.float32, (decoded_dim,))),
    )
    session = wrl.Session(agent, cenv, cfg, rng_seed=seed)
    session.start_learner()
    session.start_server(port=http_port)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            task=dataset_path.split("/")[-3], variant="faithful_state", rho=rho,
            best_of_n=best_of_n, latent_dim=latent_dim))

    rng = np.random.default_rng(seed)
    sigma = 1.0 / latent_scale

    def base_latent():
        return np.clip(rng.normal(0, sigma, latent_dim), -1, 1).astype(np.float32)

    def save(ag, path, sr):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {"params": jax.tree_util.tree_map(np.asarray, ag.state.params),
                "target_params": jax.tree_util.tree_map(np.asarray, ag.target_params),
                "latent_dim": latent_dim, "decoded_dim": decoded_dim,
                "latent_scale": latent_scale, "best_of_n": best_of_n, "rho": rho,
                "base_checkpoint": checkpoint, "state_dim": int(sample_state.shape[0]),
                "eval_success": sr}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
        os.replace(tmp, path)

    def _rollout(mode, reset_seed):
        o, _ = cenv.reset(seed=reset_seed)
        s, done, trunc = 0.0, False, False
        while not (done or trunc):
            if mode == "base":
                a = base_latent()
            else:
                a = np.asarray(session.policy.sample_best_of_n(o, best_of_n), np.float32)
            o, r, done, trunc, info = cenv.step(a)
            s = max(s, float(info.get("success", 0.0)))
        return int(s > 0)

    def evaluate(n):
        return float(np.mean([_rollout("bon", 30000 + i) for i in range(n)]))

    def paired(n):
        bs = np.array([_rollout("base", 1000 + i) for i in range(n)])
        ds = np.array([_rollout("bon", 1000 + i) for i in range(n)])
        b = int(((bs == 1) & (ds == 0)).sum()); c = int(((bs == 0) & (ds == 1)).sum())
        return bs.mean(), ds.mean(), b, c, _mcnemar_exact_p(b, c)

    # ---- collection loop (1 decode/step, decoded action carried forward) ----
    obs, _ = cenv.reset(seed=seed)
    a = base_latent()
    decoded = cenv.decode_action(a)
    chunk_step, last_eval, best_sr = 0, 0, -1.0
    ep_ret, ep_succ, t0 = 0.0, 0.0, time.time()
    try:
        while session.status()["learner_running"]:
            next_obs, r, done, trunc, info = cenv.step_decoded(decoded)
            if chunk_step < random_chunks:
                next_a = base_latent()
            else:
                next_a = np.asarray(session.policy.sample_best_of_n(next_obs, best_of_n), np.float32)
            next_decoded = cenv.decode_action(next_a)
            session.buffer.add(obs, a, next_obs, r, done,
                               decoded_actions=decoded, next_decoded_actions=next_decoded)
            ep_ret += r
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            chunk_step += 1

            if done or trunc:
                session.record_episode(ep_ret)
                st = session.status()
                print(f"[dsrl-f] ep_ret={ep_ret:.2f} success={ep_succ:.0f} "
                      f"chunks={chunk_step} learner_step={st['learner_step']} "
                      f"utd={st['effective_utd']:.2f}")
                if wandb_project:
                    import wandb
                    wandb.log({"episode/return": ep_ret, "episode/success": ep_succ,
                               "learner_step": st["learner_step"], "env_chunks": chunk_step})
                if eval_every > 0 and chunk_step - last_eval >= eval_every:
                    last_eval = chunk_step
                    sr = evaluate(eval_episodes)
                    print(f"[eval] learner_step={st['learner_step']} bon={sr:.1%}")
                    save(session.snapshot_agent(), out_path, sr)
                    if sr >= best_sr:
                        best_sr = sr
                        save(session.snapshot_agent(), out_path.replace(".pkl", "_best.pkl"), sr)
                    if wandb_project:
                        import wandb
                        wandb.log({"eval/success": sr, "learner_step": st["learner_step"]})
                obs, _ = cenv.reset()
                a = base_latent()
                decoded = cenv.decode_action(a)
                ep_ret, ep_succ = 0.0, 0.0
            else:
                obs, a, decoded = next_obs, next_a, next_decoded

        if final_eval_episodes > 0:
            bs, ds, b, c, p = paired(final_eval_episodes)
            print(f"[final-eval] base={bs:.1%} dsrl={ds:.1%} "
                  f"discordant(base+/dsrl-={b}, base-/dsrl+={c}) McNemar p={p:.4f} "
                  f"n={final_eval_episodes}")
            if wandb_project:
                import wandb
                wandb.run.summary.update({"final/base": bs, "final/dsrl": ds,
                                          "final/delta": ds - bs, "final/mcnemar_p": p})
    finally:
        session.stop_learner()
        cenv.close()


if __name__ == "__main__":
    tyro.cli(main)
