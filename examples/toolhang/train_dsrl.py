"""DSRL: steer a frozen flow DP with latent-noise RL (the expert recipe).

The DP maps an initial cold-sample noise w in R^(Tp*d_a) deterministically to an
action chunk a = f_DP(s, w) (first Ta steps executed). DSRL runs plain SAC whose
ACTION is that latent w: the actor outputs w | s, the frozen DP decodes it to the
robot chunk, and the critic scores Q(s, w). Sampling w ~ N(0,I) reproduces the
base DP, so the task reward is available immediately (the base already succeeds
~62% on tool-hang) -- SAC just steers w toward the high-return modes. The DP
weights never change.

The SAC agent is the ordinary pixel agent over a (Tp*d_a)-dim continuous action;
all DSRL-specific logic (decode, base-noise warmup) lives in the actor loop here.

    python examples/toolhang/train_dsrl.py \
        --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --checkpoint checkpoints/flowdp_toolhang_v2.pkl --wandb-project wrl-dsrl
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wrl
from envs.chunk_wrapper import ActionChunkWrapper
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.diffusion.flow_policy import FlowPolicy
from wrl.utils.launcher import make_sac_pixel_agent


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def main(
    dataset_path: str,
    checkpoint: str,
    latent_scale: float = 3.0,      # w = latent_scale * tanh(actor); covers N(0,I) bulk
    discount: float = 0.99,
    image_size: int = 84,
    max_episode_steps: int = 700,
    batch_size: int = 256,
    cta_ratio: int = 4,
    training_starts: int = 1_000,
    random_chunks: int = 200,       # base-noise warmup (w ~ N(0,I) => base DP behavior)
    max_steps: int = 200_000,
    critic_ensemble_size: int = 10,  # REDQ (pessimism under high UTD)
    critic_subsample_size: int = 2,
    min_utd: float = 0.0,
    n_sample_steps: int = 32,       # DP cold-sample steps at decode time
    http_port: int = 5599,
    eval_every: int = 2_000,        # chunks between evals (0 = off)
    eval_episodes: int = 10,
    final_eval_episodes: int = 50,
    wandb_project: str = "",
    seed: int = 0,
):
    # ---- frozen base DP --------------------------------------------------
    fp = FlowPolicy.load(checkpoint)
    fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Tp, Ta, d_a = fp.config["Tp"], fp.config["Ta"], fp.config["d_a"]
    obs_hist = fp.config["image_shape"][0]
    latent_dim = Tp * d_a
    print(f"[dsrl] Tp={Tp} Ta={Ta} d_a={d_a} latent_dim={latent_dim} "
          f"obs_hist={obs_hist} scale={latent_scale}")

    env = make_robomimic_pixel_env(dataset_path, image_size=image_size,
                                   max_episode_steps=max_episode_steps)
    assert env.action_space.shape[0] == d_a, (env.action_space.shape, d_a)
    cenv = ActionChunkWrapper(env, d_a, Ta, discount=discount, frame_history=obs_hist)

    def decode(oh, w):
        """frozen DP: latent w (latent_dim,) -> executed chunk (Ta*d_a,)."""
        chunk = fp.decode_noise(jax.device_put(oh),
                                jnp.asarray(w, jnp.float32).reshape(Tp, d_a))
        return np.asarray(jax.device_get(chunk), np.float32)

    # ---- SAC over the latent action -------------------------------------
    sample_obs = env.observation_space.sample()
    sample_action = np.zeros(latent_dim, np.float32)
    agent = make_sac_pixel_agent(
        seed, sample_obs, sample_action, image_keys=env.image_keys,
        discount=discount ** Ta, critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=200_000, demo_buffer_capacity=1, max_steps=max_steps,
        image_keys=env.image_keys,
    )
    session = wrl.Session(agent, cenv, cfg, rng_seed=seed)
    session.start_learner()
    session.start_server(port=http_port)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(
            task=dataset_path.split("/")[-3], latent_scale=latent_scale,
            discount=discount, Ta=Ta, latent_dim=latent_dim, cta_ratio=cta_ratio))

    rng = np.random.default_rng(seed)
    sigma = 1.0 / latent_scale  # so w = scale * a ~ N(0,1) in the bulk during warmup

    def base_latent():
        """SAC-space action a whose decoded w ~ N(0,I) (= base DP)."""
        return np.clip(rng.normal(0.0, sigma, latent_dim), -1.0, 1.0).astype(np.float32)

    def _rollout(use_base, reset_seed):
        o, _ = cenv.reset(seed=reset_seed)
        s, done, trunc = 0.0, False, False
        while not (done or trunc):
            oh = cenv.base_obs(obs_hist)
            if use_base:
                a = base_latent()
            else:
                a = np.asarray(session.policy.sample(o, argmax=True), np.float32)
            o, r, done, trunc, info = cenv.step(decode(oh, latent_scale * a))
            s = max(s, float(info.get("success", 0.0)))
        return int(s > 0)

    def evaluate(n):
        return float(np.mean([_rollout(False, 30000 + i) for i in range(n)]))

    def paired_eval(n):
        base_s = np.array([_rollout(True, 1000 + i) for i in range(n)])
        dsrl_s = np.array([_rollout(False, 1000 + i) for i in range(n)])
        b = int(((base_s == 1) & (dsrl_s == 0)).sum())
        c = int(((base_s == 0) & (dsrl_s == 1)).sum())
        return base_s.mean(), dsrl_s.mean(), b, c, _mcnemar_exact_p(b, c)

    obs, _ = cenv.reset(seed=seed)
    chunk_step, last_eval = 0, 0
    ep_ret, ep_succ, t0 = 0.0, 0.0, time.time()
    try:
        while session.status()["learner_running"]:
            oh = cenv.base_obs(obs_hist)
            if chunk_step < random_chunks:
                a = base_latent()
            else:
                a = np.asarray(session.policy.sample(obs), np.float32)
            next_obs, r, done, trunc, info = cenv.step(decode(oh, latent_scale * a))
            session.buffer.add(obs, a, next_obs, r, done)
            ep_ret += r
            ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            chunk_step += 1

            if done or trunc:
                session.record_episode(ep_ret)
                st = session.status()
                print(f"[dsrl] ep_ret={ep_ret:.2f} success={ep_succ:.0f} "
                      f"chunks={chunk_step} learner_step={st['learner_step']} "
                      f"utd={st['effective_utd']:.2f} online={st['online_buffer']}")
                if wandb_project:
                    import wandb
                    wandb.log({"episode/return": ep_ret, "episode/success": ep_succ,
                               "learner_step": st["learner_step"], "env_chunks": chunk_step})
                if min_utd > 0 and st["learner_step"] > 0:
                    session.wait_for_utd(min_utd)
                if eval_every > 0 and chunk_step - last_eval >= eval_every:
                    last_eval = chunk_step
                    sr = evaluate(eval_episodes)
                    print(f"[eval] learner_step={st['learner_step']} success={sr:.1%}")
                    if wandb_project:
                        import wandb
                        wandb.log({"eval/success": sr, "learner_step": st["learner_step"]})
                obs, _ = cenv.reset()
                ep_ret, ep_succ = 0.0, 0.0
            else:
                obs = next_obs

        if final_eval_episodes > 0:
            bs, ds, b, c, p = paired_eval(final_eval_episodes)
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
