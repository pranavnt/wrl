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

import gymnasium as gym
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


class LatentDecodeEnv(gym.Wrapper):
    """Make the DSRL latent the env action: the action space is the DP latent
    w-space (Tp*d_a,); step() decodes w = scale*a through the FROZEN DP to the
    robot chunk (using the chunk wrapper's obs history) and steps the inner
    chunk env. So the replay buffer stores the latent (what SAC learns) and the
    robot only ever sees decoded chunks. Obs space is unchanged (the env's)."""

    def __init__(self, chunk_env, fp, latent_scale, Tp, Ta, d_a, obs_hist,
                 norm_clip_ratio=1.1, state_obs=False):
        super().__init__(chunk_env)
        self.fp, self.latent_scale = fp, latent_scale
        self.Tp, self.Ta, self.d_a, self.obs_hist = Tp, Ta, d_a, obs_hist
        self.action_space = gym.spaces.Box(-1.0, 1.0, (Tp * d_a,), np.float32)
        # clip the latent onto the N(0,I) prior shell (radius ~sqrt(latent_dim))
        # so the frozen DP never sees an OOD-norm noise -> no decode garbage. The
        # actor keeps full DIRECTIONAL steering; only magnitude is capped.
        self.r_max = float(np.sqrt(Tp * d_a) * norm_clip_ratio)
        # state_obs: the steering policy sees the flat low-dim state ("lowdim"),
        # while the frozen DP still decodes from the pixel obs-history (base_obs).
        self.state_obs = state_obs
        if state_obs:
            self.observation_space = chunk_env.observation_space["lowdim"]

    def _obs(self, obs):
        return obs["lowdim"] if self.state_obs else obs

    def decode(self, w):
        norm = float(np.linalg.norm(w))
        if norm > self.r_max:
            w = w * (self.r_max / (norm + 1e-8))
        chunk = self.fp.decode_noise(
            jax.device_put(self.env.base_obs(self.obs_hist)),
            jnp.asarray(w, jnp.float32).reshape(self.Tp, self.d_a))
        return np.asarray(jax.device_get(chunk), np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._obs(obs), info

    def step(self, a):
        w = self.latent_scale * np.asarray(a, np.float32)
        obs, r, done, trunc, info = self.env.step(self.decode(w))
        return self._obs(obs), r, done, trunc, info


def main(
    dataset_path: str,
    checkpoint: str,
    latent_scale: float = 1.0,      # w = latent_scale * tanh(actor); =1 keeps w in [-1,1] (in-dist)
    norm_clip_ratio: float = 5.0,   # cap |w| at sqrt(latent_dim)*ratio (safety; inactive at scale=1)
    best_of_n: int = 16,            # critic-guided: sample N latents, take the highest-Q decode
    state_policy: bool = False,     # steering policy sees low-dim state (DP still decodes pixels)
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
    out_path: str = "checkpoints/dsrl_toolhang.pkl",
    wandb_project: str = "",
    seed: int = 0,
):
    import os
    import pickle

    def save_dsrl(ag, path, sr):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {
            "params": jax.tree_util.tree_map(np.asarray, ag.state.params),
            "target_params": jax.tree_util.tree_map(np.asarray, ag.state.target_params),
            "latent_dim": latent_dim, "latent_scale": latent_scale,
            "best_of_n": best_of_n, "base_checkpoint": checkpoint,
            "image_keys": list(env.image_keys), "eval_success": sr,
            "state_policy": state_policy,
        }
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
        os.replace(tmp, path)
    # ---- frozen base DP --------------------------------------------------
    fp = FlowPolicy.load(checkpoint)
    fp = fp.replace(config={**fp.config, "n_sample_steps": n_sample_steps})
    Tp, Ta, d_a = fp.config["Tp"], fp.config["Ta"], fp.config["d_a"]
    obs_hist = fp.config["image_shape"][0]
    latent_dim = Tp * d_a
    print(f"[dsrl] Tp={Tp} Ta={Ta} d_a={d_a} latent_dim={latent_dim} "
          f"obs_hist={obs_hist} scale={latent_scale}")

    env = make_robomimic_pixel_env(dataset_path, image_size=image_size,
                                   max_episode_steps=max_episode_steps,
                                   include_lowdim=state_policy)
    assert env.action_space.shape[0] == d_a, (env.action_space.shape, d_a)
    chunk_env = ActionChunkWrapper(env, d_a, Ta, discount=discount, frame_history=obs_hist)
    cenv = LatentDecodeEnv(chunk_env, fp, latent_scale, Tp, Ta, d_a, obs_hist,
                           norm_clip_ratio=norm_clip_ratio, state_obs=state_policy)

    # ---- SAC over the latent action -------------------------------------
    sample_action = np.zeros(latent_dim, np.float32)
    if state_policy:
        from wrl.utils.launcher import make_sac_state_agent
        sample_obs = cenv.observation_space.sample()
        print(f"[dsrl] STATE noise policy: state_dim={sample_obs.shape}")
        agent = make_sac_state_agent(
            seed, sample_obs, sample_action, discount=discount ** Ta,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
        )
        image_keys_cfg = None
    else:
        sample_obs = env.observation_space.sample()
        agent = make_sac_pixel_agent(
            seed, sample_obs, sample_action, image_keys=env.image_keys,
            discount=discount ** Ta, critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
        )
        image_keys_cfg = env.image_keys
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=200_000, demo_buffer_capacity=1, max_steps=max_steps,
        image_keys=image_keys_cfg,
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

    def _policy_action(o, mode):
        if mode == "base":
            return base_latent()
        if mode == "bon":
            return np.asarray(session.policy.sample_best_of_n(o, best_of_n), np.float32)
        return np.asarray(session.policy.sample(o, argmax=(mode == "argmax")), np.float32)

    def _rollout(mode, reset_seed):
        """mode: 'base' (w~N(0,I)), 'argmax', 'sample', 'bon' (best-of-n).
        Returns (success, mean latent norm ||w|| over the episode)."""
        o, _ = cenv.reset(seed=reset_seed)
        s, done, trunc, norms = 0.0, False, False, []
        while not (done or trunc):
            a = _policy_action(o, mode)
            norms.append(float(np.linalg.norm(latent_scale * a)))
            o, r, done, trunc, info = cenv.step(a)
            s = max(s, float(info.get("success", 0.0)))
        return int(s > 0), float(np.mean(norms))

    def evaluate(n):
        bon = [_rollout("bon", 30000 + i) for i in range(n)]
        amax = [_rollout("argmax", 40000 + i) for i in range(n)]
        sr_b = float(np.mean([x[0] for x in bon]))
        sr_a = float(np.mean([x[0] for x in amax]))
        wnorm = float(np.mean([x[1] for x in bon]))
        return sr_b, sr_a, wnorm

    def paired_eval(n):
        base_s = np.array([_rollout("base", 1000 + i)[0] for i in range(n)])
        dsrl_s = np.array([_rollout("bon", 1000 + i)[0] for i in range(n)])
        b = int(((base_s == 1) & (dsrl_s == 0)).sum())
        c = int(((base_s == 0) & (dsrl_s == 1)).sum())
        return base_s.mean(), dsrl_s.mean(), b, c, _mcnemar_exact_p(b, c)

    obs, _ = cenv.reset(seed=seed)
    chunk_step, last_eval, best_sr = 0, 0, -1.0
    ep_ret, ep_succ, t0 = 0.0, 0.0, time.time()
    try:
        while session.status()["learner_running"]:
            if chunk_step < random_chunks:
                a = base_latent()
            else:
                a = np.asarray(session.policy.sample_best_of_n(obs, best_of_n), np.float32)
            next_obs, r, done, trunc, info = cenv.step(a)
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
                    sr_b, sr_a, wnorm = evaluate(eval_episodes)
                    print(f"[eval] learner_step={st['learner_step']} bon={sr_b:.1%} "
                          f"argmax={sr_a:.1%} |w|={wnorm:.1f} (base|w|~{math.sqrt(latent_dim):.1f})")
                    save_dsrl(session.snapshot_agent(), out_path, sr_b)
                    if sr_b >= best_sr:
                        best_sr = sr_b
                        save_dsrl(session.snapshot_agent(), out_path.replace(".pkl", "_best.pkl"), sr_b)
                    if wandb_project:
                        import wandb
                        wandb.log({"eval/success": sr_b, "eval/success_argmax": sr_a,
                                   "eval/latent_norm": wnorm, "learner_step": st["learner_step"]})
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
