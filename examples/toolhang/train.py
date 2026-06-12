"""Tool-hang training: EXPO-FT residual RL or chunked RL-from-scratch.

The two modes share everything but where the base action chunk comes from:

  --base dp     EXPO-FT: query a running DP base-policy server (start it with
                `python -m wrl.base_policy.dp_server --checkpoint ...`).
  --base zeros  chunked RL from scratch: base chunk is zero, so the residual
                policy *is* the policy.

Example (EXPO-FT):
    python -m wrl.base_policy.dp_server --checkpoint checkpoints/dp_toolhang.pkl --port 8200 &
    python examples/toolhang/train.py --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
        --base dp --dp-port 8200 --horizon 8 --warmstart-demos
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.
    b = base success & residual fail, c = base fail & residual success."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)

import wrl
from envs.chunk_wrapper import ActionChunkWrapper
from envs.dataset_loader import load_robomimic_pixels
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.agents.residual_sac import make_residual_sac_pixel_agent
from wrl.base_policy import BasePolicyClient
from wrl.data import residual_extra_fields


def main(
    dataset_path: str,
    base: str = "dp",               # "dp" | "flow" | "zeros"
    dp_host: str = "localhost",
    dp_port: int = 8200,
    base_obs_history: int = 2,      # frames sent to a flow DP base (must match its training)
    horizon: int = 8,
    edit_scale: float = 0.25,
    edit_scale_end: float = 0.0,     # >0 enables curriculum: edit_scale ramps edit_scale->end
    edit_scale_incr: float = 0.0,    # increment per edit_scale_steps grad steps
    edit_scale_steps: int = 2500,
    bc_weight: float = 0.0,          # BC reg anchoring residual->(demo-base) on demo transitions
    discount: float = 0.99,
    image_size: int = 84,
    max_episode_steps: int = 700,
    batch_size: int = 256,
    cta_ratio: int = 4,
    training_starts: int = 1_000,
    random_chunks: int = 50,
    max_steps: int = 200_000,
    min_utd: float = 0.0,
    max_utd: float = 0.0,        # learner-side UTD cap (0=off); prevents over-training on slow tasks
    warmstart_demos: bool = False,
    warmstart_max: int = 5_000,
    http_port: int = 5588,
    encoder_type: str = "resnet",
    eval_every: int = 0,        # chunks between evals (0 = off)
    eval_episodes: int = 10,
    final_eval_episodes: int = 0,   # paired base-only vs residual at end (0 = off)
    wandb_project: str = "",
    seed: int = 0,
    smoke: bool = False,
):
    if smoke:
        base, max_steps, training_starts, random_chunks = "zeros", 30, 20, 8
        cta_ratio, max_episode_steps = 1, 60

    if wandb_project:
        import wandb

        wandb.init(project=wandb_project, config=dict(
            base=base, horizon=horizon, edit_scale=edit_scale, discount=discount,
            cta_ratio=cta_ratio, batch_size=batch_size, encoder_type=encoder_type,
        ))

    env = make_robomimic_pixel_env(
        dataset_path, image_size=image_size, max_episode_steps=max_episode_steps
    )
    A = env.action_space.shape[0]
    chunk_dim = A * horizon
    cenv = ActionChunkWrapper(
        env, A, horizon, discount=discount,
        frame_history=(base_obs_history if base == "flow" else 1),
    )

    sample_obs = env.observation_space.sample()
    agent = make_residual_sac_pixel_agent(
        seed, sample_obs, np.zeros(chunk_dim, np.float32), np.zeros(chunk_dim, np.float32),
        action_dim=A, horizon=horizon, image_keys=env.image_keys,
        encoder_type=encoder_type, edit_scale=edit_scale, discount_per_step=discount,
        edit_scale_end=(edit_scale_end if edit_scale_end > 0 else None),
        edit_scale_incr=edit_scale_incr, edit_scale_steps=edit_scale_steps,
        bc_weight=bc_weight,
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(
        batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
        replay_buffer_capacity=200_000, demo_buffer_capacity=200_000,
        max_steps=max_steps, max_utd=max_utd, image_keys=env.image_keys,
        extra_fields=residual_extra_fields(chunk_dim),
    )
    session = wrl.Session(agent, cenv, cfg, rng_seed=seed)

    if base == "flow":
        # flow DP needs the consecutive obs-history at the chunk boundary, which
        # the chunk wrapper tracks; ignore the single-frame obs passed in.
        client = BasePolicyClient(host=dp_host, port=dp_port)
        query_base = lambda _obs: np.asarray(  # noqa: E731
            client.query(cenv.base_obs(base_obs_history)), np.float32)
    elif base == "dp":
        client = BasePolicyClient(host=dp_host, port=dp_port)
        query_base = client.query
    elif base == "zeros":
        zeros = np.zeros(chunk_dim, np.float32)
        query_base = lambda _obs: zeros  # noqa: E731
    else:
        raise ValueError(f"--base must be 'flow', 'dp' or 'zeros', got {base!r}")

    if warmstart_demos:
        data = load_robomimic_pixels(dataset_path, env.image_keys, env.proprio_keys, horizon)
        # base-consistent seeding: relabel demo base_actions to the flow-DP's
        # chunk at s/s+H (keeps full action = demo, so d(s,a)=s' exact).
        bqf = (lambda oh: np.asarray(client.query(oh), np.float32)) if base == "flow" else None
        print(f"[actor] building warmstart demos (base_consistent={bqf is not None})...")
        demos = data.residual_transitions(
            discount, max_transitions=warmstart_max, seed=seed,
            base_query_fn=bqf, base_obs_history=base_obs_history,
        )
        n = session.preload_demos(demos)
        print(f"[actor] warm-started demo buffer with {n} residual transitions")

    session.start_learner()
    session.start_server(port=http_port)
    print(f"[actor] base={base} edit_scale={edit_scale} chunk_dim={chunk_dim} "
          f"action_dim={A} horizon={horizon}")

    def _rollout(use_residual: bool, reset_seed):
        """One argmax rollout. use_residual=False executes the raw base chunk
        (DP only); True executes base + residual."""
        o, _ = cenv.reset(seed=reset_seed)
        b = np.asarray(query_base(o), np.float32)
        ret, s, d, t = 0.0, 0.0, False, False
        while not (d or t):
            full = session.policy.sample(o, b, argmax=True) if use_residual else b
            o, r, d, t, info = cenv.step(full)
            b = np.asarray(query_base(o), np.float32)
            ret += r
            s = max(s, float(info.get("success", 0.0)))
        return int(s > 0), ret

    def evaluate(n):
        """Residual policy success/return over n episodes."""
        outs = [_rollout(True, None) for _ in range(n)]
        return np.mean([o[0] for o in outs]), float(np.mean([o[1] for o in outs]))

    def paired_eval(n):
        """Paired base-only vs base+residual over the same n reset seeds."""
        base_s, res_s = [], []
        for s in range(n):
            base_s.append(_rollout(False, 1000 + s)[0])
            res_s.append(_rollout(True, 1000 + s)[0])
        base_s, res_s = np.array(base_s), np.array(res_s)
        b = int(np.sum((base_s == 1) & (res_s == 0)))  # base win, res lose
        c = int(np.sum((base_s == 0) & (res_s == 1)))  # res win, base lose
        p = _mcnemar_exact_p(b, c)
        return base_s.mean(), res_s.mean(), b, c, p

    obs, _ = cenv.reset(seed=seed)
    a_base = np.asarray(query_base(obs), np.float32)
    assert a_base.shape == (chunk_dim,), (a_base.shape, chunk_dim)

    chunk_step = 0
    last_eval = 0
    ep_ret, ep_success = 0.0, 0.0
    last_log = time.time()
    try:
        while session.status()["learner_running"]:
            if chunk_step < random_chunks:
                full = cenv.action_space.sample()
            else:
                full = session.policy.sample(obs, a_base)
            next_obs, r, done, trunc, info = cenv.step(full)
            next_a_base = np.asarray(query_base(next_obs), np.float32)
            session.buffer.add(
                obs, full, next_obs, r, done,
                base_actions=a_base, next_base_actions=next_a_base,
            )
            ep_ret += r
            ep_success = max(ep_success, float(info.get("success", 0.0)))
            chunk_step += 1

            if done or trunc:
                session.record_episode(ep_ret)
                st = session.status()
                print(f"[actor] ep_ret={ep_ret:.2f} success={ep_success:.0f} "
                      f"chunks={chunk_step} learner_step={st['learner_step']} "
                      f"utd={st['effective_utd']:.2f} online={st['online_buffer']}")
                if wandb_project:
                    import wandb

                    wandb.log({
                        "episode/return": ep_ret, "episode/success": ep_success,
                        "learner_step": st["learner_step"], "env_chunks": chunk_step,
                        "effective_utd": st["effective_utd"],
                    })
                if min_utd > 0 and st["learner_step"] > 0:
                    session.wait_for_utd(min_utd)
                if eval_every > 0 and chunk_step - last_eval >= eval_every:
                    last_eval = chunk_step
                    sr, mret = evaluate(eval_episodes)
                    print(f"[eval] learner_step={st['learner_step']} "
                          f"success={sr:.1%} mean_return={mret:.2f}")
                    if wandb_project:
                        import wandb
                        wandb.log({"eval/success": sr, "eval/return": mret,
                                   "learner_step": st["learner_step"]})
                obs, _ = cenv.reset()
                a_base = np.asarray(query_base(obs), np.float32)
                ep_ret, ep_success = 0.0, 0.0
            else:
                obs, a_base = next_obs, next_a_base

            if time.time() - last_log > 30:
                last_log = time.time()
                print(f"[actor] heartbeat {session.status()}")

        if final_eval_episodes > 0:
            print(f"[final-eval] paired base-only vs residual over "
                  f"{final_eval_episodes} seeds...")
            base_sr, res_sr, b, c, p = paired_eval(final_eval_episodes)
            print(f"[final-eval] base-only={base_sr:.1%}  residual={res_sr:.1%}  "
                  f"discordant(base+/res-={b}, base-/res+={c})  McNemar p={p:.4f}  "
                  f"n={final_eval_episodes}")
            if wandb_project:
                import wandb
                wandb.run.summary.update({
                    "final/base_only_success": base_sr,
                    "final/residual_success": res_sr,
                    "final/delta": res_sr - base_sr,
                    "final/mcnemar_p": p,
                    "final/n": final_eval_episodes,
                })
    finally:
        session.stop_learner()
        cenv.close()


if __name__ == "__main__":
    tyro.cli(main)
