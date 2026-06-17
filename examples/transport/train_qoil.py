"""Q-OIL on transport (pixels): residual SAC on a suboptimal pixel DP base, with
PAM-gated expert interventions and decoupled optimistic critics.

  pi_theta : residual SAC (QOILAgent) on the 34% pixel DP base
  pi_h     : the 94% DPPO expert (served in-process, torch)
  V*       : the DPPO critic, for the PAM stagnation gate
  gate g   : intervene when V*(s_t) - V*(s_{t-k}) < delta  (k=12, delta=0, prob 1)

Algorithm variants (same loop, different flags):
  --algo qoil   : decoupled critics + optimism bonus + BC reg   (full Q-OIL)
  --algo bc_rl  : intervention_bonus=0, BC reg on              (BC+RL baseline)
  --algo hil    : intervention_bonus=0, BC off                 (HIL baseline)

    python examples/transport/train_qoil.py --algo qoil \
        --base checkpoints/flowdp_transport_pixel_avg_step45000.pkl \
        --dppo-checkpoint .../state_200.pt
"""

import collections
import time

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import wrl
from envs.chunk_wrapper import ActionChunkWrapper
from envs.robomimic_pixels import make_robomimic_pixel_env
from wrl.agents.qoil import QOILAgent
from wrl.data import residual_extra_fields
from wrl.diffusion.flow_policy import FlowPolicy

_DPPO = "/home/yam/hil-serl/dppo"


def main(
    base: str,                       # 34% pixel DP base (FlowPolicy)
    dppo_checkpoint: str,            # 94% DPPO expert state_200.pt
    dppo_config: str = f"{_DPPO}/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml",
    dppo_norm: str = f"{_DPPO}/data/robomimic/transport/normalization.npz",
    dataset_path: str = "data/robomimic/transport/ph/image_84.hdf5",
    algo: str = "qoil",              # qoil | bc_rl | hil
    pam_k: int = 12,
    pam_delta: float = 0.0,
    intervention_bonus: float = 0.1,
    bc_weight: float = 0.1,
    edit_scale: float = 0.25,
    discount: float = 0.99,
    image_size: int = 84,
    max_episode_steps: int = 800,
    batch_size: int = 256,
    cta_ratio: int = 4,
    training_starts: int = 1000,
    random_chunks: int = 50,
    max_steps: int = 200_000,
    eval_every: int = 5000,
    eval_episodes: int = 20,
    http_port: int = 5610,
    seed: int = 0,
    wandb_project: str = "",
):
    bonus = intervention_bonus if algo == "qoil" else 0.0
    bcw = 0.0 if algo == "hil" else bc_weight
    print(f"[qoil] algo={algo} bonus={bonus} bc_weight={bcw} pam_k={pam_k} delta={pam_delta}")

    # ---- frozen base DP (pixels) + DPPO expert (action + V*) -------------
    fp = FlowPolicy.load(base)
    Ta, d_a = fp.config["Ta"], fp.config["d_a"]
    base_obs_hist = fp.config["obs_history"]
    chunk_dim = Ta * d_a
    from wrl.base_policy.dppo_server import make_dppo_expert
    expert_act, expert_val = make_dppo_expert(dppo_config, dppo_checkpoint, dppo_norm, device="cuda")

    env = make_robomimic_pixel_env(dataset_path, image_size=image_size,
                                   max_episode_steps=max_episode_steps, include_lowdim=True)
    assert env.action_space.shape[0] == d_a
    cenv = ActionChunkWrapper(env, d_a, Ta, discount=discount, frame_history=base_obs_hist)
    image_keys = env.image_keys
    # "lowdim" is in the obs dict (for V*/expert) but unused by the agent's
    # EncodingWrapper (reads only image_keys + "state"); it just rides along.
    def strip(o):
        return o

    def base_chunk():                # 34% pixel DP base chunk at the boundary
        oh = cenv.base_obs(base_obs_hist)
        return np.asarray(jax.device_get(fp.sample_chunk(jax.device_put(oh),
                          jax.random.PRNGKey(np.random.randint(1 << 30)))), np.float32)

    sample_obs = strip(env.observation_space.sample())
    agent = QOILAgent.create_pixels(
        jax.random.PRNGKey(seed), sample_obs, np.zeros(chunk_dim, np.float32),
        np.zeros(chunk_dim, np.float32), action_dim=d_a, horizon=Ta, image_keys=image_keys,
        edit_scale=edit_scale, intervention_bonus=bonus, bc_weight=bcw,
        critic_ensemble_size=2, discount_per_step=discount)
    agent = jax.tree_util.tree_map(jnp.asarray, agent)

    cfg = wrl.Config(batch_size=batch_size, cta_ratio=cta_ratio, training_starts=training_starts,
                     replay_buffer_capacity=200_000, demo_buffer_capacity=200_000,
                     max_steps=max_steps, image_keys=image_keys,
                     extra_fields=residual_extra_fields(chunk_dim))
    session = wrl.Session(agent, cenv, cfg, rng_seed=seed)
    session.start_learner()
    session.start_server(port=http_port)

    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=dict(algo=algo, bonus=bonus, bc_weight=bcw,
                   pam_k=pam_k, pam_delta=pam_delta, edit_scale=edit_scale))

    # ---- PAM stagnation gate (per env-step V* history) ------------------
    vhist = collections.deque(maxlen=pam_k + 1)

    def update_vhist(step_lowdims):
        for s in step_lowdims:
            vhist.append(expert_val({"state": s}))

    def stagnating():
        return len(vhist) > pam_k and (vhist[-1] - vhist[0]) < pam_delta

    def evaluate(n):
        succ = []
        for ep in range(n):
            o, _ = cenv.reset(seed=40000 + ep)
            vhist.clear()
            done = trunc = False; s = 0.0
            while not (done or trunc):
                b = base_chunk()
                full = np.asarray(session.policy.sample(strip(o), b, argmax=True), np.float32)
                o, r, done, trunc, info = cenv.step(full)
                s = max(s, float(info.get("success", 0.0)))
            succ.append(int(s > 0))
        return float(np.mean(succ))

    obs, _ = cenv.reset(seed=seed)
    vhist.clear()
    a_base = base_chunk()
    intervene_next = False
    chunk_step, last_eval = 0, 0
    ep_ret, ep_succ, n_int = 0.0, 0.0, 0
    t0 = time.time()
    try:
        while session.status()["learner_running"]:
            lowdim = np.asarray(obs["lowdim"], np.float32)
            if chunk_step < random_chunks:
                full = cenv.action_space.sample(); did_int = False
            elif intervene_next:                     # PAM-gated expert takeover
                full = expert_act({"state": lowdim}); did_int = True
            else:
                full = np.asarray(session.policy.sample(strip(obs), a_base), np.float32); did_int = False

            next_obs, r, done, trunc, info = cenv.step(full)
            update_vhist(info.get("step_lowdims", []))
            next_a_base = base_chunk()
            session.buffer.add(strip(obs), full, strip(next_obs), r, done,
                               is_intervention=did_int, base_actions=a_base,
                               next_base_actions=next_a_base)
            ep_ret += r; ep_succ = max(ep_succ, float(info.get("success", 0.0)))
            n_int += int(did_int); chunk_step += 1
            intervene_next = stagnating()            # gate the NEXT chunk

            if done or trunc:
                session.record_episode(ep_ret)
                st = session.status()
                print(f"[qoil] ep_ret={ep_ret:.2f} success={ep_succ:.0f} interventions={n_int} "
                      f"chunks={chunk_step} learner={st['learner_step']} utd={st['effective_utd']:.1f}")
                if wandb_project:
                    import wandb
                    wandb.log({"episode/return": ep_ret, "episode/success": ep_succ,
                               "episode/interventions": n_int, "learner_step": st["learner_step"]})
                if eval_every and chunk_step - last_eval >= eval_every:
                    last_eval = chunk_step
                    sr = evaluate(eval_episodes)
                    print(f"[eval] learner={st['learner_step']} success={sr:.1%}")
                    try:
                        import os, pickle
                        ck = f"checkpoints/qoil_{algo}_transport.pkl"
                        os.makedirs("checkpoints", exist_ok=True)
                        ag = session.snapshot_agent()
                        with open(ck + ".tmp", "wb") as f:
                            pickle.dump({"params": jax.tree_util.tree_map(np.asarray, ag.state.params),
                                         "eval": sr, "learner": st["learner_step"]}, f)
                        os.replace(ck + ".tmp", ck)
                    except OSError as e:
                        print(f"[warn] checkpoint save skipped ({e})")
                    if wandb_project:
                        import wandb
                        wandb.log({"eval/success": sr, "learner_step": st["learner_step"]})
                obs, _ = cenv.reset(); vhist.clear(); a_base = base_chunk()
                intervene_next = False; ep_ret, ep_succ, n_int = 0.0, 0.0, 0
            else:
                obs, a_base = next_obs, next_a_base
    finally:
        session.stop_learner(); cenv.close()


if __name__ == "__main__":
    tyro.cli(main)
