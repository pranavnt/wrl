"""Serve a (torch) DPPO diffusion policy over wrl's base-policy protocol.

Lets a JAX wrl actor query a DPPO-trained diffusion policy (e.g. the transport
expert) with `obs -> action chunk`, exactly like the flow-DP server, WITHOUT a
torch->jax weight port: the model runs in native torch using DPPO's own code
(correct by construction). Handles DPPO's min-max obs/action normalization.

Run with the wrl venv python + the DPPO repo on PYTHONPATH:

    DPPO=/home/yam/hil-serl/dppo PYTHONPATH=$DPPO \
    /home/yam/wrl/.venv/bin/python -m wrl.base_policy.dppo_server \
        --config $DPPO/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml \
        --checkpoint $DPPO/log/.../checkpoint/state_200.pt \
        --normalization $DPPO/data/robomimic/transport/normalization.npz \
        --port 8210
"""

import numpy as np
import torch
import tyro

from wrl.base_policy.server import serve_base_policy


def make_dppo_policy_fn(config, checkpoint, normalization, device="cuda", deterministic=True):
    import os
    import hydra
    from omegaconf import OmegaConf

    # the config interpolates ${oc.env:DPPO_LOG_DIR/DATA_DIR}; derive from the
    # config path (<dppo_root>/cfg/...) so they resolve without manual setup.
    dppo_root = config.split("/cfg/")[0]
    os.environ.setdefault("DPPO_LOG_DIR", os.path.join(dppo_root, "log"))
    os.environ.setdefault("DPPO_DATA_DIR", os.path.join(dppo_root, "data"))
    os.environ.setdefault("DPPO_WANDB_ENTITY", "none")

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", lambda x: int(np.ceil(x)), replace=True)
    OmegaConf.register_new_resolver("round_down", lambda x: int(np.floor(x)), replace=True)
    cfg = OmegaConf.load(config)
    cfg.device = device                      # override the config's hardcoded cuda:0
    model = hydra.utils.instantiate(cfg.model)
    blob = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    norm = np.load(normalization)
    obs_min, obs_max = norm["obs_min"].astype(np.float32), norm["obs_max"].astype(np.float32)
    a_min, a_max = norm["action_min"].astype(np.float32), norm["action_max"].astype(np.float32)
    cond_steps = int(cfg.get("cond_steps", 1))
    obs_dim = int(cfg.obs_dim)
    print(f"[dppo-server] loaded {checkpoint}\n[dppo-server] obs_dim={obs_dim} "
          f"cond_steps={cond_steps} action_dim={int(cfg.action_dim)} ckpt eval=deterministic")

    def _cond(observation):
        state = np.asarray(observation["state"], np.float32).reshape(-1, obs_dim)
        if state.shape[0] < cond_steps:                      # pad oldest frame
            state = np.concatenate([state[:1]] * (cond_steps - state.shape[0]) + [state], 0)
        state = state[-cond_steps:]
        state = 2.0 * ((state - obs_min) / (obs_max - obs_min + 1e-6) - 0.5)
        state = np.clip(state, -1.0, 1.0)
        return {"state": torch.as_tensor(state[None], device=device)}   # (1, cond_steps, obs_dim)

    def policy_fn(observation):
        with torch.no_grad():
            samples = model(cond=_cond(observation), deterministic=deterministic)
        act = samples.trajectories.cpu().numpy()[0]          # (Ta, Da) in [-1,1]
        act = (act + 1.0) / 2.0 * (a_max - a_min) + a_min     # unnormalize
        return act.reshape(-1).astype(np.float32)             # (Ta*Da,)

    def value_fn(observation):
        """V*(s) from the DPPO critic — the PAM-gate value (state value)."""
        with torch.no_grad():
            v = model.critic(_cond(observation)).view(-1).cpu().numpy()
        return float(v[0])

    def value_batch_fn(states):
        """Batched V*: states is (N, obs_dim) or a list of (obs_dim,) -> (N,)
        values in ONE critic forward (vs N separate calls; for per-step PAM)."""
        st = np.asarray(states, np.float32).reshape(-1, obs_dim)
        st = 2.0 * ((st - obs_min) / (obs_max - obs_min + 1e-6) - 0.5)
        st = np.clip(st, -1.0, 1.0)[:, None, :]              # (N, 1, obs_dim)
        with torch.no_grad():
            v = model.critic({"state": torch.as_tensor(st, device=device)}).view(-1)
        return v.cpu().numpy()

    policy_fn.value_fn = value_fn          # attach for in-process use
    policy_fn.value_batch_fn = value_batch_fn
    return policy_fn


def make_dppo_expert(config, checkpoint, normalization, device="cuda", deterministic=True):
    """Returns (action_fn, value_fn): the DPPO expert pi_h and its V*(s) critic,
    loaded once in-process for the HIL loop (action only on intervention; V* every
    step for PAM gating)."""
    fn = make_dppo_policy_fn(config, checkpoint, normalization, device=device,
                             deterministic=deterministic)
    return fn, fn.value_batch_fn


def main(config: str, checkpoint: str, normalization: str,
         host: str = "0.0.0.0", port: int = 8210, device: str = "cuda"):
    fn = make_dppo_policy_fn(config, checkpoint, normalization, device=device)
    serve_base_policy(fn, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
