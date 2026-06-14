"""Load a rendered robomimic image dataset into:

  * DP training chunks  -- (obs, flattened H-step action chunk) for the base
    Diffusion Policy (behavior cloning).
  * residual warm-start transitions -- chunk-level transitions for the residual
    SAC demo buffer (base_actions == actions, i.e. residual 0).

Obs layout matches `envs.robomimic_pixels.RoboMimicPixelEnv`: each image key is
(1, H, W, C) uint8 and `state` is (1, proprio) float32, with proprio built by
concatenating `proprio_keys` in order. Pass `env.image_keys` / `env.proprio_keys`
so the base policy and the RL env are guaranteed consistent.
"""

import h5py
import numpy as np


class RobomimicPixelData:
    def __init__(self, dataset_path, image_keys, proprio_keys, horizon):
        self.image_keys = tuple(image_keys)
        self.proprio_keys = tuple(proprio_keys)
        self.H = horizon

        imgs = {k: [] for k in self.image_keys}
        proprio, actions, rewards, dones, demo_of_step = [], [], [], [], []

        with h5py.File(dataset_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
            for di, demo in enumerate(demos):
                g = f["data"][demo]
                obs = g["obs"]
                T = g["actions"].shape[0]
                for k in self.image_keys:
                    imgs[k].append(np.asarray(obs[k][:], np.uint8))
                p = np.concatenate(
                    [np.asarray(obs[k][:], np.float32).reshape(T, -1) for k in self.proprio_keys],
                    axis=1,
                )
                proprio.append(p)
                actions.append(np.asarray(g["actions"][:], np.float32))
                rewards.append(np.asarray(g["rewards"][:], np.float32))
                dones.append(np.asarray(g["dones"][:], np.float32))
                demo_of_step.append(np.full(T, di, np.int32))

        self.imgs = {k: np.concatenate(v, axis=0) for k, v in imgs.items()}
        self.proprio = np.concatenate(proprio, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.dones = np.concatenate(dones, axis=0)
        self.demo_of_step = np.concatenate(demo_of_step, axis=0)
        self.action_dim = self.actions.shape[1]
        self.N = self.actions.shape[0]

        # valid chunk start = same demo across [s, s+H-1]
        same_demo = self.demo_of_step[: self.N - self.H + 1] == self.demo_of_step[self.H - 1:]
        self.valid_starts = np.nonzero(same_demo)[0]

        # per-step demo boundaries (for obs-history clamping / action-window padding)
        self.demo_start = np.zeros(self.N, np.int64)
        self.demo_end = np.zeros(self.N, np.int64)
        uniq, first = np.unique(self.demo_of_step, return_index=True)
        for di, s0 in zip(uniq, first):
            idx = np.nonzero(self.demo_of_step == di)[0]
            self.demo_start[idx] = idx[0]
            self.demo_end[idx] = idx[-1]

    def action_stats(self):
        """Per-dim mean/std over all actions (for the flow policy normalizer)."""
        return self.actions.mean(0), self.actions.std(0)

    def flow_sample(self, batch_size, Tp, obs_history, rng):
        """Batch for flow-matching DP training:
        observations[<img>] : (B, obs_history, H, W, C) uint8  (frames clamped to demo start)
        observations['state']: (B, obs_history, proprio)
        actions             : (B, Tp, d_a)  (window held at last action past demo end)
        """
        s = rng.integers(0, self.N, batch_size)
        # obs history: frames [s-h+1 .. s] clamped to the demo start
        offs = np.arange(-obs_history + 1, 1)  # (h,)
        fidx = np.clip(s[:, None] + offs[None], self.demo_start[s][:, None], s[:, None])
        obs = {k: self.imgs[k][fidx] for k in self.image_keys}  # (B,h,H,W,C)
        obs["state"] = self.proprio[fidx]  # (B,h,proprio)
        # action window [s .. s+Tp-1] clamped to demo end (hold last)
        aidx = np.clip(s[:, None] + np.arange(Tp)[None], 0, self.demo_end[s][:, None])
        return {"observations": obs, "actions": self.actions[aidx]}  # (B,Tp,d_a)

    def _chunk(self, starts):
        idx = starts[:, None] + np.arange(self.H)[None, :]  # (B, H)
        return self.actions[idx].reshape(len(starts), self.H * self.action_dim)

    def _obs_at(self, idx):
        out = {k: self.imgs[k][idx][:, None] for k in self.image_keys}  # (B,1,H,W,C)
        out["state"] = self.proprio[idx][:, None]  # (B,1,proprio)
        return out

    # ---- DP behavior-cloning batches ------------------------------------

    def dp_sample(self, batch_size, rng):
        starts = self.valid_starts[rng.integers(0, len(self.valid_starts), batch_size)]
        return {"observations": self._obs_at(starts), "actions": self._chunk(starts)}

    # ---- residual warm-start transitions --------------------------------

    def obs_history(self, t, obs_history):
        """Consecutive `obs_history` frames ending at step t, clamped to the
        demo start — the obs format a base policy (e.g. the flow DP) expects."""
        offs = np.arange(-obs_history + 1, 1)
        idx = np.clip(t + offs, self.demo_start[t], t)
        out = {k: self.imgs[k][idx] for k in self.image_keys}
        out["state"] = self.proprio[idx]
        return out

    def residual_transitions(self, discount, max_transitions=None, seed=0,
                             base_query_fn=None, base_obs_history=2):
        """Chunk-level transitions for the residual demo buffer.

        Default: base == full (residual 0), so the demo seeds with base = the
        demo action. If `base_query_fn(obs_history_dict) -> (H*A,)` is given
        (e.g. the flow-DP base), the seed is made BASE-CONSISTENT: the stored
        full `actions` stays the demo chunk (so d(s,a)=s' is exact) and we only
        relabel `base_actions`/`next_base_actions` to the base policy's chunk at
        s / s+H. The implied residual = demo - base is small because the DP is a
        BC of these demos. Only starts whose next chunk also exists are used."""
        H, A = self.H, self.action_dim
        ok = self.valid_starts[
            (self.valid_starts + 2 * H - 1 < self.N)
            & (self.demo_of_step[np.clip(self.valid_starts + 2 * H - 1, 0, self.N - 1)]
               == self.demo_of_step[self.valid_starts])
        ]
        if max_transitions is not None and len(ok) > max_transitions:
            ok = np.random.default_rng(seed).choice(ok, max_transitions, replace=False)
            ok.sort()

        gammas = discount ** np.arange(H, dtype=np.float32)
        out = []
        for s in ok:
            chunk = self.actions[s : s + H].reshape(H * A).astype(np.float32)
            next_chunk = self.actions[s + H : s + 2 * H].reshape(H * A).astype(np.float32)
            if base_query_fn is not None:
                base = np.asarray(base_query_fn(self.obs_history(s, base_obs_history)), np.float32)
                next_base = np.asarray(base_query_fn(self.obs_history(s + H, base_obs_history)), np.float32)
            else:
                base, next_base = chunk, next_chunk
            r = float(np.dot(gammas, self.rewards[s : s + H]))
            done = bool(self.dones[s : s + H].any())
            obs = {k: self.imgs[k][s][None] for k in self.image_keys}
            obs["state"] = self.proprio[s][None]
            nobs = {k: self.imgs[k][s + H][None] for k in self.image_keys}
            nobs["state"] = self.proprio[s + H][None]
            out.append(
                dict(
                    observations=obs,
                    next_observations=nobs,
                    actions=chunk,          # full action = demo chunk -> d(s,a)=s' exact
                    base_actions=base,
                    next_base_actions=next_base,
                    rewards=r,
                    masks=0.0 if done else 1.0,
                    dones=done,
                    is_intervention=True,
                )
            )
        return out


def load_robomimic_pixels(dataset_path, image_keys, proprio_keys, horizon):
    return RobomimicPixelData(dataset_path, image_keys, proprio_keys, horizon)


class RobomimicStateData:
    """Low-dim (state) version of RobomimicPixelData for a STATE diffusion policy.

    Observation is the full low-dim state (robot keys + object) concatenated in the
    SAME order as `envs.robomimic_state.RoboMimicStateEnv`, so a DP trained here
    matches the env it is rolled out / deployed on. No images."""

    def __init__(self, dataset_path, horizon, lean=False):
        from envs.robomimic_state import _robot_obs_keys

        self.H = horizon
        states, actions, rewards, dones, demo_of_step = [], [], [], [], []
        with h5py.File(dataset_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
            obs_keys = list(f["data"][demos[0]]["obs"].keys())
            robot_keys = _robot_obs_keys(obs_keys, lean=lean)
            object_key = "object" if "object" in obs_keys else "object-state"
            self.state_keys = list(robot_keys) + [object_key]
            for di, demo in enumerate(demos):
                g = f["data"][demo]
                T = g["actions"].shape[0]
                s = np.concatenate(
                    [np.asarray(g["obs"][k][:], np.float32).reshape(T, -1) for k in self.state_keys],
                    axis=1,
                )
                states.append(s)
                actions.append(np.asarray(g["actions"][:], np.float32))
                rewards.append(np.asarray(g["rewards"][:], np.float32))
                dones.append(np.asarray(g["dones"][:], np.float32))
                demo_of_step.append(np.full(T, di, np.int32))

        self.proprio = np.concatenate(states, axis=0)   # named to mirror the pixel loader
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.dones = np.concatenate(dones, axis=0)
        self.demo_of_step = np.concatenate(demo_of_step, axis=0)
        self.action_dim = self.actions.shape[1]
        self.state_dim = self.proprio.shape[1]
        self.N = self.actions.shape[0]

        self.demo_start = np.zeros(self.N, np.int64)
        self.demo_end = np.zeros(self.N, np.int64)
        for di in np.unique(self.demo_of_step):
            idx = np.nonzero(self.demo_of_step == di)[0]
            self.demo_start[idx] = idx[0]
            self.demo_end[idx] = idx[-1]

    def action_stats(self):
        return self.actions.mean(0), self.actions.std(0)

    def flow_sample(self, batch_size, Tp, obs_history, rng):
        s = rng.integers(0, self.N, batch_size)
        offs = np.arange(-obs_history + 1, 1)
        fidx = np.clip(s[:, None] + offs[None], self.demo_start[s][:, None], s[:, None])
        obs = {"state": self.proprio[fidx]}  # (B, h, state_dim)
        aidx = np.clip(s[:, None] + np.arange(Tp)[None], 0, self.demo_end[s][:, None])
        return {"observations": obs, "actions": self.actions[aidx]}


def load_robomimic_state(dataset_path, horizon, lean=False):
    return RobomimicStateData(dataset_path, horizon, lean=lean)
