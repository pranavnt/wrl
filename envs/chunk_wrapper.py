"""Action-chunk wrapper: turn a single-step env into a chunk-MDP.

The EXPO-FT actor executes an `H`-step action chunk per decision. This wrapper
accepts a flattened chunk `(A*H,)`, reshapes to `(H, A)`, steps the base env up
to `H` times, and returns the chunk-level transition: the boundary observation,
the discounted-sum reward `sum_i gamma^i r_i`, and done/trunc. The matching
agent uses `discount ** H` as its bootstrap, so the chunk MDP is consistent.
"""

from collections import deque

import gymnasium as gym
import numpy as np


class ActionChunkWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_dim: int, horizon: int, discount: float = 0.97,
                 frame_history: int = 1):
        super().__init__(env)
        self.action_dim = action_dim
        self.horizon = horizon
        self.discount = discount

        base = env.action_space
        assert base.shape == (action_dim,), (base.shape, action_dim)
        low = np.tile(np.asarray(base.low, np.float32), horizon)
        high = np.tile(np.asarray(base.high, np.float32), horizon)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # rolling buffer of recent single-frame obs (for a base policy that needs
        # an obs history at the chunk boundary, e.g. the flow DP trained with
        # obs_history>1). Tracks the env's per-step obs through chunk execution.
        self._fh = deque(maxlen=max(1, frame_history))

    def _push(self, obs):
        self._fh.append(obs)

    def base_obs(self, k):
        """Stack the last `k` single-frame obs into (k, ...) per key — the
        consecutive-frame history a base policy expects. Pads by repeating the
        oldest available frame."""
        frames = list(self._fh)[-k:]
        while len(frames) < k:
            frames.insert(0, frames[0])
        out = {}
        for key in frames[0]:
            # each frame's value is (1, ...) (leading single-frame dim); squeeze it
            out[key] = np.concatenate([f[key] for f in frames], axis=0)  # (k, ...)
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._fh.clear()
        self._push(obs)
        return obs, info

    def step(self, chunk):
        chunk = np.asarray(chunk, np.float32).reshape(self.horizon, self.action_dim)
        total_reward, gamma = 0.0, 1.0
        obs, done, trunc, info = None, False, False, {}
        step_lowdims = []   # per-env-step lean state (for per-step PAM V* tracking)
        for i in range(self.horizon):
            obs, reward, done, trunc, info = self.env.step(chunk[i])
            self._push(obs)
            if isinstance(obs, dict) and "lowdim" in obs:
                step_lowdims.append(np.asarray(obs["lowdim"], np.float32))
            total_reward += gamma * float(reward)
            gamma *= self.discount
            if done or trunc:
                break
        info = {**info, "step_lowdims": step_lowdims}
        return obs, total_reward, done, trunc, info
