"""Action-chunk wrapper: turn a single-step env into a chunk-MDP.

The EXPO-FT actor executes an `H`-step action chunk per decision. This wrapper
accepts a flattened chunk `(A*H,)`, reshapes to `(H, A)`, steps the base env up
to `H` times, and returns the chunk-level transition: the boundary observation,
the discounted-sum reward `sum_i gamma^i r_i`, and done/trunc. The matching
agent uses `discount ** H` as its bootstrap, so the chunk MDP is consistent.
"""

import gymnasium as gym
import numpy as np


class ActionChunkWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_dim: int, horizon: int, discount: float = 0.97):
        super().__init__(env)
        self.action_dim = action_dim
        self.horizon = horizon
        self.discount = discount

        base = env.action_space
        assert base.shape == (action_dim,), (base.shape, action_dim)
        low = np.tile(np.asarray(base.low, np.float32), horizon)
        high = np.tile(np.asarray(base.high, np.float32), horizon)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, chunk):
        chunk = np.asarray(chunk, np.float32).reshape(self.horizon, self.action_dim)
        total_reward, gamma = 0.0, 1.0
        obs, done, trunc, info = None, False, False, {}
        for i in range(self.horizon):
            obs, reward, done, trunc, info = self.env.step(chunk[i])
            total_reward += gamma * float(reward)
            gamma *= self.discount
            if done or trunc:
                break
        return obs, total_reward, done, trunc, info
