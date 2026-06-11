"""Serve a trained Diffusion Policy checkpoint as the EXPO-FT base policy.

    uv run python -m wrl.base_policy.dp_server \
        --checkpoint checkpoints/dp_toolhang.pkl --port 8200
"""

import jax
import numpy as np
import tyro

from wrl.base_policy.server import run_base_policy_in_thread, serve_base_policy
from wrl.diffusion.policy import DiffusionPolicy


def make_dp_policy_fn(checkpoint: str, seed: int = 0):
    """Load a DP checkpoint, return a `policy_fn(obs) -> flat (A*H,) chunk`."""
    dp = DiffusionPolicy.load(checkpoint)
    rng = {"key": jax.random.PRNGKey(seed)}

    def policy_fn(observation: dict) -> np.ndarray:
        rng["key"], k = jax.random.split(rng["key"])
        obs = {kk: np.asarray(vv) for kk, vv in observation.items()}
        chunk = dp.sample(obs, k)
        return np.asarray(jax.device_get(chunk), np.float32)

    return policy_fn


def run_dp_in_thread(checkpoint: str, host: str = "localhost", port: int = 8200, seed: int = 0):
    return run_base_policy_in_thread(make_dp_policy_fn(checkpoint, seed), host=host, port=port)


def main(checkpoint: str, host: str = "0.0.0.0", port: int = 8200, seed: int = 0):
    serve_base_policy(make_dp_policy_fn(checkpoint, seed), host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
