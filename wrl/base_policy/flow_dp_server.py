"""Serve a trained flow-matching pixel DP (FlowPolicy) as the EXPO-FT base.

The actor must send the obs-history the flow DP was trained on (use the chunk
wrapper's `base_obs(obs_history)`): each image key (obs_history, H, W, C) uint8
and 'state' (obs_history, proprio). Returns a flat (Ta*d_a,) base chunk.

    uv run python -m wrl.base_policy.flow_dp_server \
        --checkpoint checkpoints/flowdp_toolhang.pkl --port 8200
"""

import jax
import numpy as np
import tyro

from wrl.base_policy.server import run_base_policy_in_thread, serve_base_policy
from wrl.diffusion.flow_policy import FlowPolicy


def make_flow_dp_policy_fn(checkpoint: str, seed: int = 0):
    fp = FlowPolicy.load(checkpoint)
    rng = {"key": jax.random.PRNGKey(seed)}

    def policy_fn(observation: dict) -> np.ndarray:
        rng["key"], k = jax.random.split(rng["key"])
        obs = {kk: np.asarray(vv) for kk, vv in observation.items()}
        chunk = fp.sample_chunk(jax.device_put(obs), k)
        return np.asarray(jax.device_get(chunk), np.float32)

    return fp, policy_fn


def run_flow_dp_in_thread(checkpoint: str, host="localhost", port=8200, seed=0):
    _, fn = make_flow_dp_policy_fn(checkpoint, seed)
    return run_base_policy_in_thread(fn, host=host, port=port)


def main(checkpoint: str, host: str = "0.0.0.0", port: int = 8200, seed: int = 0):
    fp, fn = make_flow_dp_policy_fn(checkpoint, seed)
    print(f"[flow-dp-server] Ta={fp.config['Ta']} d_a={fp.config['d_a']} "
          f"obs_history={fp.config['image_shape'][0]} chunk_dim={fp.config['Ta']*fp.config['d_a']}")
    serve_base_policy(fn, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
