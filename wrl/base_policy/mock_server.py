"""Mock base-policy server for tests and smoke runs.

Modes:
  zeros    -> returns a zero chunk. Then a_full == residual, i.e. the residual
              agent reduces to chunked SAC from scratch. Use to validate the
              EXPO-FT pipeline without a real base policy.
  scripted -> returns a fixed non-trivial chunk so you can check the residual
              *improves on* a non-zero base.
"""

import numpy as np
import tyro

from wrl.base_policy.server import run_base_policy_in_thread, serve_base_policy


def zeros_policy(action_dim: int, horizon: int):
    chunk = np.zeros(action_dim * horizon, np.float32)

    def policy_fn(_obs):
        return chunk

    return policy_fn


def scripted_policy(action_dim: int, horizon: int, value: float = 0.1):
    chunk = np.full(action_dim * horizon, value, np.float32)

    def policy_fn(_obs):
        return chunk

    return policy_fn


def make_mock_policy(mode: str, action_dim: int, horizon: int):
    if mode == "zeros":
        return zeros_policy(action_dim, horizon)
    if mode == "scripted":
        return scripted_policy(action_dim, horizon)
    raise ValueError(f"unknown mock mode {mode!r} (expected 'zeros' or 'scripted')")


def run_mock_in_thread(mode: str, action_dim: int, horizon: int,
                       host: str = "localhost", port: int = 8200):
    return run_base_policy_in_thread(
        make_mock_policy(mode, action_dim, horizon), host=host, port=port
    )


def main(
    mode: str = "zeros",
    action_dim: int = 7,
    horizon: int = 8,
    host: str = "0.0.0.0",
    port: int = 8200,
):
    serve_base_policy(make_mock_policy(mode, action_dim, horizon), host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
