"""Round-trip test for the base-policy client + mock server."""

import time

import numpy as np

from wrl.base_policy import BasePolicyClient
from wrl.base_policy.mock_server import run_mock_in_thread

A, H = 7, 8
PORT = 8231


def _dummy_obs():
    return {
        "agentview": np.zeros((1, 8, 8, 3), np.uint8),
        "state": np.zeros((1, 5), np.float32),
    }


def test_zeros_and_scripted_roundtrip():
    server, _ = run_mock_in_thread("zeros", A, H, port=PORT)
    try:
        time.sleep(0.2)  # let the server bind
        client = BasePolicyClient(port=PORT)
        chunk = client.query(_dummy_obs())
        assert chunk.shape == (A * H,)
        assert np.allclose(chunk, 0.0)
        client.close()
    finally:
        server.shutdown()

    server2, _ = run_mock_in_thread("scripted", A, H, port=PORT + 1)
    try:
        time.sleep(0.2)
        client = BasePolicyClient(port=PORT + 1)
        chunk = client.query(_dummy_obs())
        assert chunk.shape == (A * H,)
        assert np.allclose(chunk, 0.1)
        client.close()
    finally:
        server2.shutdown()
