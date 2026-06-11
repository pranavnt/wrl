"""Client for the base-policy server.

The EXPO-FT actor queries a separate (frozen) base policy each chunk:

    obs (dict of np arrays)  ->  base action chunk  (flattened, shape (A*H,))

Uses a single persistent WebSocket connection (msgpack-numpy framed) so the
hot path is cheap, with automatic reconnect. The base model itself lives in
the server process — wrl never loads it.
"""

import logging
import time

import msgpack
import msgpack_numpy
import numpy as np
import websockets.exceptions
import websockets.sync.client

msgpack_numpy.patch()

# Handshake/connection errors worth retrying (also covers SSH reverse tunnels).
_RETRY_EXC = (
    ConnectionRefusedError,
    websockets.exceptions.InvalidMessage,
    EOFError,
    OSError,
    ConnectionResetError,
)


class BasePolicyClient:
    """Persistent WebSocket client. `query(obs) -> np.ndarray` of shape (A*H,)."""

    def __init__(self, host: str = "localhost", port: int = 8200, *, timeout: float = 30.0):
        self.host = host
        self.port = port
        self._uri = f"ws://{host}:{port}"
        self._timeout = timeout
        self._conn = None

    def _connect(self):
        while True:
            try:
                return websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, close_timeout=30,
                )
            except _RETRY_EXC as e:
                logging.info("waiting for base-policy server (%s: %s)...",
                             type(e).__name__, e)
                time.sleep(2)

    def _get(self):
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def query(self, observation: dict) -> np.ndarray:
        """Return the base action chunk for `observation` as a flat (A*H,) array."""
        req = msgpack.packb({"operation": "act", "observation": observation})
        last_exc = None
        for attempt in range(3):
            try:
                conn = self._get()
                conn.send(req)
                resp = msgpack.unpackb(conn.recv())
                if resp.get("status") == "error":
                    raise RuntimeError(f"base-policy server error: {resp.get('message')}")
                return np.asarray(resp["action"], dtype=np.float32)
            except (websockets.exceptions.ConnectionClosed, *_RETRY_EXC) as e:
                last_exc = e
                self._close()
                if attempt < 2:
                    time.sleep(1)
        raise RuntimeError(f"base-policy query failed after retries: {last_exc}")

    def close(self):
        self._close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._close()
