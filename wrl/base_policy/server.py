"""Generic base-policy WebSocket server.

`serve_base_policy(policy_fn, ...)` exposes any `policy_fn(obs_dict) -> chunk`
over the same msgpack-numpy protocol `BasePolicyClient` speaks. The diffusion
policy server (step 9) and the mock server both build on this.
"""

import threading

import msgpack
import msgpack_numpy
import numpy as np
from websockets.sync.server import serve

msgpack_numpy.patch()


def _handler_factory(policy_fn):
    def handler(websocket):
        for raw in websocket:
            try:
                req = msgpack.unpackb(raw)
                if req.get("operation") != "act":
                    websocket.send(msgpack.packb(
                        {"status": "error", "message": f"unknown op {req.get('operation')}"}
                    ))
                    continue
                action = np.asarray(policy_fn(req["observation"]), dtype=np.float32)
                websocket.send(msgpack.packb({"action": action}))
            except Exception as e:  # report back instead of dropping the socket
                websocket.send(msgpack.packb({"status": "error", "message": str(e)}))

    return handler


def serve_base_policy(policy_fn, host: str = "0.0.0.0", port: int = 8200):
    """Blocking serve loop."""
    with serve(_handler_factory(policy_fn), host, port) as server:
        print(f"[base-policy] serving on ws://{host}:{port}")
        server.serve_forever()


def run_base_policy_in_thread(policy_fn, host: str = "localhost", port: int = 8200):
    """Start the server in a daemon thread. Returns (server, thread); call
    `server.shutdown()` to stop. Handy for tests / single-box runs."""
    server = serve(_handler_factory(policy_fn), host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name=f"base-policy-{port}")
    thread.start()
    return server, thread
