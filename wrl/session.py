"""wrl Session — the user-facing surface.

`Session` owns the agent, the replay buffers (online + demo/offline), and an
optional background learner thread that drains the buffers into gradient
updates. Users write their own env loop:

    session = wrl.Session(agent, env, config)
    session.start_learner()             # background updates

    obs, _ = env.reset()
    while session.status()["learner_running"]:
        action = session.policy.sample(obs)
        next_obs, r, done, _, info = env.step(action)
        session.buffer.add(obs, action, next_obs, r, done)
        if done:
            session.wait_for_utd(min_utd=5.0)   # let the learner catch up
            obs, _ = env.reset()
        else:
            obs = next_obs

For the EXPO-FT residual path the policy and buffer also carry a base action
chunk (queried from a separate base-policy server); see `_PolicyFacade.sample`
and `_BufferFacade.add`.

For fleet training, `session.start_server(port=5588)` exposes the same
buffer-add / params-fetch / status surface over HTTP so remote actor
processes can join the run.

Distilled from alder (agentlace-free; Quart-Trio HTTP transport).
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import hypercorn.config
import hypercorn.trio
import jax
import msgpack
import msgpack_numpy
import numpy as np
import trio
import wandb
from quart import Response, request
from quart_trio import QuartTrio

from wrl.config import Config
from wrl.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    ReplayBufferDataStore,
)
from wrl.utils.train_utils import concat_batches

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_np(x):
    if isinstance(x, dict):
        return {k: _to_np(v) for k, v in x.items()}
    return np.asarray(x)


def _to_pylog(info):
    """Recursively convert jax/numpy arrays in an info dict to python floats
    so wandb / json see plain numbers. Pass-through for non-numeric leaves."""
    if isinstance(info, dict):
        return {k: _to_pylog(v) for k, v in info.items()}
    if isinstance(info, (jax.Array, np.ndarray, np.number, int, float)):
        return float(np.asarray(info))
    return info


def _serialize_params(params) -> bytes:
    flat = jax.tree_util.tree_map(np.asarray, params)
    return msgpack.packb(flat)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class Session:
    """The single source of truth for one training run.

    Public surface: `session.policy.sample(obs[, base_action])`,
    `session.buffer.add(...)`, `session.utd()`, `session.wait_for_utd(...)`,
    `session.start_learner()`, `session.train_step()`, `session.start_server(...)`,
    `session.status()`, `session.preload_demos(...)`. Everything else is internal.
    """

    def __init__(self, agent, env, config: Config, *, rng_seed: int = 0):
        self._config = config
        self._agent = agent

        if config.image_keys:
            buf_cls = MemoryEfficientReplayBufferDataStore
            buf_kwargs = {"image_keys": config.image_keys}
        else:
            buf_cls = ReplayBufferDataStore
            buf_kwargs = {}

        self._online_buffer = buf_cls(
            env.observation_space, env.action_space,
            capacity=config.replay_buffer_capacity,
            extra_fields=config.extra_fields, **buf_kwargs,
        )
        self._demo_buffer = buf_cls(
            env.observation_space, env.action_space,
            capacity=config.demo_buffer_capacity,
            extra_fields=config.extra_fields, **buf_kwargs,
        )

        self._counter_lock = threading.Lock()
        self._rng_lock = threading.Lock()
        self._rng = jax.random.PRNGKey(rng_seed)
        self._learner_step = 0
        self._env_steps = 0
        self._params_version = 0
        self._episode_returns: collections.deque = collections.deque(maxlen=100)
        self._started_at = time.time()

        self._shutdown_event = threading.Event()
        self._learner_thread: Optional[threading.Thread] = None
        self._server_thread: Optional[threading.Thread] = None

        # cached serialized params for /params endpoint
        self._params_bytes: Optional[bytes] = None
        self._params_bytes_version: int = -1
        self._params_bytes_lock = threading.Lock()

        # facades
        self.policy = _PolicyFacade(self)
        self.buffer = _BufferFacade(self)

    # ---- internal mutators (thread-safe) ---------------------------------

    def _add_transition_sync(self, transition: dict) -> None:
        t = dict(transition)
        t.setdefault("is_intervention", False)
        target = (
            self._demo_buffer if t["is_intervention"] else self._online_buffer
        )
        target.insert(t)
        with self._counter_lock:
            self._env_steps += 1

    def _publish_agent(self, new_agent) -> None:
        # `self._agent` is a plain Python ref; assignment is atomic under GIL.
        self._agent = new_agent
        with self._counter_lock:
            self._learner_step += 1
            self._params_version += 1

    def snapshot_agent(self):
        """Atomic snapshot of the current agent. Use for eval rollouts where
        you want a single frozen policy across multiple episodes, regardless
        of background learner updates."""
        return self._agent

    def record_episode(self, episode_return: float) -> None:
        """Record one episode return for `status()['mean_return_100']`."""
        with self._counter_lock:
            self._episode_returns.append(float(episode_return))

    def _record_stats(self, payload: dict) -> None:
        """If `wandb.init(...)` has been called, log here. Otherwise no-op."""
        if wandb.run is None:
            return
        wandb.log(payload, step=self._learner_step)

    # ---- demos -----------------------------------------------------------

    def preload_demos(self, transitions) -> int:
        for t in transitions:
            t = dict(t)
            t.setdefault("is_intervention", True)
            self._demo_buffer.insert(t)
        return len(self._demo_buffer)

    # ---- introspection ---------------------------------------------------

    def utd(self) -> float:
        """Effective UTD: total grad updates / env steps observed."""
        if self._env_steps == 0:
            return 0.0
        return self._learner_step * self._config.cta_ratio / self._env_steps

    def status(self) -> dict:
        return {
            "learner_step": self._learner_step,
            "env_steps": self._env_steps,
            "effective_utd": self.utd(),
            "configured_utd": self._config.cta_ratio,
            "online_buffer": len(self._online_buffer),
            "demo_buffer": len(self._demo_buffer),
            "params_version": self._params_version,
            "max_steps": self._config.max_steps,
            "mean_return_100": (
                float(np.mean(self._episode_returns))
                if self._episode_returns else None
            ),
            "uptime_sec": time.time() - self._started_at,
            "learner_running": self._learner_thread is not None
                               and self._learner_thread.is_alive(),
            "server_running": self._server_thread is not None
                              and self._server_thread.is_alive(),
        }

    # ---- training: synchronous ------------------------------------------

    def _sample_batch(self) -> dict:
        cfg = self._config
        kwargs = {"batch_size": cfg.batch_size}
        if cfg.image_keys:
            kwargs["pack_obs_and_next_obs"] = True
        n_online = len(self._online_buffer)
        n_demo = len(self._demo_buffer)
        if n_online and n_demo:
            half = cfg.batch_size // 2
            return concat_batches(
                self._online_buffer.sample(**{**kwargs, "batch_size": half}),
                self._demo_buffer.sample(**{**kwargs, "batch_size": half}),
                axis=0,
            )
        if n_online:
            return self._online_buffer.sample(**kwargs)
        if n_demo:
            return self._demo_buffer.sample(**kwargs)
        raise RuntimeError("cannot sample: both buffers empty")

    def train_step(self) -> dict:
        """Run one full learner step: `cta_ratio - 1` critic-only updates
        plus one (critic + actor + temperature) update.

        Iterates the agent's `CRITIC_NETWORKS` / `ALL_NETWORKS` generically,
        so plain SAC and residual SAC are both supported with no per-agent code.
        Returns the info dict from the final update."""
        agent = self._agent
        critic_networks = agent.CRITIC_NETWORKS
        all_networks = agent.ALL_NETWORKS
        is_bc_style = len(critic_networks) == 0
        if not is_bc_style:
            for _ in range(self._config.cta_ratio - 1):
                agent, _ = agent.update(
                    self._sample_batch(), networks_to_update=critic_networks
                )
        agent, info = agent.update(
            self._sample_batch(), networks_to_update=all_networks
        )
        self._publish_agent(agent)
        return _to_pylog(info)

    def _pretrain_step(self) -> dict:
        cfg = self._config
        kwargs = {"batch_size": cfg.batch_size}
        if cfg.image_keys:
            kwargs["pack_obs_and_next_obs"] = True
        batch = self._demo_buffer.sample(**kwargs)
        agent = self._agent
        agent, info = agent.update(batch, networks_to_update=agent.ALL_NETWORKS)
        self._publish_agent(agent)
        return _to_pylog(info)

    # ---- training: background thread -------------------------------------

    def start_learner(self) -> None:
        """Start a background thread that runs `train_step()` continuously.

        Returns immediately. If `config.pretrain_steps > 0`, the thread first
        runs that many updates against the demo buffer before transitioning
        to online updates."""
        if self._learner_thread is not None and self._learner_thread.is_alive():
            raise RuntimeError("learner already running")
        self._shutdown_event.clear()
        self._learner_thread = threading.Thread(
            target=self._learner_loop, daemon=True, name="wrl-learner",
        )
        self._learner_thread.start()

    def stop_learner(self, *, timeout: Optional[float] = 5.0) -> None:
        if self._learner_thread is None:
            return
        self._shutdown_event.set()
        self._learner_thread.join(timeout=timeout)
        self._learner_thread = None

    def _learner_loop(self) -> None:
        cfg = self._config
        if cfg.pretrain_steps > 0:
            if len(self._demo_buffer) == 0:
                raise RuntimeError(
                    "pretrain_steps > 0 but demo buffer is empty — preload demos"
                )
            print(f"[learner] pretraining for {cfg.pretrain_steps} steps")
            for _ in range(cfg.pretrain_steps):
                if self._shutdown_event.is_set():
                    return
                self._pretrain_step()

        # warmup: wait for online buffer to fill, OR demo buffer to be non-empty
        while (
            len(self._online_buffer) < cfg.training_starts
            and len(self._demo_buffer) == 0
        ):
            if self._shutdown_event.is_set():
                return
            time.sleep(0.25)
        print(
            f"[learner] warmup ok (online={len(self._online_buffer)}, "
            f"demo={len(self._demo_buffer)}) — running updates"
        )

        while not self._shutdown_event.is_set():
            if cfg.max_steps and self._learner_step >= cfg.max_steps:
                print(f"[learner] reached max_steps={cfg.max_steps}")
                return
            # learner-side UTD cap: don't lap the actor (over-training on a tiny
            # online buffer corrupts the critic). Wait for more env data.
            if cfg.max_utd and self.utd() >= cfg.max_utd:
                time.sleep(0.02)
                continue
            info = self.train_step()
            if self._learner_step % 200 == 0:
                self._record_stats({"train": info, "step": self._learner_step})

    # ---- pacing ----------------------------------------------------------

    def wait_for_utd(
        self, min_utd: float, *, timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> bool:
        """Block until `session.utd() >= min_utd` or `timeout` elapses.

        Returns True if the threshold was met, False on timeout. Lets the actor
        burn time at env reset rather than starve a GPU-bound learner."""
        t_start = time.monotonic()
        while self.utd() < min_utd:
            if self._shutdown_event.is_set():
                return False
            if timeout is not None and time.monotonic() - t_start > timeout:
                return False
            time.sleep(poll_interval)
        return True

    # ---- RNG -------------------------------------------------------------

    def _next_rng_key(self):
        with self._rng_lock:
            self._rng, key = jax.random.split(self._rng)
        return key

    # ---- HTTP server (opt-in) --------------------------------------------

    def start_server(self, *, host: str = "0.0.0.0", port: int = 5588) -> None:
        """Start a Quart-Trio HTTP server in a background thread, exposing the
        same buffer-add / params-fetch / status surface over HTTP.

        Idempotent: a second call raises if a server is already running."""
        if self._server_thread is not None and self._server_thread.is_alive():
            raise RuntimeError("server already running")

        def run_server():
            trio.run(self._serve_async, host, port)

        self._server_thread = threading.Thread(
            target=run_server, daemon=True, name=f"wrl-server-{port}",
        )
        self._server_thread.start()

    async def _serve_async(self, host: str, port: int) -> None:
        app = _make_quart_app(self)
        hcfg = hypercorn.config.Config()
        hcfg.bind = [f"{host}:{port}"]
        hcfg.workers = 1
        hcfg.loglevel = "warning"
        print(f"[server] listening on http://{host}:{port}")
        await hypercorn.trio.serve(app, hcfg)

    def latest_params_bytes(
        self, since_version: Optional[int] = None
    ) -> tuple:
        """Return (msgpack-encoded params bytes, version). If `since_version`
        equals the current version, returns (None, version)."""
        if since_version is not None and since_version == self._params_version:
            return None, self._params_version
        with self._params_bytes_lock:
            if (
                self._params_bytes is None
                or self._params_bytes_version != self._params_version
            ):
                v = self._params_version
                self._params_bytes = _serialize_params(self._agent.state.params)
                self._params_bytes_version = v
            return self._params_bytes, self._params_bytes_version


# ---------------------------------------------------------------------------
# facades
# ---------------------------------------------------------------------------


class _PolicyFacade:
    def __init__(self, session: Session):
        self._session = session

    def sample(self, obs, base_action=None, *, key=None, argmax: bool = False):
        """Sample an action from the current policy.

        For plain SAC, call `sample(obs)`. For the residual (EXPO-FT) path,
        pass the base action chunk: `sample(obs, base_action)` returns the
        composed full action `a_base + edit_scale * residual`."""
        if key is None:
            key = self._session._next_rng_key()
        agent = self._session._agent
        kwargs = {} if base_action is None else {
            "base_action": jax.device_put(base_action)
        }
        action = agent.sample_actions(
            observations=jax.device_put(obs), seed=key, argmax=argmax, **kwargs,
        )
        return np.asarray(jax.device_get(action))

    def sample_best_of_n(self, obs, n: int, *, key=None):
        """Critic-guided action selection: sample `n` policy actions and return
        the highest-Q one (DSRL latent steering). Needs an agent with a critic."""
        if key is None:
            key = self._session._next_rng_key()
        action = self._session._agent.sample_best_of_n(jax.device_put(obs), n, key)
        return np.asarray(jax.device_get(action))

    @property
    def params_version(self) -> int:
        return self._session._params_version


class _BufferFacade:
    def __init__(self, session: Session):
        self._session = session

    def add(
        self, observations, actions, next_observations, rewards,
        dones, is_intervention: bool = False, **extra,
    ) -> None:
        """Add a transition. Core fields are positional; residual-RL fields
        (`base_actions`, `next_base_actions`) and any other `extra_fields`
        configured on the buffer are passed as keyword args."""
        t = dict(
            observations=_to_np(observations),
            actions=_to_np(actions),
            next_observations=_to_np(next_observations),
            rewards=float(rewards),
            masks=1.0 - float(dones),
            dones=bool(dones),
            is_intervention=bool(is_intervention),
        )
        for k, v in extra.items():
            t[k] = _to_np(v) if isinstance(v, (np.ndarray, list)) else v
        self._session._add_transition_sync(t)

    def __len__(self) -> int:
        return len(self._session._online_buffer) + len(self._session._demo_buffer)

    @property
    def online_size(self) -> int:
        return len(self._session._online_buffer)

    @property
    def demo_size(self) -> int:
        return len(self._session._demo_buffer)


# ---------------------------------------------------------------------------
# Quart-Trio HTTP app
# ---------------------------------------------------------------------------


def _make_quart_app(session: Session) -> QuartTrio:
    app = QuartTrio(__name__)

    @app.post("/transitions")
    async def post_transitions():
        body = await request.get_data()
        transitions = msgpack.unpackb(body)
        if not isinstance(transitions, list):
            return {"error": "expected msgpack list of transition dicts"}, 400
        for t in transitions:
            session._add_transition_sync(t)
        return {"received": len(transitions)}

    @app.get("/params")
    async def get_params():
        since = request.headers.get("If-None-Match")
        since_v = int(since) if since and since.isdigit() else None
        body, version = session.latest_params_bytes(since_version=since_v)
        if body is None:
            return Response(status=304, headers={"X-Params-Version": str(version)})
        return Response(
            body, status=200, mimetype="application/octet-stream",
            headers={"X-Params-Version": str(version)},
        )

    @app.get("/status")
    async def get_status():
        return session.status()

    @app.get("/config")
    async def get_config():
        return session._config.to_dict()

    @app.post("/stats")
    async def post_stats():
        payload = await request.get_json()
        session._record_stats(payload or {})
        return {"ok": True}

    return app
