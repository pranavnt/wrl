from threading import Lock
from typing import Iterable

import gymnasium as gym

from wrl.data.replay_buffer import ReplayBuffer
from wrl.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer


class ReplayBufferDataStore(ReplayBuffer):
    """Thread-safe ReplayBuffer.

    The wrl server runs JAX updates in a worker thread (via
    `trio.to_thread.run_sync`) and ingests transitions from HTTP handler
    threads, so insert/sample race. A plain `threading.Lock` is
    sufficient — `trio.Lock` only protects async code.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        extra_fields: Iterable = (),
    ):
        super().__init__(
            observation_space,
            action_space,
            capacity,
            extra_fields=extra_fields,
        )
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super().insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys: Iterable[str] = ("image",),
        extra_fields: Iterable = (),
    ):
        super().__init__(
            observation_space,
            action_space,
            capacity,
            pixel_keys=tuple(image_keys),
            extra_fields=extra_fields,
        )
        self._lock = Lock()

    def insert(self, *args, **kwargs):
        with self._lock:
            super().insert(*args, **kwargs)

    def sample(self, *args, **kwargs):
        with self._lock:
            return super().sample(*args, **kwargs)


def populate_data_store(data_store, demos_path: Iterable[str]):
    """Load pickled demo transitions into a data store."""
    import pickle as pkl

    for demo_path in demos_path:
        with open(demo_path, "rb") as f:
            demo = pkl.load(f)
            for transition in demo:
                data_store.insert(transition)
        print(f"Loaded {len(data_store)} transitions.")
    return data_store
