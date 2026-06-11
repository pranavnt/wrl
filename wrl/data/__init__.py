"""Replay buffers + data stores, plus residual-RL field helpers."""

import numpy as np

from wrl.data.data_store import (
    MemoryEfficientReplayBufferDataStore,
    ReplayBufferDataStore,
)
from wrl.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
from wrl.data.replay_buffer import ReplayBuffer

# Names of the per-transition fields the residual (EXPO-FT) path adds on top
# of the core set. Both hold a flattened base-policy action chunk of dim A*H.
RESIDUAL_FIELDS = ("base_actions", "next_base_actions")


def residual_extra_fields(chunk_dim: int, dtype=np.float32) -> tuple:
    """`Config.extra_fields` entries that carry the residual-RL base chunks.

    `chunk_dim` is the flattened chunk size A*H (action_dim * horizon)."""
    return tuple((name, dtype, (chunk_dim,)) for name in RESIDUAL_FIELDS)


__all__ = [
    "ReplayBuffer",
    "MemoryEfficientReplayBuffer",
    "ReplayBufferDataStore",
    "MemoryEfficientReplayBufferDataStore",
    "RESIDUAL_FIELDS",
    "residual_extra_fields",
]
