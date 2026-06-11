"""Per-run training config for a wrl Session."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    """Per-run training config."""

    batch_size: int = 256
    """SAC batch size (split 50/50 across online + demo buffers when both exist)."""
    cta_ratio: int = 1
    """Critic-to-actor ratio: `cta_ratio - 1` critic-only updates plus one full update per learner step."""
    training_starts: int = 1_000
    """Online buffer fill required before updates begin."""
    replay_buffer_capacity: int = 200_000
    """Online replay buffer capacity."""
    demo_buffer_capacity: int = 200_000
    """Demo / offline buffer capacity (RLPD warm-start)."""
    pretrain_steps: int = 0
    """Optional pretrain steps on the demo buffer before online updates."""
    max_steps: int = 0
    """Stop the learner after this many gradient updates. 0 means no cap."""
    image_keys: tuple[str, ...] = ()
    """Image observation keys (selects the MemoryEfficient pixel buffer when non-empty)."""
    extra_fields: tuple = ()
    """Extra per-transition fields beyond the core set. Each entry is either a
    bare name (scalar float32) or a `(name, dtype, shape)` tuple. wrl uses this
    to carry the residual-RL `base_actions` / `next_base_actions` chunks."""

    def to_dict(self) -> dict:
        return asdict(self)
