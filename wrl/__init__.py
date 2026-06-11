"""wrl — clean fleet RL codebase (RL-from-scratch + EXPO-FT residual RL)."""

import os as _os

# Disable XLA command buffers (CUDA graphs). XLA's autotuner uses CUDA-graph
# stream capture which is flaky on some new GPUs (observed on RTX 5090 /
# Blackwell + jax 0.6.2: "Failed to end stream capture /
# CUDA_ERROR_STREAM_CAPTURE_INVALIDATED" on the first pixel update). Must be set
# before XLA initializes. Appended, not overwritten, so user flags win.
if "xla_gpu_enable_command_buffer" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (
        _os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
    ).strip()

from wrl.agents.residual_sac import ResidualSACAgent
from wrl.agents.sac import SACAgent
from wrl.config import Config
from wrl.session import Session

__version__ = "0.1.0"

__all__ = ["Session", "Config", "SACAgent", "ResidualSACAgent", "__version__"]
