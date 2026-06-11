"""Render image observations from a robomimic raw/low-dim dataset.

Thin wrapper around robomimic's `dataset_states_to_obs` that first installs the
`mujoco_py` stub + EGL env vars (robomimic's env_robosuite hard-imports
mujoco_py, which we don't have), then forwards all CLI args to the robomimic
script. Example:

    uv run python envs/render_image_dataset.py \
        --dataset data/robomimic/tool_hang/ph/demo_v141.hdf5 \
        --output_name image_84.hdf5 \
        --camera_names agentview robot0_eye_in_hand \
        --camera_height 84 --camera_width 84 --done_mode 2
"""

import os
import runpy
import sys
import types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

if "mujoco_py" not in sys.modules:
    _stub = types.ModuleType("mujoco_py")
    _stub.builder = types.ModuleType("mujoco_py.builder")
    _stub.builder.MujocoException = Exception
    sys.modules["mujoco_py"] = _stub
    sys.modules["mujoco_py.builder"] = _stub.builder

if __name__ == "__main__":
    runpy.run_module(
        "robomimic.scripts.dataset_states_to_obs", run_name="__main__", alter_sys=True
    )
