"""robomimic / robosuite pixel env wrapper (tool-hang, transport, ...).

Single-frame image obs + proprio state, sparse reward, gym API. Built directly
on robosuite.make so we control the camera config; the matching base Diffusion
Policy and the residual RL agent both consume this wrapper, so they stay
consistent with each other.

Obs dict (each value has a leading frame dim of 1 for the memory-efficient
pixel buffer):
    <cam>_image : (1, H, W, 3) uint8     for each camera
    state       : (1, proprio)  float32

Action: robosuite's native action vector, Box(-1, 1). Wrap with
`envs.chunk_wrapper.ActionChunkWrapper` for chunked (EXPO-FT) control.
"""

import os
import sys
import types

# Headless GPU rendering: mujoco + robosuite use EGL via PyOpenGL. Must be set
# before robosuite is imported. (The EGLError noise at interpreter exit is a
# harmless robosuite/PyOpenGL teardown issue.)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# robomimic 0.3.0 hard-imports the deprecated mujoco_py; stub it (we never call
# the code path that uses it).
if "mujoco_py" not in sys.modules:
    _stub = types.ModuleType("mujoco_py")
    _stub.builder = types.ModuleType("mujoco_py.builder")
    _stub.builder.MujocoException = Exception
    sys.modules["mujoco_py"] = _stub
    sys.modules["mujoco_py.builder"] = _stub.builder

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import robosuite  # noqa: E402


# task -> robosuite construction. `proprio_prefixes` selects which robotN_* obs
# go into the proprio `state` vector.
TASKS = {
    "tool-hang": dict(
        robosuite_name="ToolHang",
        robots="Panda",
        cameras=("agentview", "robot0_eye_in_hand"),
        proprio_prefixes=("robot0",),
        extra_kwargs={},
    ),
    "transport": dict(
        robosuite_name="TwoArmTransport",
        robots=["Panda", "Panda"],
        cameras=("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"),
        proprio_prefixes=("robot0", "robot1"),
        extra_kwargs=dict(env_configuration="single-arm-opposed"),
    ),
}

_PROPRIO_SUFFIXES = ("eef_pos", "eef_quat", "gripper_qpos")


class RoboMimicPixelEnv(gym.Env):
    def __init__(
        self,
        task: str,
        *,
        image_size: int = 84,
        control_freq: int = 20,
        max_episode_steps: int = 400,
        reward_shaping: bool = False,
    ):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
        spec = TASKS[task]
        self.task = task
        self.cameras = tuple(spec["cameras"])
        self.image_size = image_size
        self._max_episode_steps = max_episode_steps

        self.env = robosuite.make(
            env_name=spec["robosuite_name"],
            robots=spec["robots"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            camera_names=list(self.cameras),
            camera_heights=image_size,
            camera_widths=image_size,
            control_freq=control_freq,
            horizon=max_episode_steps,
            ignore_done=True,
            hard_reset=False,
            **spec["extra_kwargs"],
        )

        # proprio keys present in robosuite obs
        first = self.env.reset()
        self._proprio_keys = [
            f"{p}_{s}"
            for p in spec["proprio_prefixes"]
            for s in _PROPRIO_SUFFIXES
            if f"{p}_{s}" in first
        ]
        proprio_dim = int(sum(np.asarray(first[k]).size for k in self._proprio_keys))

        obs_spaces = {
            f"{cam}_image": gym.spaces.Box(0, 255, (1, image_size, image_size, 3), np.uint8)
            for cam in self.cameras
        }
        obs_spaces["state"] = gym.spaces.Box(-np.inf, np.inf, (1, proprio_dim), np.float32)
        self.observation_space = gym.spaces.Dict(obs_spaces)

        adim = self.env.action_dim
        self.action_space = gym.spaces.Box(-1.0, 1.0, (adim,), np.float32)
        self.image_keys = tuple(f"{cam}_image" for cam in self.cameras)
        self._elapsed = 0

    def _obs(self, raw: dict) -> dict:
        out = {}
        for cam in self.cameras:
            # robosuite renders bottom-up; flip vertically to a natural frame.
            img = raw[f"{cam}_image"][::-1]
            out[f"{cam}_image"] = img[None].astype(np.uint8)  # (1, H, W, 3)
        state = np.concatenate(
            [np.asarray(raw[k], np.float32).reshape(-1) for k in self._proprio_keys]
        )
        out["state"] = state[None].astype(np.float32)  # (1, proprio)
        return out

    def reset(self, *, seed=None, options=None):
        self._elapsed = 0
        raw = self.env.reset()
        return self._obs(raw), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, np.float32), self.action_space.low, self.action_space.high
        )
        raw, reward, _, _ = self.env.step(action)
        self._elapsed += 1
        success = bool(self.env._check_success())
        truncated = self._elapsed >= self._max_episode_steps
        done = success
        info = {"success": float(success)}
        return self._obs(raw), float(reward), done, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_robomimic_pixel_env(task: str, **kwargs) -> RoboMimicPixelEnv:
    return RoboMimicPixelEnv(task, **kwargs)
