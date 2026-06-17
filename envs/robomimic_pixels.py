"""robomimic / robosuite pixel env wrapper (tool-hang, transport, ...).

Built from a robomimic dataset's env metadata so the controller (and thus the
action space) matches the demos the base Diffusion Policy is trained on. Adds
offscreen camera rendering on top. Single-frame image obs + proprio state,
sparse reward, gym API.

Obs dict (each value has a leading frame dim of 1 for the memory-efficient
pixel buffer):
    <cam>_image : (1, H, W, 3) uint8     for each camera
    state       : (1, proprio)  float32

Action: robosuite's native action vector (Box(-1, 1)). Wrap with
`envs.chunk_wrapper.ActionChunkWrapper` for chunked (EXPO-FT) control.
"""

import json
import os
import sys
import types

# Headless GPU rendering via EGL. Must be set before robosuite is imported.
# (The EGLError noise at interpreter exit is a harmless robosuite teardown bug.)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

# robomimic 0.3.0 hard-imports the deprecated mujoco_py; stub it.
if "mujoco_py" not in sys.modules:
    _stub = types.ModuleType("mujoco_py")
    _stub.builder = types.ModuleType("mujoco_py.builder")
    _stub.builder.MujocoException = Exception
    sys.modules["mujoco_py"] = _stub
    sys.modules["mujoco_py.builder"] = _stub.builder

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import robosuite  # noqa: E402

# camera + proprio layout per robosuite env name
_ENV_SPECS = {
    "Lift": dict(
        cameras=("agentview", "robot0_eye_in_hand"),
        proprio_prefixes=("robot0",),
    ),
    "PickPlaceCan": dict(
        cameras=("agentview", "robot0_eye_in_hand"),
        proprio_prefixes=("robot0",),
    ),
    "ToolHang": dict(
        cameras=("agentview", "robot0_eye_in_hand"),
        proprio_prefixes=("robot0",),
    ),
    "TwoArmTransport": dict(
        cameras=("agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"),
        proprio_prefixes=("robot0", "robot1"),
    ),
}

_PROPRIO_SUFFIXES = ("eef_pos", "eef_quat", "gripper_qpos")
_RENDER_KEYS = (
    "has_renderer", "has_offscreen_renderer", "use_camera_obs", "use_object_obs",
    "camera_names", "camera_heights", "camera_widths", "camera_depths",
    "reward_shaping", "render_camera", "horizon", "ignore_done", "hard_reset",
    "render_gpu_device_id",   # render device is set via MUJOCO_EGL_DEVICE_ID
)


def _coerce_env_kwargs_for_robosuite_1_4(env_kwargs: dict) -> None:
    """In-place: drop/translate fields newer datasets carry that pinned
    robosuite==1.4.1 doesn't accept (lite_physics; 1.5 composite controller)."""
    env_kwargs.pop("lite_physics", None)
    cc = env_kwargs.get("controller_configs")
    if isinstance(cc, dict) and "body_parts" in cc:
        inner = next(iter(cc["body_parts"].values()))
        for k in ("gripper", "input_ref_frame"):
            inner.pop(k, None)
        env_kwargs["controller_configs"] = inner


def read_env_meta(dataset_path: str) -> dict:
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


class RoboMimicPixelEnv(gym.Env):
    def __init__(
        self,
        dataset_path: str,
        *,
        image_size: int = 84,
        max_episode_steps: int = 700,
        reward_shaping: bool = False,
        cameras=None,
        include_lowdim: bool = False,
    ):
        # include_lowdim: also expose a flat full low-dim state (proprio +
        # object-state) under obs["lowdim"], for a STATE-conditioned steering
        # policy (DSRL) riding a pixel-conditioned base DP.
        self.include_lowdim = include_lowdim
        env_meta = read_env_meta(dataset_path)
        env_name = env_meta["env_name"]
        if env_name not in _ENV_SPECS:
            raise ValueError(f"unsupported env {env_name!r}; known: {list(_ENV_SPECS)}")
        spec = _ENV_SPECS[env_name]
        self.env_name = env_name
        self.cameras = tuple(cameras or spec["cameras"])
        self.image_size = image_size
        self._max_episode_steps = max_episode_steps

        kwargs = dict(env_meta["env_kwargs"])
        _coerce_env_kwargs_for_robosuite_1_4(kwargs)
        for k in _RENDER_KEYS:
            kwargs.pop(k, None)

        self.env = robosuite.make(
            env_name=env_name,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            camera_names=list(self.cameras),
            camera_heights=image_size,
            camera_widths=image_size,
            ignore_done=True,
            hard_reset=False,
            **kwargs,
        )

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
        self._object_key = "object-state" if "object-state" in first else "object"
        if include_lowdim:
            # lowdim (proprio + object-state) is returned in the obs dict for a
            # privileged V*/expert, but is DELIBERATELY kept OUT of
            # observation_space so it's never stored in the (pixel-policy) buffer
            # or seen by the learner's encoder -> the learner stays pixels+proprio.
            self.lowdim_dim = int(self._lowdim(first).shape[0])
        self.observation_space = gym.spaces.Dict(obs_spaces)

        adim = self.env.action_dim
        self.action_space = gym.spaces.Box(-1.0, 1.0, (adim,), np.float32)
        self.image_keys = tuple(f"{cam}_image" for cam in self.cameras)
        self.proprio_keys = tuple(self._proprio_keys)
        self._elapsed = 0

    def _lowdim(self, raw: dict) -> np.ndarray:
        """Flat full low-dim state: proprio (eef pos/quat/gripper) + object-state."""
        parts = [np.asarray(raw[k], np.float32).reshape(-1) for k in self._proprio_keys]
        parts.append(np.asarray(raw[self._object_key], np.float32).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def _obs(self, raw: dict) -> dict:
        out = {}
        for cam in self.cameras:
            # robosuite returns camera images bottom-up; robomimic's
            # dataset_states_to_obs stores them vertically flipped (top-down).
            # Flip here so live frames match the rendered demo hdf5 the base DP
            # is trained on (verified: live[::-1] vs hdf5 MSE ~2.6 vs ~8465).
            out[f"{cam}_image"] = raw[f"{cam}_image"][::-1][None].astype(np.uint8)
        state = np.concatenate(
            [np.asarray(raw[k], np.float32).reshape(-1) for k in self._proprio_keys]
        )
        out["state"] = state[None].astype(np.float32)
        if self.include_lowdim:
            out["lowdim"] = self._lowdim(raw)
        return out

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # robosuite samples object placement from the global numpy RNG;
            # seeding it makes resets reproducible (for paired evaluation).
            np.random.seed(seed)
        self._elapsed = 0
        return self._obs(self.env.reset()), {}

    def step(self, action):
        action = np.clip(
            np.asarray(action, np.float32), self.action_space.low, self.action_space.high
        )
        raw, reward, _, _ = self.env.step(action)
        self._elapsed += 1
        success = bool(self.env._check_success())
        truncated = self._elapsed >= self._max_episode_steps
        info = {"success": float(success)}
        return self._obs(raw), float(reward), success, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def make_robomimic_pixel_env(dataset_path: str, **kwargs) -> RoboMimicPixelEnv:
    return RoboMimicPixelEnv(dataset_path, **kwargs)
