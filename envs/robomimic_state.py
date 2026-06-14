"""Low-dim (state) robomimic env wrapper for RLPD-from-scratch expert training.

Concatenated robot + object state observations (no pixels, no rendering), built
from a robomimic low_dim dataset's env metadata. Auto-detects single-arm
(robot0_*) and dual-arm (robot0_* + robot1_*) tasks. `demo_transitions()` yields
the offline demos for the RLPD demo buffer. Ported from alder's
test/robomimic_can/env.py, generalized to any robomimic low_dim task.
"""

import sys
import types

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
from robomimic.utils.env_utils import create_env_from_metadata  # noqa: E402
from robomimic.utils.file_utils import get_env_metadata_from_dataset  # noqa: E402

_ROBOT_BASE_KEYS = (
    "eef_pos", "eef_quat", "eef_vel_ang", "eef_vel_lin",
    "gripper_qpos", "gripper_qvel",
    "joint_pos", "joint_pos_cos", "joint_pos_sin", "joint_vel",
)
# robomimic's canonical "low_dim" obs set (what plantok/robomimic BC uses to hit
# ~85% on tool-hang). Velocities + joint angles are noisy and hurt BC, so the lean
# set is the better state for a state diffusion policy.
_LEAN_ROBOT_KEYS = ("eef_pos", "eef_quat", "gripper_qpos")


def _coerce_env_kwargs_for_robosuite_1_4(env_kwargs: dict) -> None:
    env_kwargs.pop("lite_physics", None)
    cc = env_kwargs.get("controller_configs")
    if isinstance(cc, dict) and "body_parts" in cc:
        inner = next(iter(cc["body_parts"].values()))
        for k in ("gripper", "input_ref_frame"):
            inner.pop(k, None)
        env_kwargs["controller_configs"] = inner


def _robot_obs_keys(obs_keys, lean=False):
    base = _LEAN_ROBOT_KEYS if lean else _ROBOT_BASE_KEYS
    out = []
    for idx in range(4):
        prefix = f"robot{idx}_"
        if not any(k.startswith(prefix) for k in obs_keys):
            continue
        for bk in base:
            key = f"{prefix}{bk}"
            if key in obs_keys:
                out.append(key)
    if not out:
        raise ValueError(f"no robotN_* keys found in {sorted(obs_keys)}")
    return out


class RoboMimicStateEnv(gym.Env):
    def __init__(self, dataset_path: str, *, max_episode_steps: int = 700,
                 reset_to_demo_prob: float = 0.0, lean_obs: bool = False):
        self.dataset_path = dataset_path
        self.lean_obs = lean_obs
        self._max_episode_steps = max_episode_steps
        # fraction of TRAINING resets that start from a demo's initial sim state
        # (narrows the reset distribution so success is reachable on hard sparse
        # tasks). Eval resets ignore this (pass options={"normal": True}).
        self.reset_to_demo_prob = reset_to_demo_prob

        env_meta = get_env_metadata_from_dataset(dataset_path)
        _coerce_env_kwargs_for_robosuite_1_4(env_meta.get("env_kwargs", {}))
        self.env = create_env_from_metadata(env_meta, render=False, render_offscreen=False)

        with h5py.File(dataset_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
            obs_keys = list(f[f"data/{demos[0]}/obs"].keys())
            # ALL states along the demo trajectories (incl. near-goal states), so
            # training resets put the agent close to success -> reward propagates
            # back. Subsample for memory.
            states = np.concatenate([np.asarray(f["data"][d]["states"][:]) for d in demos], 0)
        if len(states) > 60000:
            idx = np.random.RandomState(0).choice(len(states), 60000, replace=False)
            states = states[idx]
        self._demo_init_states = states
        self.robot_obs_keys = _robot_obs_keys(obs_keys, lean=lean_obs)
        self._object_key = "object" if "object" in obs_keys else "object-state"

        # robomimic actions are normalized to [-1, 1]
        action_dim = self.env.action_dimension
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)

        obs_dim = self._get_obs().shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._elapsed = 0

    def _get_obs(self):
        # live robosuite obs uses "object-state"; the hdf5 demos use "object".
        di = self.env.env._get_observations(force_update=True)
        robot = np.concatenate([np.asarray(di[k]).reshape(-1) for k in self.robot_obs_keys])
        obj = np.asarray(di["object-state"]).reshape(-1)
        return np.concatenate([robot, obj]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.env.env.reset()
        normal = bool((options or {}).get("normal", False))
        if (not normal) and self.reset_to_demo_prob > 0 and \
                np.random.rand() < self.reset_to_demo_prob:
            st = self._demo_init_states[np.random.randint(len(self._demo_init_states))]
            self.env.env.sim.set_state_from_flattened(st)
            self.env.env.sim.forward()
        self._elapsed = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), self.action_space.low, self.action_space.high)
        _, reward, _, _ = self.env.env.step(action)
        self._elapsed += 1
        success = bool(self.env.is_success()["task"])
        truncated = self._elapsed >= self._max_episode_steps
        info = {"success": float(success)}
        return self._get_obs(), float(reward), success, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def demo_transitions(self):
        """Per-step offline demo transitions for the RLPD demo buffer."""
        out = []
        with h5py.File(self.dataset_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
            for demo in demos:
                d = f["data"][demo]
                robot = np.concatenate(
                    [np.asarray(d["obs"][k][:]).reshape(len(d["actions"]), -1)
                     for k in self.robot_obs_keys], axis=1)
                nrobot = np.concatenate(
                    [np.asarray(d["next_obs"][k][:]).reshape(len(d["actions"]), -1)
                     for k in self.robot_obs_keys], axis=1)
                obj = np.asarray(d["obs"][self._object_key][:])
                nobj = np.asarray(d["next_obs"][self._object_key][:])
                obs = np.concatenate([robot, obj], axis=1).astype(np.float32)
                nobs = np.concatenate([nrobot, nobj], axis=1).astype(np.float32)
                acts = np.asarray(d["actions"][:], np.float32).clip(-0.99, 0.99)
                rews = np.asarray(d["rewards"][:], np.float32)
                dones = np.asarray(d["dones"][:], np.float32)
                for t in range(len(acts)):
                    out.append(dict(
                        observations=obs[t], actions=acts[t], next_observations=nobs[t],
                        rewards=float(rews[t]), masks=1.0 - float(dones[t]),
                        dones=bool(dones[t]), is_intervention=True,
                    ))
        return out
