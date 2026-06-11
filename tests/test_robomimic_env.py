"""Smoke test for the robomimic pixel env wrapper (tool-hang).

Requires working mujoco/EGL offscreen rendering; skips cleanly otherwise so the
rest of the suite still runs on machines without a GPU/GL.
"""

import numpy as np
import pytest


def _make(task):
    try:
        from envs.robomimic_pixels import make_robomimic_pixel_env

        return make_robomimic_pixel_env(task, image_size=84, max_episode_steps=50)
    except Exception as e:  # missing GL / robosuite / dataset
        pytest.skip(f"robomimic pixel env unavailable: {type(e).__name__}: {e}")


def test_toolhang_pixel_env_shapes():
    from envs.chunk_wrapper import ActionChunkWrapper

    env = _make("tool-hang")
    try:
        assert env.image_keys == ("agentview_image", "robot0_eye_in_hand_image")
        A = env.action_space.shape[0]
        H = 4
        cenv = ActionChunkWrapper(env, A, H, discount=0.97)
        assert cenv.action_space.shape == (A * H,)

        obs, _ = cenv.reset()
        for key in env.image_keys:
            assert obs[key].shape == (1, 84, 84, 3)
            assert obs[key].dtype == np.uint8
        assert obs["state"].ndim == 2 and obs["state"].shape[0] == 1

        obs, r, done, trunc, info = cenv.step(cenv.action_space.sample())
        assert np.isfinite(r)
        assert "success" in info
        assert obs[env.image_keys[0]].shape == (1, 84, 84, 3)
    finally:
        env.close()
