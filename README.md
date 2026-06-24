# wrl

Clean fleet RL codebase (JAX/Flax). One central learner, one or many actor
processes joining over HTTP. Two training modes share the same `Session` core:

1. **RL from scratch** — chunked SAC from pixels (a residual policy on a *zero*
   base).
2. **EXPO-FT residual RL** — a residual SAC policy on top of a frozen base
   policy (a Diffusion Policy) served by a separate policy server. The actor
   queries `obs -> base action chunk`; the learned residual edits it
   (`a = a_base + edit_scale * residual`). A zero base reduces this to mode 1,
   so one agent + a swappable base server covers both.

Distilled from [alder](https://github.com/rail-berkeley/hil-serl)'s
learner/actor design (agentlace-free; Quart-Trio HTTP transport), with the
residual mechanism from [expo-ft](https://github.com/pd-perry/expo-ft). Target
tasks: robomimic tool-hang and transport from pixels.

```
wrl/
  session.py        Session: agent + buffers + learner thread + HTTP fleet server
  config.py         Config
  agents/           sac.py (plain SAC), residual_sac.py (chunked EXPO-FT residual)
  diffusion/        policy.py (DDPM/DDIM base DP), train_dp.py
  base_policy/      client.py, server.py, dp_server.py, mock_server.py
  networks/ common/ vision/ data/   (ported from alder)
envs/               robomimic_pixels.py, chunk_wrapper.py, dataset_loader.py,
                    render_image_dataset.py
examples/toolhang/  train.py  (task-generic: reads the env from dataset metadata)
tests/              unit + smoke tests
```

## Install

```bash
uv sync --extra cuda --extra dev   # GPU jax (CUDA 12) + pytest
uv sync                            # CPU-only (smoke tests)
```

Pixel rendering uses headless EGL (`MUJOCO_GL=egl`, set automatically by the env
wrapper). Verified on an RTX 5090 with `jax-cuda12 0.6.2`.

## Fleet / actor-learner

`Session` owns the agent, the replay buffers, and a background learner thread.
`session.start_server(port=5588)` exposes `/transitions`, `/params`, `/status`,
`/config`, `/stats` (msgpack params with `If-None-Match`/`X-Params-Version`
versioning) so remote actor processes can join. The user writes their own env
loop (`session.policy.sample`, `session.buffer.add`, `session.wait_for_utd`).

## End-to-end: tool-hang (EXPO-FT)

robomimic image datasets aren't hosted — render them from the raw demo states.

```bash
# 1. download raw demo states
uv run python -m robomimic.scripts.download_datasets \
    --download_dir data/robomimic --tasks tool_hang --dataset_types ph --hdf5_types raw

# 2. render image observations (replays states through robosuite w/ EGL)
uv run python envs/render_image_dataset.py \
    --dataset data/robomimic/tool_hang/ph/demo_v141.hdf5 --output_name image_84.hdf5 \
    --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --done_mode 2

# 3. train the base Diffusion Policy
uv run python -m wrl.diffusion.train_dp \
    --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
    --horizon 8 --train-steps 60000 --out-path checkpoints/dp_toolhang.pkl

# 4. serve the DP as the base policy
uv run python -m wrl.base_policy.dp_server --checkpoint checkpoints/dp_toolhang.pkl --port 8200 &

# 5. EXPO-FT residual RL on top of it
uv run python examples/toolhang/train.py \
    --dataset-path data/robomimic/tool_hang/ph/image_84.hdf5 \
    --base dp --dp-port 8200 --horizon 8 --warmstart-demos
```

When the DP server and the RL learner share one GPU, cap jax memory per process
(e.g. `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`).

**RL from scratch** (baseline, no base server): `--base zeros`.

**Transport** (dual-arm, 14-dof): the same scripts, pointed at a transport
dataset — `--tasks transport` in step 1, transport cameras
(`agentview robot0_eye_in_hand robot1_eye_in_hand`) in step 2. The env wrapper
auto-detects the dual-arm layout from the dataset metadata.

## Tests

```bash
uv run python -m pytest tests/ -q
```

`test_session_e2e_dummy.py` exercises the whole residual fleet pipeline (dummy
pixel env + mock base server + learner + HTTP) without mujoco; the robomimic
test skips cleanly if GL is unavailable.

## Docs

- [docs/sweep_findings.md](docs/sweep_findings.md) — Q-OIL transport
  hyperparameter sweep (v3–v5): `bc_weight` vs `intervention_bonus`, conclusions,
  chosen operating point.
- [docs/yam_port.md](docs/yam_port.md) — plan for porting the residual Q-OIL
  agent to the i2rt YAM arms via tri-ml/raiden.
