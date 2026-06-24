# Porting Q-OIL to the i2rt YAM arms (tri-ml/raiden)

Plan for running the residual Q-OIL agent on the real **i2rt YAM** arms via
**tri-ml/raiden**, instead of robomimic transport in sim. Target config from the
[sweep](sweep_findings.md): **`edit_scale=0.15`, `bc_weight=0.25`,
`intervention_bonus=0.1`**.

Status: **planning** — not yet implemented. raiden is not on this machine; the
env wrapper must be written against its real control/teleop/camera API.

## What stays the same

The agent core is hardware-agnostic and does **not** change:

- `wrl/agents/qoil.py` — decoupled critics (`Q_TD` task, `Q_Opt` optimism), the
  residual actor, BC regularization.
- `wrl/agents/residual_sac.py` — residual composition `a = a_base + edit_scale *
  residual`, chunk-MDP SAC.
- `wrl/session.py` — learner thread, replay buffer, HTTP fleet server.
- The `is_intervention` buffer field and the `hil` baseline already exist for
  human-in-the-loop training; Q-OIL is built for exactly this shape.

## What changes (the robot-coupled surface)

All robot coupling lives in the rollout loop of `examples/transport/train_qoil.py`.
A real-robot entrypoint (`examples/yam/train_qoil_yam.py`) swaps each piece:

| Piece (sim)                              | Role                                          | Real-YAM replacement                                            |
| ---------------------------------------- | --------------------------------------------- | -------------------------------------------------------------- |
| `make_robomimic_pixel_env` → `cenv`      | gym Env: Dict obs (images+lowdim), Box action | **`RaidenYAMEnv`** behind the same gym API (over raiden)        |
| `base_chunk()` (`FlowPolicy`)            | frozen base policy emits the action chunk     | **base policy trained on the YAM task** (BC/diffusion on demos) |
| `expert_act` (DPPO expert)               | supplies intervention/takeover actions        | **human teleop** (YAM leader/follower)                          |
| `expert_val_batch` + `stagnating()` (V*) | auto-detects stagnation → triggers takeover   | **human-triggered** intervention signal (no learned V* on HW)   |
| `info["success"]` (`_check_success`)     | sparse 0/1 reward                             | **success detector or human label** at episode end             |
| `cenv.reset()`                           | sim reset                                     | scripted / human reset-to-home                                 |

## Target env interface (`RaidenYAMEnv`)

Match what the agent already expects (see `envs/robomimic_pixels.py`):

- `observation_space`: `gym.spaces.Dict` with image keys (camera streams) +
  `"state"` / `"lowdim"` proprio.
- `action_space`: `gym.spaces.Box(-1, 1, (action_dim,))`, normalized; raiden
  controller un-normalizes to joint/EE targets. **Clip to action bounds on send**
  (the agent stores the unclipped `a_full`; sim relies on the env clipping —
  preserve that).
- `step(action)` → `(obs, reward, success, truncated, info)`; chunked via the
  existing `ActionChunkWrapper` (`envs/chunk_wrapper.py`) so `Ta`-step chunks and
  the `discount**Ta` bootstrap carry over unchanged.
- `reset()` → home pose + obs.

## Intervention model (the central design change)

In sim, interventions are a DPPO expert auto-gated by a learned value function
`V*` (PAM stagnation). On hardware **neither exists**: replace both with **human
teleop**. The operator watches the rollout and takes over via the YAM
leader arm; while engaged, the executed action is the teleop action and the
transition is stored with `is_intervention=True`. This is the standard HIL-SERL
loop and is what `bc_weight` + `intervention_bonus` were tuned for.

This removes `expert_act`, `expert_val_batch`, `vhist`, `stagnating()`,
`pam_k`, and `pam_delta` from the YAM entrypoint, replacing them with a
teleop read + an intervention-active flag.

## Open questions (gate implementation)

1. **raiden API access** — where is tri-ml/raiden readable (path on the robot
   box / public repo / pasted API)? The env wrapper is written against its
   control, teleop-read, and camera-stream interfaces.
2. **Base policy** — is there a YAM-task base policy to ride on, or does one need
   to be trained from demos first? (Zero-base is possible but loses the
   warm-start; Q-OIL wants a competent base.)
3. **Success signal** — human button/keypress at episode end, or an automated
   detector?
4. **Compute / run location** — training runs on the robot host (single
   real-time actor, not the klone array). The async learner can still run in-
   process or on a co-located GPU.

## Sketch entrypoint (once the above are answered)

```
examples/yam/train_qoil_yam.py
  env   = ActionChunkWrapper(RaidenYAMEnv(...), d_a, Ta, discount)
  base  = load YAM base policy  ->  base_chunk()
  agent = QOIL.create_pixels(..., edit_scale=0.15, bc_weight=0.25,
                             intervention_bonus=0.1)
  loop:
    a = teleop.action() if teleop.engaged else agent.sample(obs, a_base)
    obs', r, done, *_ = env.step(a)
    buffer.add(obs, a, obs', r, done, is_intervention=teleop.engaged,
               base_actions=a_base, next_base_actions=base_chunk())
```
