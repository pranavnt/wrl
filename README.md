# wrl

Clean fleet RL codebase (JAX/Flax). One central learner, one or many actor
processes joining over HTTP. Two training modes share the same `Session` core:

1. **RL from scratch** — standard SAC from pixels.
2. **EXPO-FT residual RL** — a residual SAC policy on top of a frozen base
   policy (a Diffusion Policy) served by a separate policy server. The actor
   queries `obs -> base action chunk`; the learned residual edits it
   (`a = a_base + edit_scale * residual`).

Distilled from [alder](https://github.com/rail-berkeley/hil-serl)'s
learner/actor design (agentlace-free), with the residual mechanism from
[expo-ft](https://github.com/pd-perry/expo-ft). Target tasks: robomimic
tool-hang and transport from pixels.

```
wrl/            # the library (Session, agents, networks, data, diffusion, base_policy)
envs/           # robomimic pixel env wrappers
examples/       # training scripts (tool-hang scratch / expo-ft, transport)
tests/          # smoke + unit tests
```

## Install

```bash
uv sync                # CPU jax
uv sync --extra cuda   # GPU jax (CUDA 12)
```

## Status

Milestone 1 (intervention-free RL): in progress. See
`.claude/plans/` for the build plan.
