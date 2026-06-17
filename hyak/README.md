# Q-OIL transport sweep on HYAK (checkpoint partition)

Sweeps **Q-OIL** on transport-from-pixels over `edit_scale × bc_weight ×
intervention_bonus × seed` (81 configs) as a Slurm array on the preemptible
`ckpt-g2` partition (L40/L40S GPUs). Each job is a self-contained, in-process
run (no fleet server). ~6h/job at 80k learner steps.

> Jobs are **not** resumable — if a job is preempted it's lost; just resubmit
> that array index. Fine while `ckpt` contention is low.

## 0. One-time setup on klone (`/gscratch/weirdlab/pranavnt/wrl`)

The DPPO expert runs from wrl's **vendored** model copy (`wrl/base_policy/_dppo_min`)
— **no DPPO repo on klone**, only the 3 expert data files. The jobs expect
(override via env vars in `sweep.sbatch`):

| What | Default path |
|---|---|
| this repo | `/gscratch/weirdlab/pranavnt/wrl` |
| frozen pixel DP base | `$WRL/checkpoints/flowdp_transport_pixel_avg_step45000.pkl` |
| expert weights / config / norm | `$WRL/checkpoints/expert/{state_200.pt, ft_ppo_diffusion_mlp.yaml, normalization.npz}` |
| transport env_meta dataset | `$WRL/data/robomimic/transport/ph/low_dim_v141.hdf5` |

Rebuild the venv for the cluster GPU (L40S = sm_89; don't copy the 5090 `.venv`):

```bash
cd /gscratch/weirdlab/pranavnt/wrl
module load cmake gcc          # cmake builds egl_probe; run on a COMPUTE node
                               # (the module cmake SIGILLs on the login node's CPU)
uv sync --extra cuda           # builds .venv with GPU jaxlib (jax[cuda12])
```

The sbatch runs `$WRL/.venv/bin/python` directly, so this `.venv` (on /gscratch)
must include the `cuda` extra — a plain `uv sync` installs CPU jaxlib and jobs
will silently run on CPU. Verify: `.venv/bin/python -c "import jax; print(jax.devices())"`
should show a `CudaDevice`.

Sanity-check one config on an interactive node before the array:

```bash
salloc -A weirdlab -p ckpt-g2 --gpus=1 --mem=48G -c 8 --time=1:00:00
# then inside the alloc:
MAX_STEPS=2000 SLURM_ARRAY_TASK_ID=41 bash hyak/sweep.sbatch   # smoke a center config
```

## 1. Generate the grid

```bash
python hyak/gen_configs.py        # writes hyak/configs.txt (81 lines)
```

Edit the arrays at the top of `gen_configs.py` to resize, then update
`--array=1-<N>` in `sweep.sbatch` to match the printed count.

## 2. Submit

```bash
sbatch hyak/sweep.sbatch
```

`--array=1-81` runs all 81 configs with no concurrency cap (Slurm schedules as
many as `ckpt` has idle nodes for). To throttle, append `%N` (e.g. `1-81%20`).

## 3. Monitor

```bash
squeue -A weirdlab --me                 # queue state
tail -f hyak/logs/qoil-sweep_*_<idx>.out
hyakalloc                               # idle capacity / your ckpt limits
```

W&B: project **`wrl-qoil-sweep`**, group **`transport-sweep-v1`** — every config
is one run named `es025_bc01_b01_s0`-style; group the panel by `edit_scale` /
`bc_weight` / `intervention_bonus` to read the axes.

## 4. Collect

Per-config best-eval checkpoints land in `checkpoints/sweep/qoil_<run>_transport.pkl`
(`{params, eval, learner}`). Compare `eval/success` across the group in W&B.

## Notes
- The expert (π_h action + V* critic) loads from the vendored DPPO model in
  `wrl/base_policy/_dppo_min` — byte-identical to the DPPO repo (parity-tested,
  Δ=0.0), so no DPPO code/PYTHONPATH is needed on klone.
- `--http-port 0` disables the in-process HTTP server so co-located array tasks
  don't clash on a port.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` leaves VRAM for the torch DPPO expert
  (JAX learner + V* critic share the GPU with it).
- `--base-n-sample-steps 8` runs the frozen base DP at 8 Euler steps (≈identical
  to 32: max action diff 0.03) for ~15ms/chunk less.
- `--requeue` + `--time=10:00:00`: if you later make runs resumable, Slurm will
  re-run preempted tasks automatically within the 10h window.
