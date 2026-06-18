#!/bin/bash
# Q-OIL transport sweep on the HYAK checkpoint partition.
#   edit_scale x bc_weight x intervention_bonus x seed   (81 configs, see configs.txt)
#
# Submit from /gscratch/weirdlab/pranavnt/wrl after generating configs.txt:
#     python hyak/gen_configs.py
#     sbatch hyak/sweep.sbatch
# Adjust --array if you resize the grid; %16 throttles concurrent jobs.
#
#SBATCH --job-name=qoil-sweep
#SBATCH --account=weirdlab
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=1-81
#SBATCH --output=/gscratch/weirdlab/pranavnt/wrl/hyak/logs/%x_%A_%a.out

set -euo pipefail

# ---- paths (override by exporting before sbatch, or edit here) -------------
# The DPPO expert runs from wrl's VENDORED model copy (wrl/base_policy/_dppo_min)
# -> no DPPO repo needed on klone, only the 3 expert data files below.
WRL=${WRL:-/gscratch/weirdlab/pranavnt/wrl}
EXPERT=${EXPERT:-$WRL/checkpoints/expert}
BASE=${BASE:-$WRL/checkpoints/flowdp_transport_pixel_avg_step45000.pkl}
DPPO_CKPT=${DPPO_CKPT:-$EXPERT/state_200.pt}
DPPO_CONFIG=${DPPO_CONFIG:-$EXPERT/ft_ppo_diffusion_mlp.yaml}
DPPO_NORM=${DPPO_NORM:-$EXPERT/normalization.npz}
DATASET=${DATASET:-$WRL/data/robomimic/transport/ph/low_dim_v141.hdf5}  # env_meta only (no demos loaded)

WANDB_PROJECT=${WANDB_PROJECT:-wrl-qoil-sweep}
WANDB_GROUP=${WANDB_GROUP:-transport-sweep-v1}
CONFIGS=${CONFIGS:-$WRL/hyak/configs.txt}   # override for other sweeps (e.g. configs_v2.txt)
MAX_STEPS=${MAX_STEPS:-80000}

cd "$WRL"

# ---- env ------------------------------------------------------------------
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7        # leave VRAM for the torch DPPO expert
export WANDB_SILENT=true

# ---- pick this task's config ----------------------------------------------
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIGS")
read -r EDIT BC BONUS SEED <<< "$LINE"
RUN="es${EDIT/./}_bc${BC/./}_b${BONUS/./}_s${SEED}"
echo "[sweep] task=$SLURM_ARRAY_TASK_ID run=$RUN  (edit=$EDIT bc=$BC bonus=$BONUS seed=$SEED)"
echo "[sweep] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES base=$BASE"

# ---- run (prebuilt venv: `uv sync --extra cuda` once; direct python avoids
#          81 array tasks each re-checking the uv lock / touching the network) --
"$WRL/.venv/bin/python" examples/transport/train_qoil.py \
    --algo qoil \
    --base "$BASE" \
    --dppo-checkpoint "$DPPO_CKPT" \
    --dppo-config "$DPPO_CONFIG" \
    --dppo-norm "$DPPO_NORM" \
    --dataset-path "$DATASET" \
    --edit-scale "$EDIT" \
    --bc-weight "$BC" \
    --intervention-bonus "$BONUS" \
    --seed "$SEED" \
    --max-steps "$MAX_STEPS" \
    --eval-every 5000 \
    --base-n-sample-steps 8 \
    --http-port 0 \
    --run-name "$RUN" \
    --ckpt-dir "$WRL/checkpoints/sweep" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$WANDB_GROUP"
