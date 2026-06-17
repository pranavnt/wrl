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
#SBATCH --array=1-81%16
#SBATCH --output=/gscratch/weirdlab/pranavnt/wrl/hyak/logs/%x_%A_%a.out

set -euo pipefail

# ---- paths (override by exporting before sbatch, or edit here) -------------
WRL=${WRL:-/gscratch/weirdlab/pranavnt/wrl}
DPPO_ROOT=${DPPO_ROOT:-/gscratch/weirdlab/pranavnt/dppo}
BASE=${BASE:-$WRL/checkpoints/flowdp_transport_pixel_avg_step45000.pkl}
DPPO_CKPT=${DPPO_CKPT:-$DPPO_ROOT/log/robomimic-finetune/transport_ft_diffusion_mlp_ta8_td20_tdf10/checkpoint/state_200.pt}
DPPO_CONFIG=${DPPO_CONFIG:-$DPPO_ROOT/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml}
DPPO_NORM=${DPPO_NORM:-$DPPO_ROOT/data/robomimic/transport/normalization.npz}
DATASET=${DATASET:-$WRL/data/robomimic/transport/ph/image_84.hdf5}

WANDB_PROJECT=${WANDB_PROJECT:-wrl-qoil-sweep}
WANDB_GROUP=${WANDB_GROUP:-transport-sweep-v1}
MAX_STEPS=${MAX_STEPS:-80000}

cd "$WRL"

# ---- env ------------------------------------------------------------------
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$DPPO_ROOT:${PYTHONPATH:-}"   # DPPO model classes for the expert
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7        # leave VRAM for the torch DPPO expert
export WANDB_SILENT=true

# ---- pick this task's config ----------------------------------------------
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$WRL/hyak/configs.txt")
read -r EDIT BC BONUS SEED <<< "$LINE"
RUN="es${EDIT/./}_bc${BC/./}_b${BONUS/./}_s${SEED}"
echo "[sweep] task=$SLURM_ARRAY_TASK_ID run=$RUN  (edit=$EDIT bc=$BC bonus=$BONUS seed=$SEED)"
echo "[sweep] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES base=$BASE"

# ---- run (uv-managed venv) ------------------------------------------------
uv run python examples/transport/train_qoil.py \
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
