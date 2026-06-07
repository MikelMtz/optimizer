#!/bin/bash
#SBATCH --job-name=lenet_guess_redo
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-19
#SBATCH --output=logs/lenet_guess_redo_%A_%a.out
#SBATCH --error=logs/lenet_guess_redo_%A_%a.err

# ============================================================
# RESUBMIT: LeNet guess tasks (original array indices 0-27)
# ============================================================
# WHY STUCK (target 50,000 unreachable):
#   - Each 2h CELL_TIMEOUT produces ~46 models for w=0.25, down to ~1 for w=4.
#   - At 50,000 target: w=0.25 needs ~2,200h; w=4 needs ~100,000h. Impossible in 1 week.
#
# FIX: per-width targets calibrated to be achievable in ~1 week from current state.
#   DB already accumulates models; restarting picks up where it left off.
#   Targets chosen so (current_models + rate/h × 168h) >= target:
#     w=0.25: rate=20/h, currently 4714 → target=8000 (finish in ~2 days)
#     w=0.5:  rate=7/h,  currently 1543 → target=2500 (finish in ~1.5 days)
#     w=1:    rate=4/h,  currently  916 → target=1500 (finish in ~1.5 days)
#     w=2:    rate=0.8/h,currently  192 → target=300  (finish in ~4 days)
#     w=4:    rate=0.9/h,currently  200 → target=300  (finish in ~4 days)
#
# WHY w=8 and w=16 are excluded:
#   After 229h of running, both show 0 COMPLETE models (91-80 PENDING rows, all failed cells).
#   The Lipschitz-normalized large-LeNet produces near-uniform logits on all inputs,
#   making 100% training accuracy essentially unachievable with random init.
#   Resubmitting them is a waste of compute without changing the acceptance criterion.
#
# CANCEL BEFORE SUBMITTING:
#   scancel 8235692_0-27
#   (DO NOT cancel 8235692_28-55 — those are SGD, handled by lenet_sgd_resubmit.sh)
# ============================================================

export CUDA_VISIBLE_DEVICES=""

# 20 combinations: 4 inits × 5 widths (0.25, 0.5, 1, 2, 4)
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(0.25 0.5 1 2 4)

N_INITS=${#INITS[@]}   # 4
N_WIDTHS=${#WIDTHS[@]} # 5

IDX=$SLURM_ARRAY_TASK_ID
INIT_IDX=$((IDX / N_WIDTHS))
WIDTH_IDX=$((IDX % N_WIDTHS))

INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="guess_${INIT}_w${WIDTH}"
FOLDER="output/mnist_lenet/${TAG}"
CONFIG="configs/mnist_sweep/mnist_lenet_width_sweep_guess.yaml"

# Per-width target: calibrated from observed rates to finish within 1 week
# Width-to-target mapping using bc for float comparison
if   [ "$(echo "$WIDTH >= 4"   | bc)" -eq 1 ]; then TARGET=300
elif [ "$(echo "$WIDTH >= 2"   | bc)" -eq 1 ]; then TARGET=300
elif [ "$(echo "$WIDTH >= 1"   | bc)" -eq 1 ]; then TARGET=1500
elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then TARGET=2500
else TARGET=8000; fi   # w=0.25

# MCTBS scaling by width (same as original script)
if   [ "$(echo "$WIDTH >= 16"  | bc)" -eq 1 ]; then MCTBS=8000
elif [ "$(echo "$WIDTH >= 8"   | bc)" -eq 1 ]; then MCTBS=16000
elif [ "$(echo "$WIDTH >= 4"   | bc)" -eq 1 ]; then MCTBS=40000
elif [ "$(echo "$WIDTH >= 2"   | bc)" -eq 1 ]; then MCTBS=80000
elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then MCTBS=160000
else MCTBS=320000; fi

mkdir -p logs

echo "Task $IDX: init=$INIT width=$WIDTH tag=$TAG MCTBS=$MCTBS target=$TARGET"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.lenet.width "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER" \
    --output.target_model_count "$TARGET"
