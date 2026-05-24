#!/bin/bash
#SBATCH --job-name=mlp_guess_redo
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-15
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/mlp_guess_redo_%A_%a.out
#SBATCH --error=logs/mlp_guess_redo_%A_%a.err

# ============================================================
# RESUBMIT: MLP guess large-width tasks (u64, u128, u256)
# ============================================================
# WHY STUCK: guess optimizer acceptance rate is ~0.007%, so each 2h cell (CELL_TIMEOUT)
# produces ~2 models. At target=500, these need 250 cells × 2h = 500h (21 days).
# FIX: lower --output.target_model_count to 200. All tasks currently have 130-296
# COMPLETE models; they need at most 200 more hours to reach 200.
#
# NOTE: u512 is excluded. All u512 tasks have 0 COMPLETE models after 229h because the
# Lipschitz-normalized u512 network has ~0% acceptance rate with random init. Rerunning
# them would be a waste of compute.
#
# CANCEL BEFORE SUBMITTING:
#   scancel 8573026_2 8573026_3 8573026_4    # guess_uniform_u64/u128/u256
#   scancel 8573026_8 8573026_9 8573026_10   # guess_uniform_02_u64/u128/u256
#   scancel 8573026_14 8573026_15 8573026_16  # guess_kaiming_uniform_u64/u128/u256
#   scancel 8573026_19 8573026_21 8573026_22  # guess_kaiming_normal_u32/u128/u256
#   (idx 5,11,17,23 are u512 → do NOT resubmit)
#   (idx 24,26 are SGD → handled by mlp_sgd_all_resubmit.sh)
# ============================================================

export CUDA_VISIBLE_DEVICES=""

# 16 combinations: 4 inits × 4 widths (u32, u64, u128, u256)
# u32 included for kaiming_normal only (idx 19 in original job, has 466 models → needs 34 more)
# All others are u64/u128/u256 × all 4 inits

INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(64 128 256 32)   # 32 only relevant for kaiming_normal

# Flat mapping: 4 inits × 3 widths (u64/u128/u256) + 1 task for kaiming_normal_u32 = 13 tasks
# Use a simpler explicit mapping for clarity

declare -A TASK_INIT
declare -A TASK_WIDTH

TASK_INIT[0]="uniform";          TASK_WIDTH[0]=64
TASK_INIT[1]="uniform";          TASK_WIDTH[1]=128
TASK_INIT[2]="uniform";          TASK_WIDTH[2]=256
TASK_INIT[3]="uniform_02";       TASK_WIDTH[3]=64
TASK_INIT[4]="uniform_02";       TASK_WIDTH[4]=128
TASK_INIT[5]="uniform_02";       TASK_WIDTH[5]=256
TASK_INIT[6]="kaiming_uniform";  TASK_WIDTH[6]=64
TASK_INIT[7]="kaiming_uniform";  TASK_WIDTH[7]=128
TASK_INIT[8]="kaiming_uniform";  TASK_WIDTH[8]=256
TASK_INIT[9]="kaiming_normal";   TASK_WIDTH[9]=32
TASK_INIT[10]="kaiming_normal";  TASK_WIDTH[10]=64
TASK_INIT[11]="kaiming_normal";  TASK_WIDTH[11]=128
TASK_INIT[12]="kaiming_normal";  TASK_WIDTH[12]=256
TASK_INIT[13]="kaiming_normal";  TASK_WIDTH[13]=512  # was making progress (178 models) before cluster-wide cancel on 2026-05-06

IDX=$SLURM_ARRAY_TASK_ID
INIT=${TASK_INIT[$IDX]}
WIDTH=${TASK_WIDTH[$IDX]}

if [ -z "$INIT" ]; then
    echo "Task $IDX: undefined. Skipping."
    exit 0
fi

TAG="guess_${INIT}_u${WIDTH}"
FOLDER="output/mnist_mlp/${TAG}"
CONFIG="configs/mnist_sweep/mnist_mlp_width_sweep_guess.yaml"

# MCTBS scaling by width (same as original script)
if   [ "$WIDTH" -ge 512 ]; then MCTBS=80000
elif [ "$WIDTH" -ge 256 ]; then MCTBS=160000
elif [ "$WIDTH" -ge 128 ]; then MCTBS=320000
elif [ "$WIDTH" -ge 64  ]; then MCTBS=640000
else MCTBS=1600000; fi

mkdir -p logs

echo "Task $IDX: init=$INIT width=$WIDTH tag=$TAG MCTBS=$MCTBS target=200"

# target_model_count=200: at ~1 model/h, tasks with 130-200 models need <70h (3 days);
# tasks already above 200 terminate immediately.
python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER" \
    --output.target_model_count 200
