#!/bin/bash
#SBATCH --job-name=lenet_sgd_redo
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --array=0-24
#SBATCH --output=logs/lenet_sgd_redo_%A_%a.out
#SBATCH --error=logs/lenet_sgd_redo_%A_%a.err

# ============================================================
# RESUBMIT: LeNet SGD tasks (original array indices 29-55 minus already-completed)
# ============================================================
# WHY STUCK: same root cause as MLP SGD — one giant batch of 30,000 models.
# For LeNet (30 epochs, lr=0.001), the periodic per-epoch Lipschitz evaluation
# calls compute_lipschitz_constant with batch_size=1 × 16 samples = 16 Jacobian
# computations per epoch, each creating a massive tensor for 30,000 models.
# Total wall time per training call: 100-250h → never finishes within CELL_TIMEOUT.
#
# FIX: reduce MCTBS to 1000 → model_count = 1000/2 = 500 per batch.
# Each training call then takes ~15-30 min → fits in CELL_TIMEOUT (2h).
# Target=5000 kept from original config (SGD converges well on this small dataset
# for small/medium widths once batch size is reasonable).
#
# NOTE on large widths (w=8, 16):
#   These MIGHT have low acceptance rate (same as guess). If they time out with 0 models
#   repeatedly, consider cancelling those subtasks. Monitor lenet_sgd_redo_*_*.out.
#
# ALREADY COMPLETED in original job 8235692 (do NOT cancel these):
#   idx 28: SGD_uniform_w0.25       (target met from prior run)
#   idx 42: SGD_kaiming_uniform_w0.25
#   idx 49: SGD_kaiming_normal_w0.25
#
# NOTE: idx 45 (SGD_kaiming_uniform_w2) and idx 53 (SGD_kaiming_normal_w4) were
#   incorrectly marked as done — their DBs have only 1-2 COMPLETE models.
#   Added back as tasks 23 and 24.
#
# CANCEL BEFORE SUBMITTING:
#   scancel 8235692_29 8235692_30 8235692_31 8235692_32 8235692_33 8235692_34
#   scancel 8235692_35 8235692_36 8235692_37 8235692_38 8235692_39 8235692_40
#   scancel 8235692_41 8235692_43 8235692_44 8235692_46 8235692_47 8235692_48
#   scancel 8235692_50 8235692_51 8235692_52 8235692_54 8235692_55
# ============================================================

export CUDA_VISIBLE_DEVICES=""

# 25 remaining SGD LeNet tasks:
# Original SGD indices 28-55, mapping: OPT=SGD, IDX_in_SGD_block = orig_idx - 28
# Actually done: orig 28,42,49 → SGD_block 0,14,21
# Remaining 25: SGD_block 1-13, 15-16, 17, 18-20, 22-24, 25, 26-27

INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(0.25 0.5 1 2 4 8 16)
N_INITS=${#INITS[@]}   # 4
N_WIDTHS=${#WIDTHS[@]} # 7

# Explicit list of the 25 remaining SGD-block indices (0-27 in the SGD block = orig 28-55)
# SGD_block_done: 0(unif,0.25), 14(kaim_unif,0.25), 21(kaim_norm,0.25)
# Re-added: 17(kaim_unif,w2) as task 23, 25(kaim_norm,w4) as task 24
REMAINING_SGDBLOCK=(1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 18 19 20 22 23 24 26 27 17 25)

IDX=$SLURM_ARRAY_TASK_ID
SGD_BLOCK_IDX=${REMAINING_SGDBLOCK[$IDX]}

INIT_IDX=$((SGD_BLOCK_IDX / N_WIDTHS))
WIDTH_IDX=$((SGD_BLOCK_IDX % N_WIDTHS))

INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="sgd_${INIT}_w${WIDTH}"
FOLDER="output/mnist_lenet/${TAG}"
CONFIG="configs/mnist_sweep/mnist_lenet_width_sweep_sgd.yaml"

# MCTBS=1000 for all widths → model_count=500 per training call.
# Jacobian size shrinks proportionally → each training call fits in CELL_TIMEOUT.
MCTBS=1000

mkdir -p logs

echo "Task $IDX (sgd_block=$SGD_BLOCK_IDX): init=$INIT width=$WIDTH tag=$TAG MCTBS=$MCTBS target=5000"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.lenet.width "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER" \
    --output.target_model_count 5000
