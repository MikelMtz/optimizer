#!/bin/bash
#SBATCH --job-name=mlp_sgd_redo
#SBATCH --partition=long,kocsisgpu,gpulong
#SBATCH --exclude=gpu01,gpu02,gpu03,gpu05,gpu06
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --array=0-23
#SBATCH --output=logs/mlp_sgd_redo_%A_%a.out
#SBATCH --error=logs/mlp_sgd_redo_%A_%a.err

# ============================================================
# RESUBMIT: ALL MLP SGD tasks (original indices 24-47)
# ============================================================
# WHY STUCK: each training call processes 30,000 models simultaneously for 100 SGD epochs.
# The per-epoch Lipschitz norm evaluation (called with batch_size=1 on 16 training samples)
# creates a Jacobian tensor of shape (8, 30000, 784, W) ≈ 12GB, repeated 16× per epoch × 100
# epochs = 1,600 Lipschitz calls per training call → total wall time ~150h per batch.
# The CELL_TIMEOUT (2h) never fires because training itself exceeds 2h.
#
# FIX: drastically reduce model_count_times_batch_size.
# With batch_size=2 and MCTBS=1000 → model_count=500.
# Jacobian tensor: (8, 500, 784, W) = 200MB → fast.
# Per training call: ~100 epochs × 16 Lip calls × <1s each = ~25 min total → fits in CELL_TIMEOUT.
# SGD converges easily on 16 MNIST samples (overparameterized, kaiming init) → high acceptance
# rate → target=500 reached in the first few cells.
#
# MEMORY: 500 models × 3.6MB (u64) = 1.8GB → --mem=30G is generous.
#
# CANCEL BEFORE SUBMITTING:
#   scancel 8573026_24 8573026_26          # stuck running SGD tasks
#   scancel 8573026_[27-47]               # pending SGD tasks
# ============================================================

export CUDA_VISIBLE_DEVICES=""

# 24 combinations: 4 inits × 6 widths (all SGD, original indices 24-47)
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(16 32 64 128 256 512)

N_INITS=${#INITS[@]}   # 4
N_WIDTHS=${#WIDTHS[@]} # 6

IDX=$SLURM_ARRAY_TASK_ID
INIT_IDX=$((IDX / N_WIDTHS))
WIDTH_IDX=$((IDX % N_WIDTHS))

INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="sgd_${INIT}_u${WIDTH}"
FOLDER="output/mnist_mlp/${TAG}"
CONFIG="configs/mnist_sweep/mnist_mlp_width_sweep_sgd.yaml"

# MCTBS=1000 for all widths → model_count = 1000/2 = 500 per batch.
# This ensures each inner training call completes within the 2h CELL_TIMEOUT.
MCTBS=1000

mkdir -p logs

echo "Task $IDX: init=$INIT width=$WIDTH tag=$TAG MCTBS=$MCTBS target=500"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER" \
    --output.target_model_count 500
