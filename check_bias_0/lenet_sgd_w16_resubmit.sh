#!/bin/bash
#SBATCH --job-name=lenet_sgd_w16
#SBATCH --partition=gpulong
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --array=0-3
#SBATCH --output=logs/lenet_sgd_w16_%A_%a.out
#SBATCH --error=logs/lenet_sgd_w16_%A_%a.err

# ============================================================
# RESUBMIT: LeNet SGD w=16, all 4 inits
# ============================================================
# WHY: lenet_sgd_resubmit.sh (job 8953141) used --mem=20G (OOM in 2 min).
# lenet_sgd_w16_resubmit.sh first run used --mem=60G (OOM after 16h, MaxRSS=60.6GB).
# Root cause: params+grads alone = 44GB for mc=500; PyTorch overhead fills 60G.
# All other widths in that job (w≤8) are unaffected and still running.
#
# WHY MCTBS=200 (not 1000):
#   LeNet w=16 has large grouped convolutions. With mc=500:
#     conv2 weight: (256*500, 96, 5,5) = 12.3GB; fc1: (1920*500, 4096,1,1) = 15.7GB
#     params + gradients = 44GB, PyTorch overhead pushes to ~60GB → OOM at 60G limit.
#   With mc=100: params+grads = 8.9GB, total ~13GB → safe with --mem=20G.
#   The outer while loop runs 50 cells of 100 models to accumulate 5000 target.
# ============================================================

export CUDA_VISIBLE_DEVICES=""

INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
IDX=$SLURM_ARRAY_TASK_ID
INIT=${INITS[$IDX]}
WIDTH=16

TAG="sgd_${INIT}_w${WIDTH}"
FOLDER="output/mnist_lenet/${TAG}"
CONFIG="configs/mnist_sweep/mnist_lenet_width_sweep_sgd.yaml"
MCTBS=200

mkdir -p logs

echo "Task $IDX: init=$INIT width=$WIDTH tag=$TAG MCTBS=$MCTBS target=5000"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.lenet.width "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER" \
    --output.target_model_count 5000
