#!/bin/bash
#SBATCH --job-name=mnist_lenet
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-55
#SBATCH --output=logs/mnist_lenet_%A_%a.out
#SBATCH --error=logs/mnist_lenet_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# 56 combinations: 2 optimizers x 4 inits x 7 widths
# Encoded as a flat index via SLURM_ARRAY_TASK_ID

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(0.25 0.5 1 2 4 8 16)

N_INITS=${#INITS[@]}    # 4
N_WIDTHS=${#WIDTHS[@]}   # 7

IDX=$SLURM_ARRAY_TASK_ID
OPT_IDX=$((IDX / (N_INITS * N_WIDTHS)))
REM=$((IDX % (N_INITS * N_WIDTHS)))
INIT_IDX=$((REM / N_WIDTHS))
WIDTH_IDX=$((REM % N_WIDTHS))

OPT=${OPTIMIZERS[$OPT_IDX]}
INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="${OPT,,}_${INIT}_w${WIDTH}"
FOLDER="output/mnist_lenet/${TAG}"

if [ "$OPT" == "guess" ]; then
    CONFIG="configs/mnist_sweep/mnist_lenet_width_sweep_guess.yaml"
    # Scale down model_count_times_batch_size for larger widths
    # LeNet fc1 memory scales as MC * 120*w * 16*w * 16 * 4 bytes ∝ MC * w^2
    if [ "$(echo "$WIDTH >= 16" | bc)" -eq 1 ]; then MCTBS=8000
    elif [ "$(echo "$WIDTH >= 8" | bc)" -eq 1 ]; then MCTBS=16000
    elif [ "$(echo "$WIDTH >= 4" | bc)" -eq 1 ]; then MCTBS=40000
    elif [ "$(echo "$WIDTH >= 2" | bc)" -eq 1 ]; then MCTBS=80000
    elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then MCTBS=160000
    else MCTBS=320000; fi
else
    CONFIG="configs/mnist_sweep/mnist_lenet_width_sweep_sgd.yaml"
    # SGD has batch_size=2 so model_count = MCTBS/2 — need much lower MCTBS
    if [ "$(echo "$WIDTH >= 16" | bc)" -eq 1 ]; then MCTBS=2000
    elif [ "$(echo "$WIDTH >= 8" | bc)" -eq 1 ]; then MCTBS=4000
    elif [ "$(echo "$WIDTH >= 4" | bc)" -eq 1 ]; then MCTBS=8000
    elif [ "$(echo "$WIDTH >= 2" | bc)" -eq 1 ]; then MCTBS=16000
    elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then MCTBS=120000
    else MCTBS=240000; fi
fi

mkdir -p logs

echo "Task $IDX: opt=$OPT init=$INIT width=$WIDTH tag=$TAG"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.lenet.width "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER"
