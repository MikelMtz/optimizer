#!/bin/bash
#SBATCH --job-name=lipschitz
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --array=0-87
#SBATCH --output=logs/lipschitz_%A_%a.out
#SBATCH --error=logs/lipschitz_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# 88 combinations: 2 optimizers x 4 inits x 11 widths
# Encoded as a flat index via SLURM_ARRAY_TASK_ID

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(2 4 8 16 32 64 128 256 512 768 1024)

N_INITS=${#INITS[@]}    # 4
N_WIDTHS=${#WIDTHS[@]}   # 11

IDX=$SLURM_ARRAY_TASK_ID
OPT_IDX=$((IDX / (N_INITS * N_WIDTHS)))
REM=$((IDX % (N_INITS * N_WIDTHS)))
INIT_IDX=$((REM / N_WIDTHS))
WIDTH_IDX=$((REM % N_WIDTHS))

OPT=${OPTIMIZERS[$OPT_IDX]}
INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="${OPT,,}_${INIT}_u${WIDTH}"
FOLDER="output/lipschitz/${TAG}"

if [ "$OPT" == "guess" ]; then
    CONFIG="configs/lipschitz_guess.yaml"
    # Scale down model_count_times_batch_size for larger widths
    if [ "$WIDTH" -ge 512 ]; then MCTBS=1600000
    elif [ "$WIDTH" -ge 128 ]; then MCTBS=8000000
    elif [ "$WIDTH" -ge 32 ]; then MCTBS=24000000
    else MCTBS=48000000; fi
else
    CONFIG="configs/lipschitz_sgd.yaml"
    if [ "$WIDTH" -ge 512 ]; then MCTBS=480000
    elif [ "$WIDTH" -ge 128 ]; then MCTBS=1200000
    elif [ "$WIDTH" -ge 32 ]; then MCTBS=2400000
    else MCTBS=4800000; fi
fi

mkdir -p logs

echo "Task $IDX: opt=$OPT init=$INIT width=$WIDTH tag=$TAG"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER"
