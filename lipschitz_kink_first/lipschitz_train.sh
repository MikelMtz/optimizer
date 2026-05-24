#!/bin/bash
#SBATCH --job-name=lipschitz
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --array=0-55
#SBATCH --output=logs/lipschitz_%A_%a.out
#SBATCH --error=logs/lipschitz_%A_%a.err

# 56 combinations: 2 optimizers x 4 inits x 7 widths
# Encoded as a flat index via SLURM_ARRAY_TASK_ID

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(2 4 6 8 10 15 20)

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

TAG="${OPT,,}_${INIT}_u${WIDTH}"
FOLDER="output/lipschitz/${TAG}"

if [ "$OPT" == "guess" ]; then
    CONFIG="configs/lipschitz_guess.yaml"
    # Scale down model_count_times_batch_size for large widths
    if [ "$WIDTH" -ge 20 ]; then MCTBS=24000000; else MCTBS=48000000; fi
else
    CONFIG="configs/lipschitz_sgd.yaml"
    if [ "$WIDTH" -ge 15 ]; then MCTBS=2400000; else MCTBS=4800000; fi
fi

mkdir -p logs

echo "Task $IDX: opt=$OPT init=$INIT width=$WIDTH tag=$TAG"

python train_distributed.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER"
