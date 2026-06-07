#!/bin/bash
#SBATCH --job-name=mnist_mlp
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-47
#SBATCH --output=logs/mnist_mlp_%A_%a.out
#SBATCH --error=logs/mnist_mlp_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# 48 combinations: 2 optimizers x 4 inits x 6 widths
# Encoded as a flat index via SLURM_ARRAY_TASK_ID

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(16 32 64 128 256 512)

N_INITS=${#INITS[@]}    # 4
N_WIDTHS=${#WIDTHS[@]}   # 6

IDX=$SLURM_ARRAY_TASK_ID
OPT_IDX=$((IDX / (N_INITS * N_WIDTHS)))
REM=$((IDX % (N_INITS * N_WIDTHS)))
INIT_IDX=$((REM / N_WIDTHS))
WIDTH_IDX=$((REM % N_WIDTHS))

OPT=${OPTIMIZERS[$OPT_IDX]}
INIT=${INITS[$INIT_IDX]}
WIDTH=${WIDTHS[$WIDTH_IDX]}

TAG="${OPT,,}_${INIT}_u${WIDTH}"
FOLDER="output/mnist_mlp/${TAG}"

if [ "$OPT" == "guess" ]; then
    CONFIG="configs/mnist_sweep/mnist_mlp_width_sweep_guess.yaml"
    # Scale down model_count_times_batch_size for larger widths
    # MNIST MLP: W0 memory = MC * 784 * hidden * 4 bytes
    if [ "$WIDTH" -ge 512 ]; then MCTBS=80000
    elif [ "$WIDTH" -ge 256 ]; then MCTBS=160000
    elif [ "$WIDTH" -ge 128 ]; then MCTBS=320000
    elif [ "$WIDTH" -ge 64 ]; then MCTBS=640000
    else MCTBS=1600000; fi
else
    CONFIG="configs/mnist_sweep/mnist_mlp_width_sweep_sgd.yaml"
    # SGD has batch_size=2 so MC = MCTBS/2
    if [ "$WIDTH" -ge 512 ]; then MCTBS=16000
    elif [ "$WIDTH" -ge 256 ]; then MCTBS=32000
    elif [ "$WIDTH" -ge 128 ]; then MCTBS=64000
    elif [ "$WIDTH" -ge 64 ]; then MCTBS=120000
    else MCTBS=240000; fi
fi

mkdir -p logs

echo "Task $IDX: opt=$OPT init=$INIT width=$WIDTH tag=$TAG"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --output.folder "$FOLDER"
