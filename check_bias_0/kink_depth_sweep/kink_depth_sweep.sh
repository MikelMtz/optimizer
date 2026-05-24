#!/bin/bash
#SBATCH --job-name=kink_depth_sweep
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --array=0-47
#SBATCH --output=logs/kink_depth_sweep_%A_%a.out
#SBATCH --error=logs/kink_depth_sweep_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# 48 combinations: 2 optimizers x 4 inits x 6 depths
# Encoded as a flat index via SLURM_ARRAY_TASK_ID
# Same grid as lipschitz_train.sh but with --model.normalization lipschitz

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
DEPTHS=(1 2 3 4 5 6)

N_INITS=${#INITS[@]}    # 4
N_DEPTHS=${#DEPTHS[@]}   # 6

IDX=$SLURM_ARRAY_TASK_ID
OPT_IDX=$((IDX / (N_INITS * N_DEPTHS)))
REM=$((IDX % (N_INITS * N_DEPTHS)))
INIT_IDX=$((REM / N_DEPTHS))
DEPTH_IDX=$((REM % N_DEPTHS))

OPT=${OPTIMIZERS[$OPT_IDX]}
INIT=${INITS[$INIT_IDX]}
DEPTH=${DEPTHS[$DEPTH_IDX]}

TAG="${OPT,,}_${INIT}_d${DEPTH}"
FOLDER="output/kink_depth_sweep/${TAG}"

if [ "$OPT" == "guess" ]; then
    CONFIG="configs/kink_sweep/kink_mlp_depth_sweep_guess.yaml"
    # Scale down model_count_times_batch_size for larger depths
    if [ "$DEPTH" -ge 5 ]; then MCTBS=1600000
    elif [ "$DEPTH" -ge 3 ]; then MCTBS=8000000
    elif [ "$DEPTH" -ge 1 ]; then MCTBS=24000000
    else MCTBS=48000000; fi
else
    CONFIG="configs/kink_sweep/kink_mlp_depth_sweep_sgd.yaml"
    if [ "$DEPTH" -ge 5 ]; then MCTBS=480000
    elif [ "$DEPTH" -ge 3 ]; then MCTBS=1200000
    elif [ "$DEPTH" -ge 1 ]; then MCTBS=2400000
    else MCTBS=4800000; fi
fi

mkdir -p logs

echo "Task $IDX: opt=$OPT init=$INIT depth=$DEPTH tag=$TAG"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.layers "$DEPTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --model.normalization lipschitz \
    --output.folder "$FOLDER"
