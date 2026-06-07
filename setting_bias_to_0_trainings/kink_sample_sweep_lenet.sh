#!/bin/bash
#SBATCH --job-name=kink_lenet_sweep
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-87
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/kink_lenet_sweep_%A_%a.out
#SBATCH --error=logs/kink_lenet_sweep_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# ── 88 tasks: 56 width sweep + 32 depth sweep ────────────────────────────────
#
# Width sweep (tasks  0-55): 2 optimizers × 4 inits × 7 widths  (depth=4, standard 2c-3f)
# Depth sweep (tasks 56-87): 2 optimizers × 4 inits × 4 depths  (width=1.0)
#   depth 4 = 2c-3f  (standard LeNet)
#   depth 3 = 2c-2f  (drop fc1)
#   depth 2 = 2c-1f  (drop fc1 + fc2)
#   depth 1 = 1c-1f  (drop conv2 + fc1 + fc2)
#
# Each task sweeps over 9 sample sizes: 2,4,6,8,12,16,20,26,30
# Each task collects 100 perfect models per sample size.
#
# Kink 2D points are embedded into 28×28 single-channel images
# (each point maps to its pixel grid cell; all other pixels are 0).

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(0.25 0.5 1 2 4 8 16)
DEPTHS=(1 2 3 4)
NUM_SAMPLES="2,4,6,8,12,16,20,26,30"

N_INITS=${#INITS[@]}       # 4
N_WIDTHS=${#WIDTHS[@]}     # 7
N_DEPTHS=${#DEPTHS[@]}     # 4
WIDTH_TASKS=$((2 * N_INITS * N_WIDTHS))  # 56

IDX=$SLURM_ARRAY_TASK_ID

mkdir -p logs

if [ "$IDX" -lt "$WIDTH_TASKS" ]; then
    # ── Width Sweep (tasks 0-55) ──────────────────────────────────────────────
    OPT_IDX=$((IDX / (N_INITS * N_WIDTHS)))
    REM=$((IDX % (N_INITS * N_WIDTHS)))
    INIT_IDX=$((REM / N_WIDTHS))
    WIDTH_IDX=$((REM % N_WIDTHS))

    OPT=${OPTIMIZERS[$OPT_IDX]}
    INIT=${INITS[$INIT_IDX]}
    WIDTH=${WIDTHS[$WIDTH_IDX]}

    TAG="${OPT,,}_${INIT}_w${WIDTH}"
    FOLDER="output/kink_lenet_sweep/width/${TAG}"

    if [ "$OPT" == "guess" ]; then
        CONFIG="configs/kink_sweep/kink_lenet_width_sweep_guess.yaml"
        # Scale MCTBS down for wider models — fc1 memory ∝ w²
        if [ "$(echo "$WIDTH >= 16" | bc)" -eq 1 ]; then MCTBS=8000
        elif [ "$(echo "$WIDTH >= 8" | bc)" -eq 1 ]; then MCTBS=16000
        elif [ "$(echo "$WIDTH >= 4" | bc)" -eq 1 ]; then MCTBS=40000
        elif [ "$(echo "$WIDTH >= 2" | bc)" -eq 1 ]; then MCTBS=80000
        elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then MCTBS=160000
        else MCTBS=320000; fi
    else
        CONFIG="configs/kink_sweep/kink_lenet_width_sweep_sgd.yaml"
        if [ "$(echo "$WIDTH >= 16" | bc)" -eq 1 ]; then MCTBS=500
        elif [ "$(echo "$WIDTH >= 8" | bc)" -eq 1 ]; then MCTBS=4000
        elif [ "$(echo "$WIDTH >= 4" | bc)" -eq 1 ]; then MCTBS=8000
        elif [ "$(echo "$WIDTH >= 2" | bc)" -eq 1 ]; then MCTBS=16000
        elif [ "$(echo "$WIDTH >= 0.5" | bc)" -eq 1 ]; then MCTBS=120000
        else MCTBS=240000; fi
    fi

    echo "Task $IDX [width]: opt=$OPT init=$INIT width=$WIDTH"

    python train_distributed_bias_0.py \
        --config "$CONFIG" \
        --model.lenet.width "$WIDTH" \
        --model.init "$INIT" \
        --model.model_count_times_batch_size "$MCTBS" \
        --distributed.num_samples "$NUM_SAMPLES" \
        --distributed.target_model_count_subrun 100 \
        --output.target_model_count 100 \
        --output.folder "$FOLDER"

else
    # ── Depth Sweep (tasks 56-87) ─────────────────────────────────────────────
    LOCAL_IDX=$((IDX - WIDTH_TASKS))
    OPT_IDX=$((LOCAL_IDX / (N_INITS * N_DEPTHS)))
    REM=$((LOCAL_IDX % (N_INITS * N_DEPTHS)))
    INIT_IDX=$((REM / N_DEPTHS))
    DEPTH_IDX=$((REM % N_DEPTHS))

    OPT=${OPTIMIZERS[$OPT_IDX]}
    INIT=${INITS[$INIT_IDX]}
    DEPTH=${DEPTHS[$DEPTH_IDX]}

    TAG="${OPT,,}_${INIT}_d${DEPTH}"
    FOLDER="output/kink_lenet_sweep/depth/${TAG}"

    if [ "$OPT" == "guess" ]; then
        CONFIG="configs/kink_sweep/kink_lenet_depth_sweep_guess.yaml"
        MCTBS=80000
    else
        CONFIG="configs/kink_sweep/kink_lenet_depth_sweep_sgd.yaml"
        MCTBS=60000
    fi

    echo "Task $IDX [depth]: opt=$OPT init=$INIT depth=$DEPTH"

    python train_distributed_bias_0.py \
        --config "$CONFIG" \
        --model.lenet.layers "$DEPTH" \
        --model.init "$INIT" \
        --model.model_count_times_batch_size "$MCTBS" \
        --distributed.num_samples "$NUM_SAMPLES" \
        --distributed.target_model_count_subrun 100 \
        --output.target_model_count 100 \
        --output.folder "$FOLDER"
fi
