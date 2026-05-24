#!/bin/bash
#SBATCH --job-name=kink_sample_sweep
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --array=0-87
#SBATCH --output=logs/kink_sample_sweep_%A_%a.out
#SBATCH --error=logs/kink_sample_sweep_%A_%a.err

export CUDA_VISIBLE_DEVICES=""

# ── 88 tasks: 40 width sweep + 48 depth sweep ──────────────────────
#
# Width sweep (tasks  0-39): 2 optimizers × 4 inits × 5 widths  (layers=1)
# Depth sweep (tasks 40-87): 2 optimizers × 4 inits × 6 depths  (hidden=10)
#
# Each task sweeps over 9 sample sizes: 2,4,6,8,12,16,20,26,30
# Each task collects 100 perfect models per sample size.
# Across 4 inits → 100 × 4 = 400 trained networks per algorithm.

OPTIMIZERS=("guess" "SGD")
INITS=("uniform" "uniform_02" "kaiming_uniform" "kaiming_normal")
WIDTHS=(2 8 32 128 512)
DEPTHS=(1 2 3 4 5 6)
NUM_SAMPLES="2,4,6,8,12,16,20,26,30"

N_INITS=${#INITS[@]}       # 4
N_WIDTHS=${#WIDTHS[@]}     # 5
N_DEPTHS=${#DEPTHS[@]}     # 6
WIDTH_TASKS=$((2 * N_INITS * N_WIDTHS))  # 48

IDX=$SLURM_ARRAY_TASK_ID

mkdir -p logs

if [ "$IDX" -lt "$WIDTH_TASKS" ]; then
    # ── Width Sweep (tasks 0-47) ──────────────────────────────────
    OPT_IDX=$((IDX / (N_INITS * N_WIDTHS)))
    REM=$((IDX % (N_INITS * N_WIDTHS)))
    INIT_IDX=$((REM / N_WIDTHS))
    WIDTH_IDX=$((REM % N_WIDTHS))

    OPT=${OPTIMIZERS[$OPT_IDX]}
    INIT=${INITS[$INIT_IDX]}
    WIDTH=${WIDTHS[$WIDTH_IDX]}

    TAG="${OPT,,}_${INIT}_w${WIDTH}"
    FOLDER="output/kink_sample_sweep/width/${TAG}"

    if [ "$OPT" == "guess" ]; then
        CONFIG="configs/kink_sweep/kink_mlp_width_sweep_guess.yaml"
    else
        CONFIG="configs/kink_sweep/kink_mlp_width_sweep_sgd.yaml"
    fi

    echo "Task $IDX [width]: opt=$OPT init=$INIT width=$WIDTH"

    python train_distributed_bias_0.py \
        --config "$CONFIG" \
        --model.mlp.hidden_units "$WIDTH" \
        --model.init "$INIT" \
        --distributed.num_samples "$NUM_SAMPLES" \
        --distributed.target_model_count_subrun 100 \
        --output.target_model_count 100 \
        --output.folder "$FOLDER"

else
    # ── Depth Sweep (tasks 48-95) ─────────────────────────────────
    LOCAL_IDX=$((IDX - WIDTH_TASKS))
    OPT_IDX=$((LOCAL_IDX / (N_INITS * N_DEPTHS)))
    REM=$((LOCAL_IDX % (N_INITS * N_DEPTHS)))
    INIT_IDX=$((REM / N_DEPTHS))
    DEPTH_IDX=$((REM % N_DEPTHS))

    OPT=${OPTIMIZERS[$OPT_IDX]}
    INIT=${INITS[$INIT_IDX]}
    DEPTH=${DEPTHS[$DEPTH_IDX]}

    TAG="${OPT,,}_${INIT}_d${DEPTH}"
    FOLDER="output/kink_sample_sweep/depth/${TAG}"

    if [ "$OPT" == "guess" ]; then
        CONFIG="configs/kink_sweep/kink_mlp_depth_sweep_guess.yaml"
    else
        CONFIG="configs/kink_sweep/kink_mlp_depth_sweep_sgd.yaml"
    fi

    echo "Task $IDX [depth]: opt=$OPT init=$INIT depth=$DEPTH"

    python train_distributed_bias_0.py \
        --config "$CONFIG" \
        --model.mlp.layers "$DEPTH" \
        --model.init "$INIT" \
        --distributed.num_samples "$NUM_SAMPLES" \
        --distributed.target_model_count_subrun 100 \
        --output.target_model_count 100 \
        --model.normalization lipschitz \
        --output.folder "$FOLDER"
fi

