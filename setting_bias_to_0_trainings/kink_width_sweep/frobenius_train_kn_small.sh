#!/bin/bash
#SBATCH --job-name=lip_kn_small
#SBATCH --partition=long
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --array=33-35
#SBATCH --output=logs/lipschitz_%A_%a.out
#SBATCH --error=logs/lipschitz_%A_%a.err

# Specialised resubmission for guess / kaiming_normal / u2, u4, u8 (array indices 33-35).
#
# Root cause of OOM at 40G and 80G (persists even at MCTBS=96000 / 6000 models/batch):
#   The inner while loop accumulates ALL perfect model weights in the list
#   perfect_model_weights until it reaches target_model_count_subrun, only
#   THEN concatenates and saves to disk. The peak memory grows with the number
#   of inner-loop iterations, not with the per-batch model count, because
#   Python/PyTorch's CPU allocator cannot return fragmented pages to the OS
#   across many allocation/free cycles, causing RSS to grow monotonically.
#
#   Fix: reduce target_model_count_subrun (here: 10000) so the inner loop
#   exits and frees perfect_model_weights after collecting just 10k models.
#   The outer loop then repeats until the DB reaches target_model_count=500000,
#   accumulating models 10k at a time — peak in-memory weight data stays
#   well under 1 MB per subrun for these tiny widths.

export CUDA_VISIBLE_DEVICES=""

# Tasks 33-35 map to: guess / kaiming_normal / u2, u4, u8
WIDTHS=(2 4 8)
WIDTH_IDX=$(( SLURM_ARRAY_TASK_ID - 33 ))
WIDTH=${WIDTHS[$WIDTH_IDX]}

OPT="guess"
INIT="kaiming_normal"
CONFIG="configs/lipschitz_guess.yaml"

# Original MCTBS (48M, giving 30k models/batch) is fine for memory per-iteration;
# the problem is the inner loop running too long. Keep MCTBS at the original value.
# Reduce target_model_count_subrun to 10000 so perfect_model_weights is flushed
# to disk (and freed) after every 10k models instead of every 500k models.
MCTBS=48000000
TARGET_SUBRUN=10000

TAG="${OPT,,}_${INIT}_u${WIDTH}"
FOLDER="output/lipschitz/${TAG}"

mkdir -p logs

echo "Task $SLURM_ARRAY_TASK_ID: opt=$OPT init=$INIT width=$WIDTH tag=$TAG (MCTBS=$MCTBS, target_subrun=$TARGET_SUBRUN)"

python train_distributed_bias_0.py \
    --config "$CONFIG" \
    --model.mlp.hidden_units "$WIDTH" \
    --model.init "$INIT" \
    --model.model_count_times_batch_size "$MCTBS" \
    --distributed.target_model_count_subrun "$TARGET_SUBRUN" \
    --output.folder "$FOLDER"
