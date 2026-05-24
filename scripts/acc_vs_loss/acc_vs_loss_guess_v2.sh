#!/bin/bash
# Re-run G&C (guess) experiments with the ORIGINAL config values:
#   model_count_times_batch_size: 16000  (not 1600000)
#   target_model_count_subrun:    100    (not 30)
#
# These settings match the first batch of models that were collected and
# produce a wider Lipschitz-normalized loss distribution (as in the paper).
#
# Writes into the same output folders so new models accumulate alongside
# existing ones until target_model_count (2000) is reached.
#
# SLURM usage (40 tasks = 4 configs × 10 parallel workers):
#   sbatch scripts/acc_vs_loss/acc_vs_loss_guess_v2.sh
#
# Local sequential usage:
#   bash scripts/acc_vs_loss/acc_vs_loss_guess_v2.sh

#SBATCH --job-name=avl_guess_v2
#SBATCH --partition=long
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/avl_guess_v2_%A_%a.out
#SBATCH --error=logs/avl_guess_v2_%A_%a.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

CONFIGS=(
    configs/acc_vs_loss_bias_paper_v2/mnist_guess_uniform.yaml
    configs/acc_vs_loss_bias_paper_v2/mnist_guess_uniform_02.yaml
    configs/acc_vs_loss_bias_paper_v2/mnist_guess_kaiming_uniform.yaml
    configs/acc_vs_loss_bias_paper_v2/mnist_guess_kaiming_normal.yaml
)

NUM_CONFIGS=${#CONFIGS[@]}

mkdir -p logs

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # SLURM: 10 workers per config
    CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))
    echo "Task $SLURM_ARRAY_TASK_ID → config $CONFIG_IDX: ${CONFIGS[$CONFIG_IDX]}"
    python train_distributed.py -C "${CONFIGS[$CONFIG_IDX]}"
else
    # Local: run all configs sequentially
    for cfg in "${CONFIGS[@]}"; do
        echo "========================================"
        echo "Running: $cfg"
        echo "========================================"
        python train_distributed.py -C "$cfg"
    done
fi
