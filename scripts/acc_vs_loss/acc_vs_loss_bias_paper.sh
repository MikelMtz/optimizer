#!/bin/bash
# Train all 8 configurations for the test-accuracy-vs-train-loss experiment.
# 4 initializations x 2 optimizers (SGD / G&C guess) on MNIST (classes 0 vs 7, 16 training samples).
#
# SLURM usage:
#   sbatch --array=0-7 scripts/acc_vs_loss.sh
#
# Local sequential usage (all configs):
#   bash scripts/acc_vs_loss.sh

#SBATCH --job-name=acc_vs_loss
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --array=0-7
#SBATCH --output=logs/acc_vs_loss_%A_%a.out
#SBATCH --error=logs/acc_vs_loss_%A_%a.err

CONFIGS=(
    configs/acc_vs_loss_bias_paper/mnist_sgd_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_sgd_uniform_02.yaml
    configs/acc_vs_loss_bias_paper/mnist_sgd_kaiming_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_sgd_kaiming_normal.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform_02.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_kaiming_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_kaiming_normal.yaml
)

mkdir -p logs

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # SLURM: run the single config for this array index
    python train_distributed.py -C "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
else
    # Local: run all configs sequentially
    for cfg in "${CONFIGS[@]}"; do
        echo "========================================"
        echo "Running: $cfg"
        echo "========================================"
        python train_distributed.py -C "$cfg"
    done
fi
