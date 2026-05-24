#!/bin/bash

#SBATCH --job-name=acc_vs_loss_cpu
#SBATCH --partition=long
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --array=0-23
#SBATCH --output=logs/acc_vs_loss_%A_%a.out
#SBATCH --error=logs/acc_vs_loss_%A_%a.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

# Skip kaiming_uniform/normal SGD — already done
# 4 parallel workers per config (24 total / 6 configs)
CONFIGS=(
    configs/acc_vs_loss_bias_paper/mnist_sgd_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_sgd_uniform_02.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform_02.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_kaiming_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_kaiming_normal.yaml
)

NUM_CONFIGS=${#CONFIGS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

mkdir -p logs
python train_distributed.py -C "${CONFIGS[$CONFIG_IDX]}"