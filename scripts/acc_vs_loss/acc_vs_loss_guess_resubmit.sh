#!/bin/bash

#SBATCH --job-name=avl_guess
#SBATCH --partition=long
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/avl_guess_%A_%a.out
#SBATCH --error=logs/avl_guess_%A_%a.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

# 3 guess configs × 10 workers each = 30 tasks
CONFIGS=(
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_uniform_02.yaml
    configs/acc_vs_loss_bias_paper/mnist_guess_kaiming_normal.yaml
)

NUM_CONFIGS=${#CONFIGS[@]}
CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

mkdir -p logs
echo "Task $SLURM_ARRAY_TASK_ID → config $CONFIG_IDX: ${CONFIGS[$CONFIG_IDX]}"
python train_distributed.py -C "${CONFIGS[$CONFIG_IDX]}"
