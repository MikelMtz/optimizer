#!/bin/bash
#SBATCH --job-name=figure2
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --array=0-13
#SBATCH --output=logs/figure2_%A_%a.out
#SBATCH --error=logs/figure2_%A_%a.err

# All 14 configs run in parallel as independent array tasks.
# Each task picks its config by index ($SLURM_ARRAY_TASK_ID).
# Total CPUs requested: 14 tasks x 4 CPUs = 56, fits within comp01 (64 CPUs, short partition).

CONFIGS=(
    configs/figure2/guess_u2.yaml
    configs/figure2/guess_u4.yaml
    configs/figure2/guess_u6.yaml
    configs/figure2/guess_u8.yaml
    configs/figure2/guess_u10.yaml
    configs/figure2/guess_u15.yaml
    configs/figure2/guess_u20.yaml
    configs/figure2/sgd_u2.yaml
    configs/figure2/sgd_u4.yaml
    configs/figure2/sgd_u6.yaml
    configs/figure2/sgd_u8.yaml
    configs/figure2/sgd_u10.yaml
    configs/figure2/sgd_u15.yaml
    configs/figure2/sgd_u20.yaml
)

mkdir -p logs

python train_distributed.py --config ${CONFIGS[$SLURM_ARRAY_TASK_ID]}