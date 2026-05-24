#!/bin/bash

#SBATCH --job-name=avl_sgd
#SBATCH --partition=gpulong
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/avl_sgd_%A_%a.out
#SBATCH --error=logs/avl_sgd_%A_%a.err

export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

mkdir -p logs
echo "Task $SLURM_ARRAY_TASK_ID: mnist_sgd_uniform (GPU)"
python train_distributed.py -C configs/acc_vs_loss_bias_paper/mnist_sgd_uniform.yaml
