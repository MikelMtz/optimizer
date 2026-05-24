#!/bin/bash
#python train_distributed.py -C configs/table2/mnist_guess.yaml
#python train_distributed.py -C configs/table2/mnist_linear.yaml --distributed.excluded_cells="32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
#python train_distributed.py -C configs/table2/mnist_sgd.yaml --distributed.excluded_cells="32_(0.3, 0.35)/32_(0.35, 0.4)"

#SBATCH --job-name=table2_original
#SBATCH --partition=long
#SBATCH --cpus-per-task=20
#SBATCH --mem=80GB
#SBATCH --array=0-79
#SBATCH --output=logs/table2_%A_%a.out
#SBATCH --error=logs/table2_%A_%a.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

# Guess is already completed — only linear and sgd remain
# Array 0-39:  40 parallel workers on mnist_linear
# Array 40-79: 40 parallel workers on mnist_sgd

if [ $SLURM_ARRAY_TASK_ID -lt 40 ]; then
    CONFIG="configs/table2/mnist_linear.yaml"
    EXCL="32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
else
    CONFIG="configs/table2/mnist_sgd.yaml"
    EXCL="32_(0.3, 0.35)/32_(0.35, 0.4)"
fi

mkdir -p logs
python train_distributed.py -C "$CONFIG" --distributed.excluded_cells="$EXCL"