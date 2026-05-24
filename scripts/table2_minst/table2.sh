#!/bin/bash
#python train_distributed.py -C configs/table2/mnist_guess.yaml
#python train_distributed.py -C configs/table2/mnist_linear.yaml --distributed.excluded_cells="32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
#python train_distributed.py -C configs/table2/mnist_sgd.yaml --distributed.excluded_cells="32_(0.3, 0.35)/32_(0.35, 0.4)"

#SBATCH --job-name=table2_original
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --array=0-2
#SBATCH --output=logs/table2_%A_%a.out
#SBATCH --error=logs/table2_%A_%a.err

CONFIGS=(
    configs/table2/mnist_guess.yaml
    configs/table2/mnist_linear.yaml
    configs/table2/mnist_sgd.yaml
)

EXCLUDED=(
    "32_(0.3, 0.35)/32_(0.35, 0.4)/32_(0.4, 0.45)/32_(0.45, 0.5)/32_(0.5, 0.55)/32_(0.55, 0.6)/32_(0.6, 0.65)/16_(0.35, 0.4)/16_(0.3, 0.35)"
    "32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
    "32_(0.3, 0.35)/32_(0.35, 0.4)"
)

mkdir -p logs

EXCL="${EXCLUDED[$SLURM_ARRAY_TASK_ID]}"
if [ -n "$EXCL" ]; then
    python train_distributed.py -C ${CONFIGS[$SLURM_ARRAY_TASK_ID]} --distributed.excluded_cells="$EXCL"
else
    python train_distributed.py -C ${CONFIGS[$SLURM_ARRAY_TASK_ID]}
fi