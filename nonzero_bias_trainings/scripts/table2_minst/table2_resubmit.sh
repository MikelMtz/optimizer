#!/bin/bash
#SBATCH --job-name=table2_resubmit
#SBATCH --partition=long 
#SBATCH --cpus-per-task=20
#SBTACH --mem=80GB
#SBTACH --array=0-89
#SBATCH --output=logs/table2_resubmit_%A_%a.out
#SBATCH --error=logs/table2_resubmit_%A_%a.err

export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20

if [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    CONFIG="configs/table2/mnist_guess.yaml"
    EXCL=""
elif [ $SLURM_ARRAY_TASK_ID -lt 40 ]; then
    CONFIG="configs/table2/mnist_linear.yaml"
    EXCL="32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
else
    CONFIG="configs/table2/mnist_sgd.yaml"
    EXCL="32_(0.3, 0.35)/32_(0.35, 0.4)"
fi

mkdir -p logs
python train_distributed.py -C "$CONFIG" --distributed.excluded_cells="$EXCL"