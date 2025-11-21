#!/bin/bash
#SBATCH --job-name=mnist_basic_align
#SBATCH --output=logs/mnist_basic_%j.out
#SBATCH --error=logs/mnist_basic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "Starting MNIST basic alignment experiment at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
source activate alignment || conda activate alignment || true

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_experiment.py \
    --config configs/examples/mnist_basic.yaml \
    --device cpu

echo "MNIST basic alignment experiment completed at $(date)"


