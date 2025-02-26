#!/bin/bash
#SBATCH --job-name=alignment_experiment
#SBATCH --account=kempner_dev
#SBATCH --output=/n/netscratch/kempner_dev/hsafaai/results/alignment_experiment/%j.out
#SBATCH --error=/n/netscratch/kempner_dev/hsafaai/results/alignment_experiment/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=1200GB
#SBATCH --partition=kempner_h100

# ================================================================
# This script runs the alignment experiment using the specified
# configuration file.
# ================================================================

# Load necessary modules
module purge
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Activate the conda environment
conda activate /n/home13/hsafaai/.conda/envs/networkAlignmentAnalysis

# Define the Python interpreter path from the activated environment
PYTHON=/n/home13/hsafaai/.conda/envs/networkAlignmentAnalysis/bin/python

echo "Starting alignment experiment at $(date)"
start_time=$(date +%s)

# Execute the alignment experiment
srun --cpus-per-task=${SLURM_CPUS_PER_TASK} --kill-on-bad-exit \
  ${PYTHON} -u src/alignment/experiments/alignment_experiments.py \
  --config .vscode/config_alignment_stats_v2.yaml

end_time=$(date +%s)
echo "Alignment experiment finished at $(date)"
echo "Total duration: $((end_time - start_time)) seconds."