#!/bin/bash
#SBATCH --job-name=vision_synergy_resnet18
#SBATCH --output=logs/vision_synergy_%j.out
#SBATCH --error=logs/vision_synergy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "Starting vision synergy / redundancy experiment at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"

# Environment setup (conda env: networkAlignmentAnalysis)
module purge
module load cuda/12.2.0-fasrc01
source activate networkAlignmentAnalysis || conda activate networkAlignmentAnalysis || true

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_experiment.py \
    --config configs/projects/vision_synergy.yaml \
    --device cuda

echo "Vision synergy / redundancy experiment completed at $(date)"


