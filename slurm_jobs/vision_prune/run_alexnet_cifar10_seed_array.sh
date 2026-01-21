#!/bin/bash
#SBATCH --job-name=vision_alexnet_cifar10_seed
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:30:00
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/slurm_jobs/vision_prune/logs/%x_%A_%a.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/slurm_jobs/vision_prune/logs/%x_%A_%a.err
#SBATCH --array=0-2

# ============================================================================
# AlexNet / CIFAR-10 multi-seed experiment
# ============================================================================

set -euo pipefail

# Activate environment
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

# Seed from array index
SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Output base (allow override from environment)
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

echo "=== AlexNet / CIFAR-10 seed=${SEED} ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "OUTPUT_BASE=$OUTPUT_BASE"

python scripts/run_experiment.py \
    --config configs/vision_prune/alexnet_cifar10_unified.yaml \
    --output-dir "${OUTPUT_BASE}/PAPER" \
    --experiment.seed "$SEED" \
    --job-id "$SLURM_JOB_ID"

echo "=== Done ==="
