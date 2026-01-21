#!/bin/bash
#SBATCH --job-name=vision_r50_imnet100_seed
#SBATCH --output=logs/vision_r50_imnet100_seed_%A_%a.out
#SBATCH --error=logs/vision_r50_imnet100_seed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#SBATCH --array=0-1

# ----------------------------------------------------------------------------
# ResNet-50 / ImageNet-100: multi-seed final runs (2 seeds)
# ----------------------------------------------------------------------------

set -euo pipefail

SEEDS=(42 123)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$IDX]}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper (final): ResNet-50/ImageNet-100 seed=${SEED}"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: $OUTPUT_BASE"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python scripts/run_experiment.py \
  --config configs/vision_prune/resnet50_imagenet100_unified.yaml \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"

