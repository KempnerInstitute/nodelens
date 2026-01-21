#!/bin/bash
#SBATCH --job-name=vision_mbv2_cifar10_seed
#SBATCH --output=logs/vision_mbv2_cifar10_seed_%A_%a.out
#SBATCH --error=logs/vision_mbv2_cifar10_seed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:30:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-2

# ----------------------------------------------------------------------------
# MobileNetV2 / CIFAR-10: multi-seed final runs (3 seeds)
# ----------------------------------------------------------------------------

set -euo pipefail

SEEDS=(42 123 456)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$IDX]}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper (final): MobileNetV2/CIFAR-10 seed=${SEED}"
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
  --config configs/vision_prune/mobilenetv2_cifar10_unified.yaml \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"

