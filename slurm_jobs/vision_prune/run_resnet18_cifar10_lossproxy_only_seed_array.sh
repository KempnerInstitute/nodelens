#!/bin/bash
#SBATCH --job-name=vision_r18_lp_only
#SBATCH --output=logs/vision_r18_lp_only_%A_%a.out
#SBATCH --error=logs/vision_r18_lp_only_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-2

# ----------------------------------------------------------------------------
# ResNet-18 / CIFAR-10: LP-only analysis (no pruning grid)
#
# Purpose: quickly produce results.json with `layer_metrics[*].loss_proxy` so we can
# generate:
#   - drafts/alignment_notes/paper_figures_vision/loss_proxy_depth.pdf
#   - drafts/alignment_notes/paper_figures_vision/lp_prediction_feature_sets.pdf
#
# This does NOT replace the full PAPER pruning suite; it just avoids waiting for
# the full method×ratio grid to finish.
# ----------------------------------------------------------------------------

set -euo pipefail

SEEDS=(42 123 456)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$IDX]}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER}"

echo "============================================================================"
echo "Vision LP-only: ResNet-18/CIFAR-10 seed=${SEED}"
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
  --config configs/vision_prune/resnet18_cifar10_unified.yaml \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE" \
  "pruning.ratios=[]"

echo ""
echo "Done: $(date)"

