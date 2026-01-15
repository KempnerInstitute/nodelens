#!/bin/bash
# ============================================================================
# SUBMIT VISION PAPER APPENDIX SUITE (robustness + sweeps)
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_appendix.sh
# ============================================================================

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

if [[ "$OUTPUT_BASE" != /* ]]; then
  echo "[error] OUTPUT_BASE must be an absolute path. Got: $OUTPUT_BASE"
  echo "[hint] Use: export OUTPUT_BASE=\"/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red\""
  exit 1
fi
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Submitting Vision Paper Appendix Suite"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

JOB_GAP=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10_gap.sh | awk '{print $4}')
echo "GAP robustness (ResNet-18): $JOB_GAP"

JOB_ABL=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10_ablation.sh | awk '{print $4}')
echo "Ablation (ResNet-18 @ 50%): $JOB_ABL"

JOB_WS=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_weightsweep_resnet18_array.sh | awk '{print $4}')
echo "Weight sweep array (ResNet-18): $JOB_WS"

JOB_DP=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_damage_prediction_resnet18.sh | awk '{print $4}')
echo "Damage prediction eval (ResNet-18): $JOB_DP"

echo ""
echo "=============================================="
echo "Appendix jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

