#!/bin/bash
# ============================================================================
# SUBMIT ALEXNET / IMAGENET-100 (MULTI-SEED) into OUTPUT_BASE/PAPER
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   export PARTITION="kempner_eng"
#   bash slurm_jobs/vision_prune/submit_alexnet_paper_folder_multiseed.sh
# ============================================================================

set -euo pipefail

OUTPUT_BASE_ROOT="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"
OUTPUT_BASE="${OUTPUT_BASE_ROOT}/PAPER"
PARTITION="${PARTITION:-kempner_eng}"

if [[ "$OUTPUT_BASE" != /* ]]; then
  echo "[error] OUTPUT_BASE must be an absolute path. Got: $OUTPUT_BASE"
  exit 1
fi
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Submitting AlexNet/ImageNet-100 (PAPER folder, multi-seed)"
echo "=============================================="
echo "OUTPUT_BASE_ROOT: $OUTPUT_BASE_ROOT"
echo "OUTPUT_BASE (runs): $OUTPUT_BASE"
echo "PARTITION: $PARTITION"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

JOB_ALEX=$(sbatch -p "$PARTITION" --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_alexnet_imagenet100_seed_array.sh | awk '{print $4}')
echo "AlexNet/ImageNet-100 (3 seeds): $JOB_ALEX"

echo ""
echo "=============================================="
echo "AlexNet/ImageNet-100 jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u $USER"
echo ""
