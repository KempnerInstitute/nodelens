#!/bin/bash
# ============================================================================
# SUBMIT CIFAR-100 COMPARISON RUNS (MULTI-SEED) into OUTPUT_BASE/PAPER
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_cifar100_paper_folder_multiseed.sh
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
echo "Submitting CIFAR-100 comparison runs (PAPER folder, multi-seed)"
echo "=============================================="
echo "OUTPUT_BASE_ROOT: $OUTPUT_BASE_ROOT"
echo "OUTPUT_BASE (runs): $OUTPUT_BASE"
echo "PARTITION: $PARTITION"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

JOB_R18=$(sbatch -p "$PARTITION" --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar100_seed_array.sh | awk '{print $4}')
echo "ResNet-18/CIFAR-100 (3 seeds): $JOB_R18"

echo ""
echo "=============================================="
echo "CIFAR-100 jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u $USER"
echo ""

