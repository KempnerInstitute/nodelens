#!/bin/bash
# ============================================================================
# SUBMIT FULL VISION PAPER SUITE
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_suite.sh
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
echo "Submitting Vision Paper Suite"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

JOB_R18=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10.sh | awk '{print $4}')
echo "ResNet-18/CIFAR-10: $JOB_R18"

JOB_VGG=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_vgg16_cifar10.sh | awk '{print $4}')
echo "VGG-16-BN/CIFAR-10: $JOB_VGG"

JOB_MBV2=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_mobilenetv2_cifar10.sh | awk '{print $4}')
echo "MobileNetV2/CIFAR-10: $JOB_MBV2"

JOB_R50=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet50_imagenet100.sh | awk '{print $4}')
echo "ResNet-50/ImageNet-100: $JOB_R50"

echo ""
echo "=============================================="
echo "All suite jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

