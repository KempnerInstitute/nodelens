#!/bin/bash
# ============================================================================
# SUBMIT FULL VISION PAPER SUITE (MULTI-SEED) into OUTPUT_BASE/PAPER
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_suite_paper_folder_multiseed.sh
# ============================================================================

set -euo pipefail

OUTPUT_BASE_ROOT="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"
OUTPUT_BASE="${OUTPUT_BASE_ROOT}/PAPER"

if [[ "$OUTPUT_BASE" != /* ]]; then
  echo "[error] OUTPUT_BASE must be an absolute path. Got: $OUTPUT_BASE"
  exit 1
fi
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Submitting Vision Paper Suite (PAPER folder, multi-seed)"
echo "=============================================="
echo "OUTPUT_BASE_ROOT: $OUTPUT_BASE_ROOT"
echo "OUTPUT_BASE (runs): $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

JOB_R18=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10_seed_array.sh | awk '{print $4}')
echo "ResNet-18/CIFAR-10 (3 seeds): $JOB_R18"

JOB_VGG=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_vgg16_cifar10_seed_array.sh | awk '{print $4}')
echo "VGG-16-BN/CIFAR-10 (3 seeds): $JOB_VGG"

JOB_MBV2=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_mobilenetv2_cifar10_seed_array.sh | awk '{print $4}')
echo "MobileNetV2/CIFAR-10 (3 seeds): $JOB_MBV2"

JOB_R50=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet50_imagenet100_seed_array.sh | awk '{print $4}')
echo "ResNet-50/ImageNet-100 (2 seeds): $JOB_R50"

echo ""
echo "=============================================="
echo "All PAPER-folder multi-seed jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

