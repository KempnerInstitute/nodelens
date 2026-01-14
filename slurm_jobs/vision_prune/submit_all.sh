#!/bin/bash
# ============================================================================
# SUBMIT FULL VISION PAPER: SUITE + APPENDIX + (DEPENDENT) ARTIFACT BUILD JOB
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_all.sh
#
# This submits:
#   - main suite jobs (4 models)
#   - appendix jobs (GAP, ablation, weight sweep array, damage prediction)
#   - a final build job that runs build_all_artifacts.py after all above succeed
# ============================================================================

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

# Guardrail: avoid accidentally writing into the repo via a relative placeholder.
if [[ "$OUTPUT_BASE" != /* ]]; then
  echo "[error] OUTPUT_BASE must be an absolute path. Got: $OUTPUT_BASE"
  echo "[hint] Use: export OUTPUT_BASE=\"/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red\""
  exit 1
fi
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Submitting Vision Paper: ALL jobs"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

# ----------------------------
# Main suite
# ----------------------------
JOB_R18=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10.sh | awk '{print $4}')
echo "ResNet-18/CIFAR-10: $JOB_R18"

JOB_VGG=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_vgg16_cifar10.sh | awk '{print $4}')
echo "VGG-16-BN/CIFAR-10: $JOB_VGG"

JOB_MBV2=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_mobilenetv2_cifar10.sh | awk '{print $4}')
echo "MobileNetV2/CIFAR-10: $JOB_MBV2"

JOB_R50=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet50_imagenet100.sh | awk '{print $4}')
echo "ResNet-50/ImageNet-100: $JOB_R50"

# ----------------------------
# Appendix / robustness
# ----------------------------
JOB_GAP=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10_gap.sh | awk '{print $4}')
echo "GAP robustness (ResNet-18): $JOB_GAP"

JOB_ABL=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_resnet18_cifar10_ablation.sh | awk '{print $4}')
echo "Ablation (ResNet-18 @ 50%): $JOB_ABL"

JOB_WS=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_weightsweep_resnet18_array.sh | awk '{print $4}')
echo "Weight sweep array (ResNet-18): $JOB_WS"

# Damage prediction should wait for the main ResNet-18 run (needs its checkpoint/results).
JOB_DP=$(sbatch --dependency=afterok:${JOB_R18} --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_damage_prediction_resnet18.sh | awk '{print $4}')
echo "Damage prediction eval (ResNet-18, afterok:$JOB_R18): $JOB_DP"

# ----------------------------
# Final artifact build job (depends on all above)
# ----------------------------
DEP="afterany:${JOB_R18}:${JOB_VGG}:${JOB_MBV2}:${JOB_R50}:${JOB_GAP}:${JOB_ABL}:${JOB_WS}:${JOB_DP}"
JOB_BUILD=$(sbatch --dependency=$DEP --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/build_artifacts.sh | awk '{print $4}')
echo "Build all artifacts (afterany:all): $JOB_BUILD"

echo ""
echo "=============================================="
echo "All jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

