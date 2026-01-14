#!/bin/bash
#SBATCH --job-name=vision_paper_all
#SBATCH --output=logs/vision_paper_all_%A_%a.out
#SBATCH --error=logs/vision_paper_all_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#
# One array job that runs the full vision paper suite + appendix, throttled to
# at most 16 concurrent tasks (== 16 GPUs if each task requests 1 GPU).
# -----------------------------------------------------------------------------
# Task map:
#   0  resnet18_cifar10_cluster_analysis
#   1  vgg16_cifar10_cluster_analysis
#   2  mobilenetv2_cifar10_cluster_analysis
#   3  resnet50_imagenet100_cluster_analysis
#   4  GAP robustness (resnet18, activation_samples=gap)
#   5  Ablation (resnet18 @ 50%: cluster_aware variants + composite)
#   6-20  Weight sweep (15 tasks): gamma∈{0.10,0.30,0.50} × lambda∈{0.00,0.25,0.50,0.75,1.00}
#         Each sweep run prunes across multiple sparsity ratios so the per-run figures show pruning effects.
#
# Submit via: slurm_jobs/vision_prune/submit_all_array.sh
# -----------------------------------------------------------------------------

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: ALL runs (single SLURM array, max 16 GPUs)"
echo "============================================================================"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID:-N/A}   Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

TASK="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

run_py() {
  echo ""
  echo "$ $*"
  python "$@"
}

prepare_imagenet100() {
  # Robust ImageNet-100 subset prep (safe with set -o pipefail)
  IMAGENET1K_ROOT="${IMAGENET1K_ROOT:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k}"
  IMAGENET100_ROOT="${IMAGENET100_ROOT:-$PWD/data/imagenet100}"
  IMAGENET100_NCLASSES="${IMAGENET100_NCLASSES:-100}"

  if [ ! -d "${IMAGENET1K_ROOT}/train" ] || [ ! -d "${IMAGENET1K_ROOT}/val" ]; then
    echo "[error] IMAGENET1K_ROOT does not look like ImageFolder (missing train/val): ${IMAGENET1K_ROOT}"
    exit 2
  fi

  need_prepare=0
  if [ ! -d "${IMAGENET100_ROOT}/train" ] || [ ! -d "${IMAGENET100_ROOT}/val" ]; then
    need_prepare=1
  else
    n_train=$(find -L "${IMAGENET100_ROOT}/train" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)
    n_val=$(find -L "${IMAGENET100_ROOT}/val" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)
    if [ "${n_train}" -lt 1 ] || [ "${n_val}" -lt 1 ]; then
      need_prepare=1
    fi
  fi

  if [ "${need_prepare}" -eq 1 ]; then
    echo "[info] Preparing ImageNet-100 subset under: ${IMAGENET100_ROOT}"
    rm -rf "${IMAGENET100_ROOT}/train" "${IMAGENET100_ROOT}/val"
    mkdir -p "${IMAGENET100_ROOT}/train" "${IMAGENET100_ROOT}/val"

    find "${IMAGENET1K_ROOT}/train" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort > "${IMAGENET100_ROOT}/classes_all.txt"
    head -n "${IMAGENET100_NCLASSES}" "${IMAGENET100_ROOT}/classes_all.txt" > "${IMAGENET100_ROOT}/classes.txt"
    rm -f "${IMAGENET100_ROOT}/classes_all.txt"

    while read -r syn; do
      ln -sfn "${IMAGENET1K_ROOT}/train/${syn}" "${IMAGENET100_ROOT}/train/${syn}"
      ln -sfn "${IMAGENET1K_ROOT}/val/${syn}" "${IMAGENET100_ROOT}/val/${syn}"
    done < "${IMAGENET100_ROOT}/classes.txt"

    n_train=$(find -L "${IMAGENET100_ROOT}/train" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)
    n_val=$(find -L "${IMAGENET100_ROOT}/val" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l || true)
    echo "[info] ImageNet-100 class dirs: train=${n_train} val=${n_val}"
    if [ "${n_train}" -lt 1 ] || [ "${n_val}" -lt 1 ]; then
      echo "[error] ImageNet-100 subset prep failed: no class dirs found under ${IMAGENET100_ROOT}/{train,val}"
      exit 3
    fi
  fi
}

case "${TASK}" in
  0)
    echo "[task 0] ResNet-18 / CIFAR-10"
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/resnet18_cifar10_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE"
    ;;
  1)
    echo "[task 1] VGG-16-BN / CIFAR-10"
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/vgg16_cifar10_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE"
    ;;
  2)
    echo "[task 2] MobileNetV2 / CIFAR-10"
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/mobilenetv2_cifar10_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE"
    ;;
  3)
    echo "[task 3] ResNet-50 / ImageNet-100"
    prepare_imagenet100
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/resnet50_imagenet100_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE"
    ;;
  4)
    echo "[task 4] GAP robustness (ResNet-18, activation_samples=gap)"
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/resnet18_cifar10_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE" \
      name="resnet18_cifar10_cluster_analysis_gap" \
      metrics.activation_samples="gap" \
      pruning_amounts="[]"
    ;;
  5)
    echo "[task 5] Ablation (ResNet-18 @ 50%: cluster_aware variants + composite)"
    run_py scripts/run_experiment.py \
      --config configs/vision_prune/resnet18_cifar10_unified.yaml \
      --device cuda \
      --base-output-dir "$OUTPUT_BASE" \
      name="resnet18_cifar10_cluster_analysis_ablation" \
      pruning_amounts="[0.5]" \
      pruning_distribution="global_threshold" \
      pruning_strategies="['cluster_aware','cluster_aware_no_halo','cluster_aware_no_constraints','composite']"
    ;;
  *)
    # Weight sweep tasks 6-20 (15 tasks)
    if [ "${TASK}" -ge 6 ] && [ "${TASK}" -le 20 ]; then
      SWEEP_IDX=$((TASK - 6))
      GAMMAS=(0.10 0.30 0.50)
      LAMBDAS=(0.00 0.25 0.50 0.75 1.00)
      GI=$((SWEEP_IDX / ${#LAMBDAS[@]}))
      LI=$((SWEEP_IDX % ${#LAMBDAS[@]}))
      GAMMA="${GAMMAS[$GI]}"
      LAMBDA="${LAMBDAS[$LI]}"
      echo "[task ${TASK}] Weight sweep (ResNet-18, multi-sparsity): gamma=${GAMMA}, lambda_halo=${LAMBDA}"
      run_py scripts/run_experiment.py \
        --config configs/vision_prune/resnet18_cifar10_unified.yaml \
        --device cuda \
        --base-output-dir "$OUTPUT_BASE" \
        name="resnet18_cifar10_weightsweep_g${GAMMA}_l${LAMBDA}" \
        pruning_amounts="[0.1,0.3,0.5,0.7,0.8,0.9]" \
        pruning_distribution="global_threshold" \
        pruning_strategies="['cluster_aware']" \
        pruning.cluster_aware.gamma="${GAMMA}" \
        pruning.cluster_aware.lambda_halo="${LAMBDA}"
    else
      echo "[error] Unknown task id: ${TASK}"
      exit 2
    fi
    ;;
esac

echo ""
echo "Done: $(date)"

