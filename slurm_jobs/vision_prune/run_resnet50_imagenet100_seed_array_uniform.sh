#!/bin/bash
#SBATCH --job-name=vision_r50_imnet100_uniform_seed
#SBATCH --output=logs/vision_r50_imnet100_uniform_seed_%A_%a.out
#SBATCH --error=logs/vision_r50_imnet100_uniform_seed_%A_%a.err
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
# ResNet-50 / ImageNet-100: uniform distribution + per-layer cap (paper rerun)
# ----------------------------------------------------------------------------

set -euo pipefail

SEEDS=(42 123)
IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$IDX]}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: ResNet-50/ImageNet-100 (UNIFORM) seed=${SEED}"
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

# ----------------------------------------------------------------------------
# ImageNet-100 data prep (symlink subset from ImageNet-1k if needed)
# ----------------------------------------------------------------------------
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

python scripts/run_experiment.py \
  --config configs/vision_prune/resnet50_imagenet100_unified_paper_uniform.yaml \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"

