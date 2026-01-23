#!/bin/bash
#SBATCH --job-name=vision_unified
#SBATCH --output=logs/vision_unified_%j.out
#SBATCH --error=logs/vision_unified_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --account=kempner_dev

# -----------------------------------------------------------------------------
# Generic vision unified runner (single seed)
# -----------------------------------------------------------------------------
# Usage (example):
#   sbatch -p kempner_eng --export=ALL,SEED=42,CFG=configs/vision_prune/resnet18_cifar100_unified.yaml,OUTPUT_BASE=/.../PAPER run_vision_unified_single.sh

set -euo pipefail

SEED="${SEED:-42}"
CFG="${CFG:?Must set CFG=/abs/or/rel/path/to/config.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER}"
DEVICE="${DEVICE:-cuda}"

echo "============================================================================"
echo "Vision unified run: CFG=${CFG} seed=${SEED}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}  JobID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: ${OUTPUT_BASE}"
echo "============================================================================"

module purge
module load cuda/12.2.0-fasrc01

# Conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate networkAlignmentAnalysis
fi

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python scripts/run_experiment.py \
  --config "${CFG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --base-output-dir "${OUTPUT_BASE}"

echo "Done: $(date)"
