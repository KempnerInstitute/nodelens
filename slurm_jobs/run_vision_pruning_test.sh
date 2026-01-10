#!/bin/bash
#SBATCH --job-name=vision_pruning_test
#SBATCH --output=logs/vision_pruning_test_%j.out
#SBATCH --error=logs/vision_pruning_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
 
set -euo pipefail

# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_vision_pruning_test.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Vision Pruning Test (AlexNet on ImageNet)"
echo "=========================================="
echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-networkAlignmentAnalysis}"
else
    echo "WARN: conda not found; assuming environment already activated." >&2
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_TOKEN_FILE="${HF_HOME}/token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "Running experiment..."
python scripts/run_experiment.py \
    --config configs/examples/vision_pruning_test.yaml \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

