#!/bin/bash
#SBATCH --job-name=test_all_layers
#SBATCH --output=logs/test_all_layers_%j.out
#SBATCH --error=logs/test_all_layers_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=320GB

set -euo pipefail

# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_test_all_layers.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Test: All Layers (MLP + Attention)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Make logs directory if it doesn't exist
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
else
    echo "WARN: HF token file not found at $HF_TOKEN_FILE (set HF_TOKEN env var if needed)" >&2
fi

echo "Testing with ALL layers (MLP + Attention)..."
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_test_all_layers.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
