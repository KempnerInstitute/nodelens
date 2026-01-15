#!/bin/bash
#SBATCH --job-name=baseline_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_test_%j.out
#SBATCH --error=logs/baseline_test_%j.err

# Quick test for Wanda/SparseGPT integration
# Expected runtime: ~30-60 minutes

set -euo pipefail

# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_baseline_test.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"

echo "=========================================="
echo "Baseline Pruning Test (Wanda + SparseGPT)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-networkAlignmentAnalysis}"
else
    echo "WARN: conda not found; assuming environment already activated." >&2
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HF_TOKEN_FILE="${HF_HOME}/token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
else
    echo "WARN: HF token file not found at $HF_TOKEN_FILE (set HF_TOKEN env var if needed)" >&2
fi

# Create logs directory
mkdir -p logs

# Run experiment
echo "Running baseline test..."
python scripts/run_experiment.py \
    --config configs/examples/llama3_baseline_test.yaml

echo ""
echo "=========================================="
echo "Baseline test completed at $(date)"
echo "=========================================="

