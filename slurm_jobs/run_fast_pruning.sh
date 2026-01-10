#!/bin/bash
#SBATCH --job-name=fast_prune
#SBATCH --output=logs/fast_pruning_%j.out
#SBATCH --error=logs/fast_pruning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=80GB

# ============================================================================
# FAST LLM PRUNING COMPARISON
# ============================================================================
# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_fast_pruning.sh
#
# Quick iteration version for development and testing
# Expected runtime: ~30-60 minutes on H100
# 
# Changes from comprehensive version:
# - 3 sparsity levels (0.3, 0.5, 0.7) instead of 9
# - 1 selection mode (low) instead of 2
# - 4 algorithms instead of 9
# - Dropped slow benchmarks (GSM8k, MBPP, HumanEval)
# - 50 eval samples instead of 100
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================================"
echo "FAST LLM PRUNING COMPARISON"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

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

echo "============================================================================"
echo "FAST MODE CONFIGURATION:"
echo "============================================================================"
echo ""
echo "PRUNING METHODS (4 key methods):"
echo "  - rayleigh_quotient      (Our main alignment method)"
echo "  - scar_loss_proxy        (Gradient-informed)"
echo "  - activation_l2_norm     (Magnitude baseline)"
echo "  - wanda                  (SOTA baseline)"
echo ""
echo "SPARSITY LEVELS: 30%, 50%, 70%"
echo "SELECTION MODE: low only"
echo ""
echo "EVALUATION BENCHMARKS (fast only):"
echo "  - Perplexity, Loss, Bits-per-Byte"
echo "  - MMLU, HellaSwag, ARC-Easy/Challenge"
echo "  - WinoGrande, PIQA, BoolQ, TruthfulQA"
echo ""
echo "SKIPPED (slow generation-based):"
echo "  - GSM8k, MBPP, HumanEval"
echo "============================================================================"
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_fast_pruning.yaml \
    --device cuda

echo ""
echo "============================================================================"
echo "Fast pruning comparison completed at $(date)"
echo "============================================================================"

