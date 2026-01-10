#!/bin/bash
#SBATCH --job-name=single_prune
#SBATCH --output=logs/single_model_%j.out
#SBATCH --error=logs/single_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320GB

# ============================================================================
# SINGLE MODEL PRUNING (Specify config via argument)
# ============================================================================
# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_single_model.sh <config_name>
#
# Usage: sbatch slurm_jobs/run_single_model.sh <config_name>
# 
# Examples:
#   sbatch slurm_jobs/run_single_model.sh mistral7b_pruning
#   sbatch slurm_jobs/run_single_model.sh llama2_7b_pruning
#   sbatch slurm_jobs/run_single_model.sh gemma2b_pruning
#   sbatch slurm_jobs/run_single_model.sh phi3_mini_pruning
#   sbatch slurm_jobs/run_single_model.sh qwen2_7b_pruning
#   sbatch slurm_jobs/run_single_model.sh gpt2_fast_test
#
# Available configs:
#   - mistral7b_pruning     (Mistral-7B)
#   - llama2_7b_pruning     (Llama-2-7B)
#   - gemma2b_pruning       (Gemma-2B, smaller)
#   - phi3_mini_pruning     (Phi-3 Mini, smaller)
#   - qwen2_7b_pruning      (Qwen2-7B)
#   - gpt2_fast_test        (GPT-2, very fast)
#   - llama3_minitron_comparison (Llama-3.1-8B, original)
# ============================================================================

# Get config name from argument, default to llama3 if not provided
CONFIG_NAME=${1:-"llama3_minitron_comparison"}
CONFIG="configs/examples/${CONFIG_NAME}.yaml"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================================"
echo "SINGLE MODEL PRUNING: ${CONFIG_NAME}"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -1 configs/examples/*.yaml | sed 's|configs/examples/||' | sed 's|.yaml||'
    exit 1
fi

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
echo "Running experiment..."
echo "============================================================================"

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --device cuda

echo ""
echo "============================================================================"
echo "${CONFIG_NAME} pruning completed at $(date)"
echo "============================================================================"
