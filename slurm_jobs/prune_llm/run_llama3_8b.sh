#!/bin/bash
#SBATCH --job-name=paper_llama3_8b
#SBATCH --output=logs/paper_llama3_8b_%j.out
#SBATCH --error=logs/paper_llama3_8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

# ============================================================================
# LLAMA-3.1-8B PAPER RESULTS
# ============================================================================
# Full SCAR analysis including:
# - Supernode distribution & robustness
# - Halo redundancy analysis
# - Cross-layer importance
# - Within-layer importance
# - All pruning methods + SOTA baselines (Wanda, SparseGPT)
# - Full benchmark evaluation
#
# Expected runtime: ~6-8 hours on H100
#
# Output Directory Structure:
#   /n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/
#       llama3_8b_paper_results_{timestamp}_{SLURM_JOB_ID}/
#           results/      - JSON results files
#           logs/         - experiment.log
#           figures/      - All visualizations
#           checkpoints/  - Model checkpoints
#           analysis/     - Post-analysis outputs
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper: LLaMA-3.1-8B"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
echo "Output Base: $OUTPUT_BASE"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

# Prefer SLURM_SUBMIT_DIR (repo root) when available.
cd "${SLURM_SUBMIT_DIR:-/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment}"

# Create local logs directory for SLURM output files
mkdir -p logs

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HuggingFace auth/cache:
# - Respect HF_HOME if already set (e.g. exported from submission script).
# - Else, if you ran `hf auth login` with HF_HOME under OUTPUT_BASE, prefer that token/cache.
# - Else fall back to scratch cache, then ~/.cache.
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -f "${OUTPUT_BASE}/huggingface_cache/token" ]]; then
    export HF_HOME="${OUTPUT_BASE}/huggingface_cache"
  elif [[ -d /n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache ]]; then
    export HF_HOME="/n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache"
  else
    export HF_HOME="/n/home13/hsafaai/.cache/huggingface"
  fi
fi
HF_TOKEN_FILE="${HF_HOME}/token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF token file not found at $HF_TOKEN_FILE (set HF_TOKEN env var if needed)" >&2
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
echo "HF_HOME: $HF_HOME"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: set"
else
  echo "HF_TOKEN: unset"
fi

echo ""
echo "Running LLaMA-3.1-8B full paper analysis..."
echo ""

python scripts/run_experiment.py \
    --config configs/prune_llm/llama3_8b_full.yaml \
    --device cuda \
    --base-output-dir "$OUTPUT_BASE"

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B completed at $(date)"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo "Look for directory: llama3_8b_paper_results_*_$SLURM_JOB_ID"
