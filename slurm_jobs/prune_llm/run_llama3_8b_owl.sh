#!/bin/bash
#SBATCH --job-name=paper_llama3_owl
#SBATCH --output=logs/paper_llama3_owl_%j.out
#SBATCH --error=logs/paper_llama3_owl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

# ============================================================================
# LLaMA-3.1-8B PAPER BASELINE: OWL (Outlier-aware Wanda)
# ============================================================================
# OWL uses non-uniform layer-wise sparsity based on activation outlier ratios.
# Layers with more outliers get lower sparsity (keep more weights).
# Reference: Yin et al. 2024 - "OWL: A Missing Secret Sauce for Pruning LLMs"
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper Baseline: OWL | LLaMA-3.1-8B (4xGPU)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
echo "Output Base: $OUTPUT_BASE"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

# Prefer SLURM_SUBMIT_DIR (repo root) when available.
cd "${SLURM_SUBMIT_DIR:-/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment}"
mkdir -p logs

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace auth/cache
if [[ -z "${HF_HOME:-}" ]]; then
  HF_TOKEN_BASE="${OUTPUT_BASE}"
  if [[ "$(basename "${OUTPUT_BASE}")" == "PAPER" ]]; then
    HF_TOKEN_BASE="$(dirname "${OUTPUT_BASE}")"
  fi

  if [[ -f "${HF_TOKEN_BASE}/huggingface_cache/token" ]]; then
    export HF_HOME="${HF_TOKEN_BASE}/huggingface_cache"
  elif [[ -d /n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache ]]; then
    export HF_HOME="/n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache"
  else
    export HF_HOME="/n/home13/hsafaai/.cache/huggingface"
  fi
fi
HF_TOKEN_FILE="${HF_HOME}/token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
echo "HF_HOME: $HF_HOME"
echo "HF_TOKEN: ${HF_TOKEN:+set}"

# Run OWL structured pruning (channel-wise with outlier-aware sparsity allocation)
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_owl" \
  generate_plots=true \
  pruning_strategies="['owl']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity']" \
  "llm.calibration_num_samples=128" \
  "llm.evaluation_num_samples=128" \
  do_connectivity_pruning=false \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "OWL baseline completed at $(date)"
echo "============================================================================"
