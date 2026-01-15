#!/bin/bash
#SBATCH --job-name=paper_llama3_noprotect
#SBATCH --output=logs/paper_llama3_noprotect_%j.out
#SBATCH --error=logs/paper_llama3_noprotect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B CONTROL: LP-no-protect + "remove supernodes early" (mode=high)
#
# Produces (at 50%):
# - LP-no-protect:    metric=scar_loss_proxy, mode=low, protect_core=false
# - Remove-core-early metric=scar_loss_proxy, mode=high, protect_core=false
# ----------------------------------------------------------------------------

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper Control: LLaMA-3.1-8B (no-protect LP control)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
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

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_noprotect" \
  generate_plots=false \
  supernode.protect_core=false \
  pruning_strategies="['scar_loss_proxy']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low','high']" \
  do_connectivity_pruning=false \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B no-protect control completed at $(date)"
echo "============================================================================"

