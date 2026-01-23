#!/bin/bash
#SBATCH --job-name=paper_llama3_readhalo
#SBATCH --output=logs/paper_llama3_readhalo_%A_%a.out
#SBATCH --error=logs/paper_llama3_readhalo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --mem=256GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-1
#
# ----------------------------------------------------------------------------
# LLaMA-3.1-8B: read-halo diagnostic (analysis-only)
#
# Runs 2 lightweight jobs (array):
#   0: supernodes by scar_loss_proxy      (paper-aligned)
#   1: supernodes by scar_activation_power (sanity / comparison)
#
# This DOES NOT change the pruning method; it only records an additional analysis
# block ("next_layer_read_halo") inside supernode connection analysis outputs.
# ----------------------------------------------------------------------------

set -euo pipefail

METRICS=("scar_loss_proxy" "scar_activation_power")
TAGS=("lp" "act")

IDX="${SLURM_ARRAY_TASK_ID}"
SUP_METRIC="${METRICS[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Diagnostic: LLaMA-3.1-8B read-halo (${TAG})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Default to PAPER folder (fresh, isolated artifacts).
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
echo "Output Base: $OUTPUT_BASE"
echo "Supernode metric: ${SUP_METRIC}"
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
if [[ -z "${HF_HOME:-}" ]]; then
  # If running in OUTPUT_BASE/PAPER, shared cache/token typically lives in OUTPUT_BASE_ROOT/huggingface_cache.
  OUTPUT_BASE_ROOT="${OUTPUT_BASE}"
  if [[ "${OUTPUT_BASE_ROOT}" == */PAPER ]]; then
    OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT%/PAPER}"
  fi

  if [[ -f "${OUTPUT_BASE_ROOT}/huggingface_cache/token" ]]; then
    export HF_HOME="${OUTPUT_BASE_ROOT}/huggingface_cache"
  elif [[ -f "${OUTPUT_BASE}/huggingface_cache/token" ]]; then
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

# Keep this run lightweight:
# - fewer SCAR samples
# - no pruning sweeps
# - no downstream benchmark evaluation
# - only adds the read-halo diagnostic block + small plots under plots/read_halo/
N=16
MAXLEN=256

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_read_halo_${TAG}" \
  generate_plots=false \
  alignment_data_num_samples="${N}" \
  scar_num_samples="${N}" \
  scar_max_length="${MAXLEN}" \
  "llm.scar_num_samples=${N}" \
  "llm.scar_max_length=${MAXLEN}" \
  "llm.evaluate_perplexity=false" \
  "llm.evaluation_metrics=[]" \
  do_pruning_experiments=false \
  do_directed_redundancy=false \
  do_connectivity_pruning=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  "supernode.score_metric=${SUP_METRIC}" \
  "supernode.read_halo.enabled=true" \
  "supernode.read_halo.read_halo_fraction=0.10" \
  "supernode.read_halo.num_texts=4" \
  "supernode.read_halo.max_length=${MAXLEN}" \
  "supernode.read_halo.random_seed=0"

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B read-halo (${TAG}) completed at $(date)"
echo "============================================================================"

