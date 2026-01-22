#!/bin/bash
#SBATCH --job-name=paper_llama3_xfer
#SBATCH --output=logs/paper_llama3_xfer_%A_%a.out
#SBATCH --error=logs/paper_llama3_xfer_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-3

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B: Cross-domain calibration → pruning transfer (SCAR-Conn @ 50%)
#
# Goal: calibrate/score/prune on domain A, evaluate on *fixed* target eval sets
# (WikiText-2 + C4 perplexity) to quantify transfer vs calibration domain shift.
#
# Task mapping:
#   0: wikitext (WikiText-2), n=64
#   1: c4      (C4),         n=64
#   2: code    (CodeSearchNet python), n=64
#   3: arxiv   (scientific_papers/arxiv), n=64
# ----------------------------------------------------------------------------

set -euo pipefail

DATASETS=("wikitext" "c4" "code" "arxiv")
NSAMPLES=(64 64 64 64)
TAGS=("wikitext" "c4" "code" "arxiv")

IDX="${SLURM_ARRAY_TASK_ID}"
DATASET="${DATASETS[$IDX]}"
N="${NSAMPLES[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Sweep: LLaMA-3.1-8B cross-domain transfer (${TAG})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
echo "Calibration dataset: ${DATASET}"
echo "Calibration samples: ${N}"
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

# NOTE: SCAR-Conn depends on directed redundancy + connectivity scoring.
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_xfer_${TAG}" \
  generate_plots=false \
  dataset_name="${DATASET}" \
  alignment_data_num_samples="${N}" \
  scar_num_samples="${N}" \
  pruning_strategies="['supernode_connectivity_score']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity']" \
  do_directed_redundancy=true \
  do_connectivity_pruning=true \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_summary.enabled=false \
  halo_analysis.enabled=false \
  generalized_importance.enabled=false \
  supernode_robustness.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B cross-domain transfer (${TAG}) completed at $(date)"
echo "============================================================================"

