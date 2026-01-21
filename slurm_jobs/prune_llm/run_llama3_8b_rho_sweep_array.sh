#!/bin/bash
#SBATCH --job-name=paper_llama3_rho
#SBATCH --output=logs/paper_llama3_rho_%A_%a.out
#SBATCH --error=logs/paper_llama3_rho_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#SBATCH --array=0-3

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B SWEEP: supernode threshold sensitivity (ρ) for SCAR-Conn @ 50%
#
# Task mapping:
#   0: ρ = 0.5%
#   1: ρ = 1.0% (default)
#   2: ρ = 2.0%
#   3: ρ = 5.0%
# ----------------------------------------------------------------------------

set -euo pipefail

RHOS=(0.005 0.01 0.02 0.05)
TAGS=("rho_0p5" "rho_1p0" "rho_2p0" "rho_5p0")

IDX="${SLURM_ARRAY_TASK_ID}"
RHO="${RHOS[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Sweep: LLaMA-3.1-8B ρ-sensitivity (${TAG})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
echo "Supernode fraction (ρ): ${RHO}"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd "${SLURM_SUBMIT_DIR:-/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment}"
mkdir -p logs

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

# NOTE: SCAR-Conn depends on directed redundancy + connectivity scoring.
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_rho_${TAG}" \
  generate_plots=false \
  dataset_name="wikitext" \
  alignment_data_num_samples=64 \
  scar_num_samples=64 \
  do_directed_redundancy=true \
  do_connectivity_pruning=true \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_summary.enabled=false \
  halo_analysis.enabled=false \
  generalized_importance.enabled=false \
  supernode_robustness.enabled=false \
  "llm.evaluation_metrics=['perplexity']" \
  pruning_strategies="['supernode_connectivity_score']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  supernode.core_fraction="${RHO}"

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B ρ-sensitivity (${TAG}) completed at $(date)"
echo "============================================================================"

