#!/bin/bash
#SBATCH --job-name=paper_llama3_all_baselines
#SBATCH --output=logs/paper_llama3_all_baselines_%j.out
#SBATCH --error=logs/paper_llama3_all_baselines_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ============================================================================
# LLaMA-3.1-8B ALL STRUCTURED PRUNING BASELINES
# ============================================================================
# Compares SCAR against: Wanda, SparseGPT, OWL, LLM-Pruner, FLAP, RIA, SlimLLM
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "SCAR vs All Baselines: Llama-3.1-8B (4xGPU)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis
# Robustly locate the `alignment/` repo even if `sbatch` was invoked from the monorepo root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
  cd "${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/alignment/scripts" ]]; then
  cd "${SLURM_SUBMIT_DIR}/alignment"
else
  cd "/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment"
fi
mkdir -p logs
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace setup
if [[ -z "${HF_HOME:-}" ]]; then
  HF_TOKEN_BASE="$(dirname "${OUTPUT_BASE}")"
  if [[ -f "${HF_TOKEN_BASE}/huggingface_cache/token" ]]; then
    export HF_HOME="${HF_TOKEN_BASE}/huggingface_cache"
  else
    export HF_HOME="/n/home13/hsafaai/.cache/huggingface"
  fi
fi
HF_TOKEN_FILE="${HF_HOME}/token"
[[ -f "$HF_TOKEN_FILE" ]] && export HF_TOKEN="$(cat "$HF_TOKEN_FILE")" && export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# Run experiment with ALL structured pruning baselines
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_all_baselines" \
  generate_plots=true \
  pruning_strategies="['scar_loss_proxy', 'wanda', 'sparsegpt', 'owl', 'llm_pruner', 'flap', 'ria', 'slimllm', 'weight_magnitude', 'random']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity','accuracy_mmlu','accuracy_hellaswag','accuracy_piqa','accuracy_boolq']" \
  "llm.calibration_num_samples=128" \
  "llm.evaluation_num_samples=128" \
  do_connectivity_pruning=true \
  do_directed_redundancy=false \
  do_halo_analysis=false

echo "============================================================================"
echo "All baselines completed at $(date)"
echo "============================================================================"
