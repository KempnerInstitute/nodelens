#!/bin/bash
#SBATCH --job-name=paper_llama3_full_baselines
#SBATCH --output=logs/paper_llama3_full_baselines_%j.out
#SBATCH --error=logs/paper_llama3_full_baselines_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

echo "============================================================================"
echo "SCAR Full Baselines: Llama-3.1-8B (4xGPU)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis
cd "${SLURM_SUBMIT_DIR:-/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment}"
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

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_full_baselines" \
  generate_plots=true \
  pruning_strategies="['scar_loss_proxy', 'wanda', 'sparsegpt', 'owl', 'llm_pruner', 'weight_magnitude']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity']" \
  "llm.calibration_num_samples=128" \
  "llm.evaluation_num_samples=128"

echo "Full baselines completed at $(date)"
