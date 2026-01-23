#!/bin/bash
#SBATCH --job-name=paper_scar_ablations
#SBATCH --output=logs/paper_scar_ablations_%j.out
#SBATCH --error=logs/paper_scar_ablations_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=kempner_eng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320GB
#SBATCH --gres=gpu:4
#SBATCH --account=kempner_dev

# SCAR Ablations v2: Using config-based experiment runner
# Tests:
# 1. Standard SCAR (baseline)
# 2. Random supernode protection (ablation)
# 3. SCAR-optimal (learned weights)

set -euo pipefail

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
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
if [[ -z "${HF_HOME:-}" ]]; then
  HF_TOKEN_BASE="$(dirname "${OUTPUT_BASE}")"
  if [[ -f "${HF_TOKEN_BASE}/huggingface_cache/token" ]]; then
    export HF_HOME="${HF_TOKEN_BASE}/huggingface_cache"
  else
    export HF_HOME="/n/home13/hsafaai/.cache/huggingface"
  fi
fi
HF_TOKEN_FILE="${HF_HOME}/token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

echo "=========================================="
echo "SCAR Ablation Experiments v2"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Run main SCAR experiment with ablation flags
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_scar_ablations_v2" \
  generate_plots=true \
  pruning_strategies="['scar_loss_proxy', 'supernode_protection_score', 'supernode_connectivity_score']" \
  pruning_amounts="[0.3, 0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity','accuracy_mmlu','accuracy_hellaswag','accuracy_piqa','accuracy_boolq']" \
  "llm.calibration_num_samples=64" \
  "llm.evaluation_num_samples=64" \
  do_connectivity_pruning=true \
  do_directed_redundancy=true \
  do_halo_analysis=true \
  do_scar_optimal=true \
  do_random_supernode_ablation=true \
  supernode.rho=0.01 \
  supernode.eta=0.10

echo "=========================================="
echo "Completed at $(date)"
echo "=========================================="
