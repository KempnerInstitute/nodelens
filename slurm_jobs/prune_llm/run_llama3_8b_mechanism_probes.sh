#!/bin/bash
#SBATCH --job-name=paper_llama3_mech
#SBATCH --output=logs/paper_llama3_mech_%j.out
#SBATCH --error=logs/paper_llama3_mech_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=240GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#
# ============================================================================
# LLaMA-3.1-8B MECHANISM PROBES (paper figures only)
# ============================================================================
# Purpose:
# - Generate the new mechanistic figures that require running the model:
#   - LP vs magnitude controls (fig_lp_vs_magnitude.png)
#   - Bus concentration (fig_bus_concentration.png)
#   - Read-halo dependence under bus ablation (fig_read_halo_dependence.png)
#   - Conditional halo ablation (fig_halo_conditional_ablation.png)
#
# This job is intentionally lighter than the full paper run:
# - No large benchmark sweeps
# - No structured pruning baseline suite
# - Focus on mechanism-only analyses + paper figures
#
# Output:
#   $OUTPUT_BASE/llama3_8b_paper_results_mechanism_probes_<timestamp>_<jobid>/
#
# ============================================================================

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper: LLaMA-3.1-8B Mechanism Probes (1xGPU)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Start time: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

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

# HuggingFace setup (token + cache)
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

# ---- Mechanism probe knobs (keep runtime reasonable) ----
CAL_N=64
CAL_MAXLEN=512
RH_NUM_TEXTS=3
RH_MAXLEN=256

# Conditional halo ablation: evaluate a subset of layers (stride) for tractability.
COND_LAYER_STRIDE=4
COND_NUM_TEXTS=16
COND_MAXLEN=256

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_mechanism_probes" \
  generate_plots=true \
  alignment_data_num_samples="${CAL_N}" \
  scar_num_samples="${CAL_N}" \
  scar_max_length="${CAL_MAXLEN}" \
  "llm.scar_num_samples=${CAL_N}" \
  "llm.scar_max_length=${CAL_MAXLEN}" \
  "llm.evaluation_metrics=['perplexity']" \
  "llm.evaluation_num_samples=64" \
  do_pruning_experiments=false \
  do_halo_analysis=false \
  do_directed_redundancy=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false \
  "supernode.read_halo_analysis.enabled=true" \
  "supernode.read_halo_analysis.read_halo_fraction=0.10" \
  "supernode.read_halo_analysis.num_texts=${RH_NUM_TEXTS}" \
  "supernode.read_halo_analysis.max_length=${RH_MAXLEN}" \
  "supernode.read_halo_analysis.random_seed=0" \
  "supernode.read_halo_analysis.compute_dependence=true" \
  "supernode.read_halo_analysis.dependence_max_points=20000" \
  "supernode.conditional_halo_ablation.enabled=true" \
  "supernode.conditional_halo_ablation.layer_stride=${COND_LAYER_STRIDE}" \
  "supernode.conditional_halo_ablation.layer_indices=null" \
  "supernode.conditional_halo_ablation.num_texts=${COND_NUM_TEXTS}" \
  "supernode.conditional_halo_ablation.max_length=${COND_MAXLEN}" \
  "supernode.conditional_halo_ablation.match_bins=10" \
  "supernode.conditional_halo_ablation.seed=0"

echo "============================================================================"
echo "Mechanism probes completed at $(date)"
echo "============================================================================"
