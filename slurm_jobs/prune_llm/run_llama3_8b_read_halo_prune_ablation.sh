#!/bin/bash
#SBATCH --job-name=paper_llama3_readhalo_prune
#SBATCH --output=logs/paper_llama3_readhalo_prune_%j.out
#SBATCH --error=logs/paper_llama3_readhalo_prune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#
# ----------------------------------------------------------------------------
# LLaMA-3.1-8B: pruning ablation to test read-halo modifier
# ----------------------------------------------------------------------------

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper Ablation: LLaMA-3.1-8B read-halo pruning"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Default to PAPER folder (fresh, isolated artifacts).
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/PAPER}"
echo "Output Base: $OUTPUT_BASE"
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

# HuggingFace auth/cache:
if [[ -z "${HF_HOME:-}" ]]; then
  OUTPUT_BASE_ROOT="${OUTPUT_BASE}"
  if [[ "${OUTPUT_BASE_ROOT}" == */PAPER ]]; then
    OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT%/PAPER}"
  fi
  if [[ -f "${OUTPUT_BASE_ROOT}/huggingface_cache/token" ]]; then
    export HF_HOME="${OUTPUT_BASE_ROOT}/huggingface_cache"
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

# Keep this run reasonably light.
CAL_N=32
CAL_MAXLEN=512

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_read_halo_prune_ablation" \
  generate_plots=false \
  alignment_data_num_samples="${CAL_N}" \
  scar_num_samples="${CAL_N}" \
  scar_max_length="${CAL_MAXLEN}" \
  "llm.scar_num_samples=${CAL_N}" \
  "llm.scar_max_length=${CAL_MAXLEN}" \
  "llm.evaluation_metrics=['perplexity']" \
  "llm.evaluation_num_samples=64" \
  "llm.perplexity_protocol=legacy" \
  pruning_strategies="['scar_loss_proxy','supernode_protection_score','supernode_connectivity_score','supernode_read_halo_score','wanda']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  do_connectivity_pruning=true \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  "supernode.read_halo_pruning.enabled=true" \
  "supernode.read_halo_pruning.read_halo_fraction=0.10" \
  "supernode.read_halo_pruning.rank_power=8.0" \
  "supernode.read_halo_pruning.protection_floor=0.2" \
  supernode.protect_core=true \
  supernode.protect_core_metrics="['scar_loss_proxy','supernode_protection_score','supernode_connectivity_score','supernode_read_halo_score']"

echo ""
echo "============================================================================"
echo "Completed at $(date)"
echo "============================================================================"

