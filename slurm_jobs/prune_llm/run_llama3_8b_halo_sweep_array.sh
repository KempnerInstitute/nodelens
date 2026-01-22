#!/bin/bash
#SBATCH --job-name=paper_llama3_halo
#SBATCH --output=logs/paper_llama3_halo_%A_%a.out
#SBATCH --error=logs/paper_llama3_halo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-8

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B SWEEP: halo definition sensitivity (K, η) for SCAR-Conn @ 50%
#
# We sweep:
#   K   ∈ {128, 256, 512}   (top-K output dims used in Conn)
#   η   ∈ {  5%,  10%, 20%} (halo fraction among non-supernodes)
#
# Total 9 jobs (array 0-8).
# ----------------------------------------------------------------------------

set -euo pipefail

K_LIST=(128 128 128 256 256 256 512 512 512)
ETA_LIST=(0.05 0.10 0.20 0.05 0.10 0.20 0.05 0.10 0.20)
TAG_LIST=("K128_eta5" "K128_eta10" "K128_eta20" "K256_eta5" "K256_eta10" "K256_eta20" "K512_eta5" "K512_eta10" "K512_eta20")

IDX="${SLURM_ARRAY_TASK_ID}"
K="${K_LIST[$IDX]}"
ETA="${ETA_LIST[$IDX]}"
TAG="${TAG_LIST[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Sweep: LLaMA-3.1-8B halo sensitivity (${TAG})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
echo "Conn top-K: ${K}"
echo "Halo fraction (η): ${ETA}"
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
  name="llama3_8b_paper_results_halo_${TAG}" \
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
  supernode.connectivity_topk="${K}" \
  supernode.halo_fraction="${ETA}" \
  supernode.follower_fraction="${ETA}"

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B halo sensitivity (${TAG}) completed at $(date)"
echo "============================================================================"

