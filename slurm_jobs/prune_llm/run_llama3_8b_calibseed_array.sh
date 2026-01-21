#!/bin/bash
#SBATCH --job-name=paper_llama3_calibseed
#SBATCH --output=logs/paper_llama3_calibseed_%A_%a.out
#SBATCH --error=logs/paper_llama3_calibseed_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev
#SBATCH --array=0-4

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B: Within-domain supernode stability across calibration draws
#
# We keep the dataset fixed (WikiText) and change the *calibration draw* by
# deterministically shuffling the calibration text pool with different seeds.
#
# This is the key "final-run" robustness check for supernode identity stability
# *within* a dataset.
#
# Task mapping (5 calibration-draw seeds):
#   0: seed 42
#   1: seed 123
#   2: seed 456
#   3: seed 789
#   4: seed 1000
#
# Outputs are used by paper artifact collection to compute overlap statistics.
# ----------------------------------------------------------------------------

set -euo pipefail

SEEDS=(42 123 456 789 1000)
TAGS=("s42" "s123" "s456" "s789" "s1000")

IDX="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Sweep: LLaMA-3.1-8B within-domain stability (${TAG})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
echo "Calibration dataset: wikitext"
echo "Calibration shuffle seed: ${SEED}"
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

# We only need supernode robustness results (LP supernode sets) for this sweep.
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_calibseed_${TAG}" \
  generate_plots=false \
  dataset_name="wikitext" \
  alignment_data_num_samples=512 \
  scar_num_samples=64 \
  do_pruning_experiments=false \
  do_directed_redundancy=false \
  do_connectivity_pruning=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_summary.enabled=false \
  halo_analysis.enabled=false \
  generalized_importance.enabled=false \
  "llm.evaluation_metrics=[]" \
  "llm.shuffle_calibration_texts=true" \
  "llm.calibration_seed=${SEED}" \
  supernode_robustness.enabled=true \
  "supernode_robustness.metrics=['scar_loss_proxy']" \
  supernode_robustness.num_bootstrap_samples=1 \
  supernode_robustness.max_samples=256

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B within-domain stability (${TAG}) completed at $(date)"
echo "============================================================================"

