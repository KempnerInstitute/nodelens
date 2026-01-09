#!/bin/bash
#SBATCH --job-name=paper_llama3_calib
#SBATCH --output=logs/paper_llama3_calib_%A_%a.out
#SBATCH --error=logs/paper_llama3_calib_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-4

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B SWEEP: calibration sensitivity for SCAR-Conn @ 50% sparsity
#
# Task mapping:
#   0: wikitext, n=128
#   1: wikitext, n=64
#   2: wikitext, n=32
#   3: c4,      n=128
#   4: mixed_wikitext_c4, n=128
#
# Notes:
# - We restrict pruning to SCAR-Conn at 50% and evaluate perplexity only (fast).
# ----------------------------------------------------------------------------

set -euo pipefail

DATASETS=("wikitext" "wikitext" "wikitext" "c4" "mixed_wikitext_c4")
NSAMPLES=(128 64 32 128 128)
TAGS=("wikitext_128" "wikitext_64" "wikitext_32" "c4_128" "mixed_128")

IDX="${SLURM_ARRAY_TASK_ID}"
DATASET="${DATASETS[$IDX]}"
N="${NSAMPLES[$IDX]}"
TAG="${TAGS[$IDX]}"

echo "============================================================================"
echo "SCAR Paper Sweep: LLaMA-3.1-8B calibration sensitivity (${TAG})"
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

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_full.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="llama3_8b_paper_results_calib_${TAG}" \
  generate_plots=false \
  dataset_name="${DATASET}" \
  alignment_data_num_samples="${N}" \
  scar_num_samples="${N}" \
  pruning_strategies="['supernode_connectivity_score']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  "llm.evaluation_metrics=['perplexity']" \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B calibration sweep (${TAG}) completed at $(date)"
echo "============================================================================"

