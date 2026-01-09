#!/bin/bash
#SBATCH --job-name=paper_llama3_posred
#SBATCH --output=logs/paper_llama3_posred_%A_%a.out
#SBATCH --error=logs/paper_llama3_posred_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-1

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B ABLATION: positive-only redundancy vs rho^2 redundancy
#
# Task 0: positive_redundancy=false (rho^2 counts anti-correlation as redundancy)
# Task 1: positive_redundancy=true  (rho^+ only; anti-correlation NOT redundant)
# ----------------------------------------------------------------------------

set -euo pipefail

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  POS_RED="false"
  TAG="rho2"
else
  POS_RED="true"
  TAG="posonly"
fi

echo "============================================================================"
echo "SCAR Paper Ablation: LLaMA-3.1-8B (positive redundancy = ${POS_RED})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
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
  name="llama3_8b_paper_results_posred_${TAG}" \
  generate_plots=false \
  supernode.positive_redundancy="${POS_RED}" \
  supernode.protect_core=true \
  "supernode.protect_core_metrics=['supernode_connectivity_score']" \
  pruning_strategies="['supernode_connectivity_score']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B pos-redundancy ablation (${TAG}) completed at $(date)"
echo "============================================================================"

