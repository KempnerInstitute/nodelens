#!/bin/bash
#SBATCH --job-name=paper_llama3_protect_base
#SBATCH --output=logs/paper_llama3_protect_base_%j.out
#SBATCH --error=logs/paper_llama3_protect_base_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B CONTROL: Protect+Baseline variants
#
# Produces (at 50%):
# - Protect+Wanda:       metric=wanda, protect_core_metrics includes wanda
# - Protect+Magnitude:   metric=weight_magnitude, protect_core_metrics includes weight_magnitude
# ----------------------------------------------------------------------------

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper Control: LLaMA-3.1-8B (protect baselines)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
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
  name="llama3_8b_paper_results_protect_baselines" \
  generate_plots=false \
  supernode.protect_core=true \
  "supernode.protect_core_metrics=['wanda','weight_magnitude']" \
  pruning_strategies="['wanda','weight_magnitude']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low']" \
  do_connectivity_pruning=false \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B protect-baselines completed at $(date)"
echo "============================================================================"

