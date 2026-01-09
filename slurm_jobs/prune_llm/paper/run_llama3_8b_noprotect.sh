#!/bin/bash
#SBATCH --job-name=paper_llama3_noprotect
#SBATCH --output=logs/paper_llama3_noprotect_%j.out
#SBATCH --error=logs/paper_llama3_noprotect_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# LLaMA-3.1-8B CONTROL: LP-no-protect + "remove supernodes early" (mode=high)
#
# Produces (at 50%):
# - LP-no-protect:    metric=scar_loss_proxy, mode=low, protect_core=false
# - Remove-core-early metric=scar_loss_proxy, mode=high, protect_core=false
# ----------------------------------------------------------------------------

set -euo pipefail

echo "============================================================================"
echo "SCAR Paper Control: LLaMA-3.1-8B (no-protect LP control)"
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
  name="llama3_8b_paper_results_noprotect" \
  generate_plots=false \
  supernode.protect_core=false \
  pruning_strategies="['scar_loss_proxy']" \
  pruning_amounts="[0.5]" \
  pruning_selection_mode="['low','high']" \
  do_connectivity_pruning=false \
  do_directed_redundancy=false \
  do_halo_analysis=false \
  do_generalized_importance=false \
  supernode_robustness.enabled=false \
  supernode_summary.enabled=false

echo ""
echo "============================================================================"
echo "LLaMA-3.1-8B no-protect control completed at $(date)"
echo "============================================================================"

