#!/bin/bash
#SBATCH --job-name=vision_r18_ablation
#SBATCH --output=logs/vision_r18_ablation_%j.out
#SBATCH --error=logs/vision_r18_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# ResNet-18 ablation at 50% sparsity:
# - cluster_aware (full)
# - cluster_aware_no_halo (lambda=0)
# - cluster_aware_no_constraints
# - composite
# ----------------------------------------------------------------------------

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper Ablation: ResNet-18/CIFAR-10 @ 50% sparsity"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: $OUTPUT_BASE"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python scripts/run_experiment.py \
  --config configs/vision_prune/resnet18_cifar10_unified.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE" \
  name="resnet18_cifar10_cluster_analysis_ablation" \
  pruning_amounts="[0.5]" \
  pruning_distribution="global_threshold" \
  pruning_strategies="['cluster_aware','cluster_aware_no_halo','cluster_aware_no_constraints','composite']"

echo ""
echo "Done: $(date)"

