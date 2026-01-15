#!/bin/bash
#SBATCH --job-name=vision_r18_gap
#SBATCH --output=logs/vision_r18_gap_%j.out
#SBATCH --error=logs/vision_r18_gap_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper (Appendix): ResNet-18 GAP robustness run"
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
  name="resnet18_cifar10_cluster_analysis_gap" \
  metrics.activation_samples="gap" \
  pruning_amounts="[]"

echo ""
echo "Done: $(date)"

