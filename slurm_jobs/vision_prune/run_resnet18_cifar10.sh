#!/bin/bash
#SBATCH --job-name=vision_resnet18_cifar10
#SBATCH --output=logs/vision_resnet18_cifar10_%j.out
#SBATCH --error=logs/vision_resnet18_cifar10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: ResNet-18 on CIFAR-10"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Output Base: $OUTPUT_BASE"
echo ""

# Environment setup (adjust to your cluster defaults)
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python scripts/run_experiment.py \
  --config configs/vision_prune/resnet18_cifar10_unified.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"
echo "Look under: $OUTPUT_BASE/ (experiment name: resnet18_cifar10_cluster_analysis_*)"

