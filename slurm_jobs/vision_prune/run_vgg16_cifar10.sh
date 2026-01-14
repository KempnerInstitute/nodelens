#!/bin/bash
#SBATCH --job-name=vision_vgg16_cifar10
#SBATCH --output=logs/vision_vgg16_cifar10_%j.out
#SBATCH --error=logs/vision_vgg16_cifar10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: VGG-16-BN on CIFAR-10"
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
  --config configs/vision_prune/vgg16_cifar10_unified.yaml \
  --device cuda \
  --base-output-dir "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"
echo "Look under: $OUTPUT_BASE/ (experiment name: vgg16_cifar10_cluster_analysis_*)"

