#!/bin/bash
#SBATCH --job-name=vision_r18_damagepred
#SBATCH --output=logs/vision_r18_damagepred_%j.out
#SBATCH --error=logs/vision_r18_damagepred_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# Mechanism evaluation: per-channel damage prediction correlation (ResNet-18)
# ----------------------------------------------------------------------------

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: Damage prediction eval (ResNet-18)"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python drafts/alignment_notes/paper/scripts/run_damage_prediction.py \
  --results-base "$OUTPUT_BASE" \
  --exp "resnet18_cifar10_cluster_analysis" \
  --damage-frac 0.15 \
  --eval-examples 2000

echo ""
echo "Done: $(date)"

