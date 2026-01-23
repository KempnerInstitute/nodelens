#!/bin/bash
#SBATCH --job-name=vision_r18_abperm
#SBATCH --output=logs/vision_r18_abperm_%j.out
#SBATCH --error=logs/vision_r18_abperm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:30:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

# ----------------------------------------------------------------------------
# ResNet-18 / CIFAR-10: ablation + permutation diagnostics (single seed)
# ----------------------------------------------------------------------------

set -euo pipefail

SEED="${SEED:-42}"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER}"

echo "============================================================================"
echo "Vision Paper (diagnostics): ResNet-18/CIFAR-10 seed=${SEED}"
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
  --seed "${SEED}" \
  --base-output-dir "$OUTPUT_BASE" \
  clustering.ablation.enabled=true \
  clustering.ablation.modes="['all','rq_red','rq_syn','red_syn']" \
  halo_analysis.permutation_baseline.enabled=true \
  halo_analysis.permutation_baseline.n_permutations=100

echo ""
echo "Done: $(date)"
