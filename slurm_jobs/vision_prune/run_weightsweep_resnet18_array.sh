#!/bin/bash
#SBATCH --job-name=vision_r18_wtsweep
#SBATCH --output=logs/vision_r18_wtsweep_%A_%a.out
#SBATCH --error=logs/vision_r18_wtsweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-14

# ----------------------------------------------------------------------------
# Vision paper sweep: cluster-aware score weight sensitivity
# We sweep (gamma, lambda_halo) while holding alpha=1.0, beta=0.5.
# Each task runs:
#   - ResNet-18 / CIFAR-10
#   - method: cluster_aware
#   - pruning across multiple sparsity ratios (so figures show the pruning effect)
# ----------------------------------------------------------------------------

set -euo pipefail

GAMMAS=(0.10 0.30 0.50)
LAMBDAS=(0.00 0.25 0.50 0.75 1.00)

IDX="${SLURM_ARRAY_TASK_ID}"
GI=$((IDX / ${#LAMBDAS[@]}))
LI=$((IDX % ${#LAMBDAS[@]}))

GAMMA="${GAMMAS[$GI]}"
LAMBDA="${LAMBDAS[$LI]}"

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper Sweep: ResNet-18 weight sensitivity (gamma=${GAMMA}, lambda=${LAMBDA})"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Array Task: ${SLURM_ARRAY_TASK_ID}"
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
  name="resnet18_cifar10_weightsweep_g${GAMMA}_l${LAMBDA}" \
  pruning_amounts="[0.1,0.3,0.5,0.7,0.8,0.9]" \
  pruning_distribution="global_threshold" \
  pruning_strategies="['cluster_aware']" \
  pruning.cluster_aware.gamma="${GAMMA}" \
  pruning.cluster_aware.lambda_halo="${LAMBDA}"

echo ""
echo "Done: $(date)"

