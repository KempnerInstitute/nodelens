#!/bin/bash
#SBATCH --job-name=repro_ckpt_r18c10
#SBATCH --output=logs/repro_ckpt_r18c10_%j.out
#SBATCH --error=logs/repro_ckpt_r18c10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --account=kempner_dev

# -----------------------------------------------------------------------------
# Reproduce analysis + pruning from a saved trained checkpoint (vision, cluster paper).
#
# This script uses explicit config knobs (task sampling, type mapping, calibration mode,
# pruning caps) rather than any date-specific compatibility flag.
#
# Usage:
#   sbatch -p kempner_eng slurm_jobs/vision_prune/repro_from_checkpoint_resnet18_cifar10_seed42.sh
# -----------------------------------------------------------------------------

set -euo pipefail

CFG="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER/resnet18_cifar10_cluster_analysis_20260120_183641_56123534/experiment_config.yaml"
CKPT="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER/resnet18_cifar10_cluster_analysis_20260120_183641_56123534/checkpoints/trained_model.pth"
OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER_REPRO_FROM_CKPT"
SEED="42"

echo "============================================================================"
echo "Repro from checkpoint (seed=${SEED})"
echo "CFG: ${CFG}"
echo "CKPT: ${CKPT}"
echo "Output Base: ${OUTPUT_BASE}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}  JobID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================================"

module purge
module load cuda/12.2.0-fasrc01

# Conda
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate networkAlignmentAnalysis
fi

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

python scripts/run_experiment.py \
  --config "${CFG}" \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "${OUTPUT_BASE}" \
  calibration_mode=train_loader \
  task_activation_samples=match \
  type_mapping_mode=greedy \
  pruning_max_per_layer_sparsity_cap=1.0 \
  do_train=False \
  model_checkpoint="${CKPT}" \
  generate_plots=False \
  pruning_amounts='[0.9,0.95]' \
  pruning_strategies='["cluster_aware","cluster_aware_annealed","taylor"]'

echo "Done: $(date)"

