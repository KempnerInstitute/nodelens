#!/bin/bash
#SBATCH --job-name=cmp_cfgs_42
#SBATCH --output=logs/cmp_cfgs_42_%j.out
#SBATCH --error=logs/cmp_cfgs_42_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --account=kempner_dev

# -----------------------------------------------------------------------------
# Compare two analysis/pruning configurations on the *same trained checkpoint*.
#
# This isolates analysis/pruning configuration changes (task sampling, type mapping,
# pruning distribution caps, etc) from training randomness.
#
# Usage:
#   sbatch -p kempner_eng slurm_jobs/vision_prune/compare_configs_from_checkpoint_seed42.sh
# -----------------------------------------------------------------------------

set -euo pipefail

SRC_DIR="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER/resnet18_cifar10_cluster_analysis_20260120_183641_56123534"
CFG="${SRC_DIR}/experiment_config.yaml"
CKPT="${SRC_DIR}/checkpoints/trained_model.pth"
OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER_COMPARE_CONFIGS_FROM_CKPT"
SEED="42"

echo "============================================================================"
echo "Compare configs (seed=${SEED})"
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

# ---------------------------------------------------------------------------
# Run A: "current" analysis/pruning choices (per-image task stats; stable mapping; safety cap on)
# ---------------------------------------------------------------------------
python scripts/run_experiment.py \
  --config "${CFG}" \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "${OUTPUT_BASE}/A_current" \
  calibration_mode=train_loader \
  task_activation_samples=None \
  type_mapping_mode=global \
  pruning_max_per_layer_sparsity_cap=0.90 \
  do_train=False \
  model_checkpoint="${CKPT}" \
  generate_plots=False \
  pruning_amounts='[0.9,0.95]' \
  pruning_strategies='["cluster_aware","cluster_aware_annealed","taylor"]'

# ---------------------------------------------------------------------------
# Run B: "greedy/match/no-cap" configuration (useful for reproducing historical behavior)
# ---------------------------------------------------------------------------
python scripts/run_experiment.py \
  --config "${CFG}" \
  --device cuda \
  --seed "${SEED}" \
  --base-output-dir "${OUTPUT_BASE}/B_greedy_match_nocap" \
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

