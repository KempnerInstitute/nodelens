#!/bin/bash
#SBATCH --job-name=repro_from_dir
#SBATCH --output=logs/repro_from_dir_%j.out
#SBATCH --error=logs/repro_from_dir_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --account=kempner_dev

# -----------------------------------------------------------------------------
# Generic "reproduce from an existing run directory" runner.
#
# Expected SRC_DIR layout:
#   SRC_DIR/experiment_config.yaml
#   SRC_DIR/checkpoints/trained_model.pth
#
# Usage:
#   sbatch -p kempner_eng --export=ALL,SRC_DIR=/abs/path/to/old_run_dir,OUTPUT_BASE=/abs/path/to/output_base slurm_jobs/vision_prune/repro_from_dir.sh
# -----------------------------------------------------------------------------

set -euo pipefail

SRC_DIR="${SRC_DIR:?Must set SRC_DIR=/abs/path/to/old_run_dir}"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER_REPRO_FROM_DIR}"

CFG="${SRC_DIR}/experiment_config.yaml"
CKPT="${SRC_DIR}/checkpoints/trained_model.pth"
SEED="${SEED:-42}"

echo "============================================================================"
echo "Repro from dir (seed=${SEED})"
echo "SRC_DIR: ${SRC_DIR}"
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

