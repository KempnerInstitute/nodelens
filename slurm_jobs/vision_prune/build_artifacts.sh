#!/bin/bash
#SBATCH --job-name=vision_paper_build
#SBATCH --output=logs/vision_paper_build_%j.out
#SBATCH --error=logs/vision_paper_build_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_h100_priority3
#SBATCH --account=kempner_dev

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

echo "============================================================================"
echo "Vision Paper: Build all figures + tables from existing runs"
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

python drafts/alignment_notes/paper/scripts/build_all_artifacts.py \
  --results-base "$OUTPUT_BASE"

echo ""
echo "Done: $(date)"
echo "Paper figures: drafts/alignment_notes/paper_figures_vision/"
echo "Paper tables:  drafts/alignment_notes/paper_artifacts/tables/"

