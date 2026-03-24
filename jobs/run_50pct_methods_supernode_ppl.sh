#!/bin/bash
#SBATCH --job-name=llama3_50pct_methods
#SBATCH --output=logs/llama3_50pct_methods_%j.out
#SBATCH --error=logs/llama3_50pct_methods_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

set -euo pipefail

echo "============================================================================"
echo "LLAMA3 50% METHODS + SUPERNODE/PPL"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate alignenv2

cd /n/holylfs06/LABS/kempner_undergrads/Lab/acherilyn/alignment

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Concurrency-safe run directory: unique per SLURM job (and array task if present).
RUN_TAG="${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_TAG="${RUN_TAG}_${SLURM_ARRAY_TASK_ID}"
fi
RUN_DIR="results/llama3_50pct_methods_supernode_ppl/job_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/run_experiment.py \
    --config configs/examples/llama3_50pct_methods_supernode_ppl.yaml \
    --device cuda \
    --output-dir "$RUN_DIR" \
    --allow-dirty

# Generate only the LaTeX summary table (no figures) from this exact run.
python3 scripts/generate_pruning_supernode_ppl_table.py \
    --experiment-dir "$RUN_DIR"

echo ""
echo "============================================================================"
echo "Completed at $(date)"
echo "============================================================================"
