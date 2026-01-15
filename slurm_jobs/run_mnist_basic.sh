#!/bin/bash
#SBATCH --job-name=mnist_basic_align
#SBATCH --output=logs/mnist_basic_%j.out
#SBATCH --error=logs/mnist_basic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
 
set -euo pipefail

# NOTE: Cluster-specific SBATCH settings like --partition/--account are intentionally omitted.
# Submit with your local settings, e.g.:
#   sbatch --partition=<PARTITION> --account=<ACCOUNT> slurm_jobs/run_mnist_basic.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"

echo "Starting MNIST basic alignment experiment at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on: $(hostname)"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-networkAlignmentAnalysis}"
else
    echo "WARN: conda not found; assuming environment already activated." >&2
fi

mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python scripts/run_experiment.py \
    --config configs/examples/mnist_basic.yaml \
    --device cpu

echo "MNIST basic alignment experiment completed at $(date)"


