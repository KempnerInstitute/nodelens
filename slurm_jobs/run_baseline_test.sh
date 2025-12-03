#!/bin/bash
#SBATCH --job-name=baseline_test
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_test_%j.out
#SBATCH --error=logs/baseline_test_%j.err

# Quick test for Wanda/SparseGPT integration
# Expected runtime: ~30-60 minutes

set -e

echo "=========================================="
echo "Baseline Pruning Test (Wanda + SparseGPT)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

# Set up paths
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

# Create logs directory
mkdir -p logs

# Run experiment
echo "Running baseline test..."
python scripts/run_experiment.py \
    --config configs/examples/llama3_baseline_test.yaml

echo ""
echo "=========================================="
echo "Baseline test completed at $(date)"
echo "=========================================="

