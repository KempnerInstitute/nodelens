#!/bin/bash
#SBATCH --job-name=test_all_layers
#SBATCH --output=logs/test_all_layers_%j.out
#SBATCH --error=logs/test_all_layers_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

echo "=========================================="
echo "Test: All Layers (MLP + Attention)"
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

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Make logs directory if it doesn't exist
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "Testing with ALL layers (MLP + Attention)..."
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_test_all_layers.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
