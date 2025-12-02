#!/bin/bash
#SBATCH --job-name=llama3_rq_l2
#SBATCH --output=logs/llama3_rq_l2_%j.out
#SBATCH --error=logs/llama3_rq_l2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

echo "=========================================="
echo "LLaMA-3 RQ + L2 Pruning Experiment"
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

echo "Running experiment..."
python scripts/run_experiment.py \
    --config configs/examples/llama3_rq_l2_pruning.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Experiment completed at $(date)"
echo "=========================================="

