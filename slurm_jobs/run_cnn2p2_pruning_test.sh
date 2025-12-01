#!/bin/bash
#SBATCH --job-name=cnn2p2_pruning_test
#SBATCH --output=logs/cnn2p2_pruning_test_%j.out
#SBATCH --error=logs/cnn2p2_pruning_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "=========================================="
echo "CNN2P2 Pruning Test (CIFAR-10)"
echo "Simple CNN baseline before AlexNet"
echo "=========================================="
echo "Started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Create logs directory if it doesn't exist
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Default to CIFAR-10 config, can override with argument
CONFIG=${1:-configs/examples/cnn2p2_pruning.yaml}

echo "Config: $CONFIG"
echo ""
echo "Running experiment..."
python scripts/run_experiment.py \
    --config "$CONFIG" \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

