#!/bin/bash
#SBATCH --job-name=vision_pruning_test
#SBATCH --output=logs/vision_pruning_test_%j.out
#SBATCH --error=logs/vision_pruning_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

echo "=========================================="
echo "Vision Pruning Test (AlexNet on ImageNet)"
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

echo "Running experiment..."
python scripts/run_experiment.py \
    --config configs/examples/vision_pruning_test.yaml \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

