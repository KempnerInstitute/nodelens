#!/bin/bash
#SBATCH --job-name=cluster_analysis_resnet50
#SBATCH --output=logs/cluster_analysis_resnet50_%j.out
#SBATCH --error=logs/cluster_analysis_resnet50_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ============================================================================
# CLUSTER-BASED ANALYSIS: ResNet-50 on ImageNet-100
# ============================================================================
# Full cluster-based analysis including:
# - Per-channel metrics (RQ, Redundancy, Synergy with continuous target)
# - K-means clustering into functional types
# - Cross-layer halo analysis with activation weighting
# - Cascade damage testing
# - Pruning experiments with fine-tuning
# - Visualization generation
#
# Expected runtime: ~6-10 hours on single GPU (A100)
# ============================================================================

echo "============================================================================"
echo "Cluster-Based Analysis: ResNet-50 on ImageNet-100"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Create directories
mkdir -p logs
mkdir -p results/vision/resnet50_imagenet100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check for ImageNet-100 data
if [ ! -d "data/imagenet100" ]; then
    echo "WARNING: ImageNet-100 data not found at data/imagenet100"
    echo "Please download or symlink the ImageNet-100 subset before running."
    echo ""
fi

echo ""
echo "Running ResNet-50 cluster analysis..."
echo ""

python scripts/run_experiment.py \
    --config configs/cluster_analysis/resnet50_imagenet100.yaml \
    --device cuda

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "ResNet-50 cluster analysis completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================================"
echo ""
echo "Results saved to: results/vision/resnet50_imagenet100/"

exit $EXIT_CODE
