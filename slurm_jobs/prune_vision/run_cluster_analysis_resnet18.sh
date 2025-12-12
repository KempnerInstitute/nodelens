#!/bin/bash
#SBATCH --job-name=cluster_analysis_resnet18
#SBATCH --output=logs/cluster_analysis_resnet18_%j.out
#SBATCH --error=logs/cluster_analysis_resnet18_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ============================================================================
# CLUSTER-BASED ANALYSIS: ResNet-18 on CIFAR-10
# ============================================================================
# Full cluster-based analysis including:
# - Per-channel metrics (RQ, Redundancy, Synergy with continuous target)
# - K-means clustering into functional types
# - Cross-layer halo analysis
# - Cascade damage testing
# - Pruning experiments (without fine-tuning to see raw impact)
# - Organized visualization output
#
# Figure Organization:
#   figures/01_distributions/  - Per-layer metric histograms
#   figures/02_summary/        - Layer-wise violin plots, trends
#   figures/03_clustering/     - Cluster scatter plots, evolution
#   figures/04_cascade/        - Cascade damage test results
#   figures/05_halo/           - Halo analysis plots
#   figures/06_pruning/        - Pruning comparison charts
#
# Expected runtime: ~1-2 hours on single GPU
# ============================================================================

OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM"

echo "============================================================================"
echo "Cluster-Based Analysis: ResNet-18 on CIFAR-10"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output Base: $OUTPUT_BASE"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Create local logs directory for SLURM output files
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "Running ResNet-18 cluster analysis..."
echo "Fine-tuning after pruning: DISABLED (seeing raw pruning impact)"
echo ""
    
python scripts/run_experiment.py \
    --config configs/vision_prune/resnet18_cifar10_unified.yaml \
    --device cuda

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "ResNet-18 cluster analysis completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo "Look for directory starting with: resnet18_cifar10_cluster_analysis_"

exit $EXIT_CODE
