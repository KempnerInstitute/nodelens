#!/bin/bash
#SBATCH --job-name=sn_robust
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/supernode_robustness_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/supernode_robustness_%j.err
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# ============================================================================
# SUPERNODE ROBUSTNESS ANALYSIS JOB
# ============================================================================
# Analyzes the consistency of supernode identification across:
# - Different metrics (RQ, MI, SCAR, magnitude)
# - Different data batches (bootstrap sampling)
#
# Key outputs:
# - Jaccard similarity heatmaps
# - Spearman correlation heatmaps
# - Bootstrap stability distributions
# - Cross-metric consistency plots
# ============================================================================

set -e

echo "============================================================================"
echo "SUPERNODE ROBUSTNESS ANALYSIS"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Setup environment
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate environment
conda activate networkAlignmentAnalysis

# Verify Python environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "============================================================================"
echo "ANALYSIS FOCUS:"
echo "============================================================================"
echo ""
echo "1. CROSS-METRIC CONSISTENCY:"
echo "   - Do different metrics identify the same neurons as supernodes?"
echo "   - Metrics: SCAR activation power, SCAR loss proxy, SCAR taylor,"
echo "             Rayleigh quotient, Gaussian MI, Activation L2 norm"
echo ""
echo "2. BOOTSTRAP STABILITY:"
echo "   - Are supernodes consistent across different input samples?"
echo "   - 10 bootstrap resamples per layer"
echo ""
echo "3. LAYERS ANALYZED:"
echo "   - Layer 5 (early)"
echo "   - Layer 15 (middle)"  
echo "   - Layer 25 (late)"
echo ""
echo "============================================================================"
echo ""

# Set HuggingFace cache (optional but recommended for cluster)
export HF_HOME=/n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache
export TRANSFORMERS_CACHE=/n/holyscratch01/kempner_dev/Users/hsafaai/huggingface_cache

# Run the experiment
python scripts/run_experiment.py \
    --config configs/examples/llama3_supernode_robustness.yaml

echo ""
echo "============================================================================"
echo "Job completed at: $(date)"
echo "============================================================================"

