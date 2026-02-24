#!/bin/bash
#SBATCH --job-name=fast_prune
#SBATCH --output=logs/fast_pruning_%j.out
#SBATCH --error=logs/fast_pruning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=80GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

# ============================================================================
# FAST LLM PRUNING COMPARISON
# ============================================================================
# Quick iteration version for development and testing
# Expected runtime: ~30-60 minutes on H100
# 
# Changes from comprehensive version:
# - 3 sparsity levels (0.3, 0.5, 0.7) instead of 9
# - 1 selection mode (low) instead of 2
# - 4 algorithms instead of 9
# - Dropped slow benchmarks (GSM8k, MBPP, HumanEval)
# - 50 eval samples instead of 100
# ============================================================================

echo "============================================================================"
echo "FAST LLM PRUNING COMPARISON"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate alignenv2

cd /n/holylfs06/LABS/kempner_undergrads/Lab/acherilyn/alignment

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/n/home13/hsafaai/.cache/huggingface
# export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "============================================================================"
echo "FAST MODE CONFIGURATION:"
echo "============================================================================"
echo ""
echo "PRUNING METHODS (4 key methods):"
echo "  - rayleigh_quotient      (Our main alignment method)"
echo "  - scar_loss_proxy        (Gradient-informed)"
echo "  - activation_l2_norm     (Magnitude baseline)"
echo "  - wanda                  (SOTA baseline)"
echo ""
echo "SPARSITY LEVELS: 30%, 50%, 70%"
echo "SELECTION MODE: low only"
echo ""
echo "EVALUATION BENCHMARKS (fast only):"
echo "  - Perplexity, Loss, Bits-per-Byte"
echo "  - MMLU, HellaSwag, ARC-Easy/Challenge"
echo "  - WinoGrande, PIQA, BoolQ, TruthfulQA"
echo ""
echo "SKIPPED (slow generation-based):"
echo "  - GSM8k, MBPP, HumanEval"
echo "============================================================================"
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_activation_vs_scar.yaml \
    --device cuda

echo ""
echo "============================================================================"
echo "Fast pruning comparison completed at $(date)"
echo "============================================================================"
