#!/bin/bash
#SBATCH --job-name=comp_prune
#SBATCH --output=logs/comprehensive_pruning_%j.out
#SBATCH --error=logs/comprehensive_pruning_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ============================================================================
# COMPREHENSIVE LLM PRUNING COMPARISON
# ============================================================================
# Compares ALL custom pruning methods vs SOTA baselines
# Expected runtime: 6-12 hours on H100
# ============================================================================

echo "============================================================================"
echo "COMPREHENSIVE LLM PRUNING COMPARISON"
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
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "============================================================================"
echo "PRUNING METHODS TO COMPARE:"
echo "============================================================================"
echo ""
echo "ALIGNMENT-BASED (Our Methods):"
echo "  - rayleigh_quotient          (RQ - alignment measure)"
echo "  - gaussian_mi_analytic       (MI - mutual information)"
echo "  - average_redundancy         (Information-theoretic)"
echo ""
echo "SCAR-BASED (Gradient-Informed):"
echo "  - scar_loss_proxy            (Activation power + curvature)"
echo ""
echo "SUPERNODE-AWARE (Novel Contribution):"
echo "  - supernode_protection_score (Protects unique halo neurons)"
echo "  - supernode_connectivity_score (Low connectivity = safe to prune)"
echo ""
echo "MAGNITUDE-BASED (Baseline):"
echo "  - activation_l2_norm         (Standard magnitude)"
echo ""
echo "SOTA BASELINES:"
echo "  - wanda                      (Sun et al., 2023)"
echo "  - sparsegpt                  (Frantar & Alistarh, 2023)"
echo ""
echo "============================================================================"
echo "SPARSITY LEVELS: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%"
echo "SELECTION MODES: low (prune lowest), high (prune highest)"
echo "============================================================================"
echo ""
echo "EVALUATION BENCHMARKS:"
echo "  - Perplexity, Loss, Bits-per-Byte (WikiText-2)"
echo "  - MMLU (57 subjects)"
echo "  - HellaSwag (Commonsense)"
echo "  - ARC-Easy/Challenge (Science)"
echo "  - WinoGrande (Schemas)"
echo "  - PIQA (Physical intuition)"
echo "  - BoolQ (Boolean QA)"
echo "  - GSM8k (Math)"
echo "  - TruthfulQA"
echo "  - MBPP/HumanEval (Code)"
echo "============================================================================"
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_comprehensive_pruning.yaml \
    --device cuda

echo ""
echo "============================================================================"
echo "Comprehensive pruning comparison completed at $(date)"
echo "============================================================================"

