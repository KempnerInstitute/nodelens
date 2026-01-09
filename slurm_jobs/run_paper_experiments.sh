#!/bin/bash
#SBATCH --job-name=scar_paper
#SBATCH --output=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/paper_%j.out
#SBATCH --error=/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/logs/paper_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

# ============================================================================
# SCAR PAPER EXPERIMENTS
# ============================================================================
# This script runs all experiments needed for the SCAR paper:
# 1. LLaMA-3.1-8B full analysis (main results)
# 2. Mistral-7B (generalization)
# 3. LLaMA-2-7B (generalization)
# 4. Qwen2-7B (generalization)
#
# Expected runtime: ~20-30 hours total (6-8h per model)
# ============================================================================

set -e

echo "=============================================="
echo "SCAR Paper Experiments"
echo "=============================================="
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Setup
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
source ~/.bashrc
conda activate alignment

# Create output directories
mkdir -p logs
mkdir -p results/paper

# ============================================================================
# Experiment 1: LLaMA-3.1-8B (Main Results)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 1: LLaMA-3.1-8B (Main Results)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/prune_llm/llama3_8b_full.yaml \
    2>&1 | tee logs/llama3_8b_paper.log

echo "LLaMA-3.1-8B completed at $(date)"

# ============================================================================
# Experiment 2: Mistral-7B (Generalization)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 2: Mistral-7B (Generalization)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/prune_llm/mistral_7b_full.yaml \
    2>&1 | tee logs/mistral_7b_paper.log

echo "Mistral-7B completed at $(date)"

# ============================================================================
# Experiment 3: LLaMA-2-7B (Generalization)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 3: LLaMA-2-7B (Generalization)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/prune_llm/llama2_7b_full.yaml \
    2>&1 | tee logs/llama2_7b_paper.log

echo "LLaMA-2-7B completed at $(date)"

# ============================================================================
# Experiment 4: Qwen2-7B (Generalization)
# ============================================================================
echo ""
echo "=============================================="
echo "Experiment 4: Qwen2-7B (Generalization)"
echo "=============================================="

python scripts/run_experiment.py \
    --config configs/prune_llm/qwen2_7b_full.yaml \
    2>&1 | tee logs/qwen2_7b_paper.log

echo "Qwen2-7B completed at $(date)"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - results/paper/llama3_8b/"
echo "  - results/paper/mistral_7b/"
echo "  - results/paper/llama2_7b/"
echo "  - results/paper/qwen2_7b/"
echo ""
echo "Figures ready for paper in:"
echo "  - results/paper/*/figures/"
