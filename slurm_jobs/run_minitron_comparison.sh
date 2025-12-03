#!/bin/bash
#SBATCH --job-name=minitron_cmp
#SBATCH --output=logs/minitron_cmp_%j.out
#SBATCH --error=logs/minitron_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

echo "=========================================="
echo "NVIDIA Minitron Comparison Experiment"
echo "Reference: https://arxiv.org/abs/2408.11796"
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

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "=========================================="
echo "NVIDIA Minitron Reference (Llama 3.1 8B → 4B, 50% pruned):"
echo "  Winogrande:     77.3% → 73.5%"
echo "  ARC-Challenge:  57.9% → 55.6%"
echo "  MMLU:           65.3% → 60.5%"
echo "  HellaSwag:      81.8% → 76.1%"
echo "  GSM8k:          48.6% → 41.2%"
echo "  TruthfulQA:     45.0% → 42.9%"
echo "  MBPP:           42.3% → 32.4%"
echo "=========================================="
echo ""
echo "Our Methods to Compare:"
echo "  - Magnitude (L2 norm)"
echo "  - SCAR loss proxy"
echo "  - Rayleigh Quotient"
echo "  - Supernode protection"
echo "  - Supernode connectivity"
echo "  - Wanda (Sun et al., 2023)"
echo "  - SparseGPT (Frantar & Alistarh, 2023)"
echo ""
echo "Sparsity Levels: 25%, 50%, 75%"
echo "=========================================="
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_minitron_comparison.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Minitron comparison completed at $(date)"
echo "=========================================="

