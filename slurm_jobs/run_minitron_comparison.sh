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
echo "NVIDIA MINITRON-COMPATIBLE BENCHMARK"
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
echo ""
echo "  Benchmark         │ Baseline │ Pruned  │ Few-shot"
echo "  ─────────────────┼──────────┼─────────┼──────────"
echo "  Winogrande        │ 77.3%    │ 73.5%   │ 5-shot"
echo "  ARC-Challenge     │ 57.9%    │ 55.6%   │ 25-shot"
echo "  MMLU              │ 65.3%    │ 60.5%   │ 5-shot"
echo "  HellaSwag         │ 81.8%    │ 76.1%   │ 10-shot"
echo "  GSM8k             │ 48.6%    │ 41.2%   │ 5-shot+CoT"
echo "  TruthfulQA        │ 45.0%    │ 42.9%   │ 0-shot"
echo "  MBPP              │ 42.3%    │ 32.4%   │ 0-shot"
echo "  HumanEval         │ 24.8%    │ -       │ 0-shot"
echo "=========================================="
echo ""
echo "PRUNING METHODS BEING COMPARED:"
echo ""
echo "  ALIGNMENT-BASED (Our Novel Methods):"
echo "    - rayleigh_quotient (RQ)"
echo "    - gaussian_mi_analytic (MI)"
echo "    - average_redundancy"
echo ""
echo "  SCAR-BASED (Gradient-Informed):"
echo "    - scar_loss_proxy"
echo ""
echo "  SUPERNODE-AWARE (Novel Contribution):"
echo "    - supernode_protection_score"
echo "    - supernode_connectivity_score"
echo ""
echo "  MAGNITUDE-BASED (Baseline):"
echo "    - activation_l2_norm"
echo ""
echo "  SOTA BASELINES (NVIDIA Minitron comparison):"
echo "    - wanda (Sun et al., 2023)"
echo "    - sparsegpt (Frantar & Alistarh, 2023)"
echo ""
echo "Sparsity Levels: 25%, 50%, 75%"
echo "Selection Modes: low, high"
echo "=========================================="
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_minitron_comparison.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Minitron comparison completed at $(date)"
echo "=========================================="
echo ""
echo "Results saved to: results/llama3_minitron_comparison_*/"
echo "Check plots/pruning/ for comparison plots"
echo "=========================================="
