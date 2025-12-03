#!/bin/bash
#SBATCH --job-name=llama3_bench
#SBATCH --output=logs/llama3_bench_%j.out
#SBATCH --error=logs/llama3_bench_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

echo "=========================================="
echo "LLaMA-3 Full Benchmark Suite"
echo "Including NVIDIA Minitron benchmarks + Wanda/SparseGPT baselines"
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

# Make logs directory if it doesn't exist
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "FULL BENCHMARK SUITE:"
echo "====================="
echo "Language Model Metrics:"
echo "  - Perplexity, Loss, Bits-per-byte"
echo ""
echo "NVIDIA Minitron Benchmarks (https://arxiv.org/abs/2407.14679):"
echo "  - MMLU (Massive Multitask Language Understanding)"
echo "  - HellaSwag (Commonsense reasoning)"
echo "  - ARC-Challenge (Hard science questions)"
echo "  - WinoGrande (Winograd schemas)"
echo "  - PIQA (Physical intuition)"
echo "  - TruthfulQA (Truthfulness)"
echo ""
echo "Additional Benchmarks:"
echo "  - ARC-Easy (Science questions)"
echo "  - BoolQ (Boolean questions)"
echo ""
echo "Pruning Strategies:"
echo "  - Magnitude (L2), SCAR, RQ, MI, Redundancy"
echo "  - Supernode protection/connectivity"
echo "  - Wanda (Sun et al., 2023)"
echo "  - SparseGPT (Frantar & Alistarh, 2023)"
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_full_benchmark.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Full benchmark completed at $(date)"
echo "=========================================="

