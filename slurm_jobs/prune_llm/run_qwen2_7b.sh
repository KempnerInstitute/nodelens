#!/bin/bash
#SBATCH --job-name=paper_qwen2_7b
#SBATCH --output=logs/paper_qwen2_7b_%j.out
#SBATCH --error=logs/paper_qwen2_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

# ============================================================================
# QWEN2-7B PAPER RESULTS (Generalization)
# ============================================================================
# Cross-model generalization experiment
# Qwen2 has different FFN architecture (28 layers, larger intermediate)
# Expected runtime: ~4-6 hours on H100
# ============================================================================

echo "============================================================================"
echo "SCAR Paper: Qwen2-7B (Generalization)"
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
mkdir -p results/paper/qwen2_7b

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo ""
echo "Running Qwen2-7B full paper analysis..."
echo ""

python scripts/run_experiment.py \
    --config configs/paper/qwen2_7b_full.yaml \
    --device cuda

echo ""
echo "============================================================================"
echo "Qwen2-7B completed at $(date)"
echo "============================================================================"
echo ""
echo "Results saved to: results/paper/qwen2_7b/"
