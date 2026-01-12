#!/bin/bash
#SBATCH --job-name=llama3_scar
#SBATCH --output=logs/llama3_scar_%j.out
#SBATCH --error=logs/llama3_scar_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_undergrads

echo "=========================================="
echo "LLaMA-3 SCAR-Based Pruning with Supernode Protection"
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
conda activate alignenv2

cd /n/holylfs06/LABS/kempner_undergrads/Lab/acherilyn/alignment

# Make logs directory if it doesn't exist
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/n/home13/hsafaai/.cache/huggingface
# export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo "Running SCAR-based pruning experiment..."
echo "Pruning metrics: L2 norm, SCAR loss proxy"
echo "Selection modes: low, high, random"
echo "Evaluation: perplexity, bits_per_byte, MMLU"
echo ""

python scripts/run_experiment.py \
    --config configs/examples/llama3_scar_pruning.yaml \
    --device cuda

echo ""
echo "=========================================="
echo "Experiment completed at $(date)"
echo "=========================================="
