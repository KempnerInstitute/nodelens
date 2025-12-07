#!/bin/bash
#SBATCH --job-name=single_prune
#SBATCH --output=logs/single_model_%j.out
#SBATCH --error=logs/single_model_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev

# ============================================================================
# SINGLE MODEL PRUNING (Specify config via argument)
# ============================================================================
# Usage: sbatch run_single_model.sh <config_name>
# 
# Examples:
#   sbatch run_single_model.sh mistral7b_pruning
#   sbatch run_single_model.sh llama2_7b_pruning
#   sbatch run_single_model.sh gemma2b_pruning
#   sbatch run_single_model.sh phi3_mini_pruning
#   sbatch run_single_model.sh qwen2_7b_pruning
#   sbatch run_single_model.sh gpt2_fast_test
#
# Available configs:
#   - mistral7b_pruning     (Mistral-7B)
#   - llama2_7b_pruning     (Llama-2-7B)
#   - gemma2b_pruning       (Gemma-2B, smaller)
#   - phi3_mini_pruning     (Phi-3 Mini, smaller)
#   - qwen2_7b_pruning      (Qwen2-7B)
#   - gpt2_fast_test        (GPT-2, very fast)
#   - llama3_minitron_comparison (Llama-3.1-8B, original)
# ============================================================================

# Get config name from argument, default to llama3 if not provided
CONFIG_NAME=${1:-"llama3_minitron_comparison"}
CONFIG="configs/examples/${CONFIG_NAME}.yaml"

echo "============================================================================"
echo "SINGLE MODEL PRUNING: ${CONFIG_NAME}"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo ""
    echo "Available configs:"
    ls -1 configs/examples/*.yaml | sed 's|configs/examples/||' | sed 's|.yaml||'
    exit 1
fi

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
echo "Running experiment..."
echo "============================================================================"

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --device cuda

echo ""
echo "============================================================================"
echo "${CONFIG_NAME} pruning completed at $(date)"
echo "============================================================================"
