#!/bin/bash
#SBATCH --job-name=multimodel_prune
#SBATCH --output=logs/multimodel_pruning_%j_%a.out
#SBATCH --error=logs/multimodel_pruning_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-4

# ============================================================================
# MULTI-MODEL PRUNING COMPARISON (Array Job)
# ============================================================================
# Runs SCAR pruning comparison across multiple LLM architectures
# 
# Models tested:
#   0: Mistral-7B     (mistralai/Mistral-7B-v0.1)
#   1: Llama-2-7B     (meta-llama/Llama-2-7b-hf)
#   2: Gemma-2B       (google/gemma-2b) - smaller/faster
#   3: Phi-3 Mini     (microsoft/Phi-3-mini-4k-instruct) - smaller/faster
#   4: Qwen2-7B       (Qwen/Qwen2-7B)
#
# Expected runtime per model: 6-12 hours on H100
# ============================================================================

# Define model configs as array
CONFIGS=(
    "configs/examples/mistral7b_pruning.yaml"
    "configs/examples/llama2_7b_pruning.yaml"
    "configs/examples/gemma2b_pruning.yaml"
    "configs/examples/phi3_mini_pruning.yaml"
    "configs/examples/qwen2_7b_pruning.yaml"
)

MODEL_NAMES=(
    "Mistral-7B"
    "Llama-2-7B"
    "Gemma-2B"
    "Phi-3-Mini"
    "Qwen2-7B"
)

# Get config for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================================"
echo "MULTI-MODEL PRUNING: ${MODEL_NAME}"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
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
echo "Running experiment for ${MODEL_NAME}..."
echo "============================================================================"

python scripts/run_experiment.py \
    --config "$CONFIG" \
    --device cuda

echo ""
echo "============================================================================"
echo "${MODEL_NAME} pruning completed at $(date)"
echo "============================================================================"
