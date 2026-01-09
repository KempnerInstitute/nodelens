#!/bin/bash
#SBATCH --job-name=paper_mistral_7b
#SBATCH --output=logs/paper_mistral_7b_%j.out
#SBATCH --error=logs/paper_mistral_7b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=320GB
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev

# ============================================================================
# MISTRAL-7B PAPER RESULTS (Generalization)
# ============================================================================
# Cross-model generalization experiment
# Expected runtime: ~4-6 hours on H100
#
# Output Directory Structure:
#   /n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/
#       mistral_7b_paper_results_{timestamp}_{SLURM_JOB_ID}/
#           results/      - JSON results files
#           logs/         - experiment.log
#           figures/      - All visualizations
#           checkpoints/  - Model checkpoints
#           analysis/     - Post-analysis outputs
# ============================================================================

echo "============================================================================"
echo "SCAR Paper: Mistral-7B (Generalization)"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
echo "Output Base: $OUTPUT_BASE"
echo ""

# Environment setup
module purge
module load cuda/12.2.0-fasrc01
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Create local logs directory for SLURM output files
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/home13/hsafaai/.cache/huggingface
export HF_TOKEN=$(cat /n/home13/hsafaai/.cache/huggingface/token)

echo ""
echo "Running Mistral-7B full paper analysis..."
echo ""

python scripts/run_experiment.py \
    --config configs/prune_llm/mistral_7b_full.yaml \
    --device cuda \
    --base-output-dir "$OUTPUT_BASE"

echo ""
echo "============================================================================"
echo "Mistral-7B completed at $(date)"
echo "============================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo "Look for directory: mistral_7b_paper_results_*_$SLURM_JOB_ID"
