#!/bin/bash
# ============================================================================
# SUBMIT ALL PAPER EXPERIMENTS
# ============================================================================
# This script submits all 4 paper experiments as separate SLURM jobs
# They will run in parallel if resources are available
#
# Usage: 
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/paper/run_all_paper.sh
# ============================================================================

echo "=============================================="
echo "Submitting SCAR Paper Experiments"
echo "=============================================="
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Submit all jobs
echo "Submitting LLaMA-3.1-8B (main results)..."
JOB1=$(sbatch slurm_jobs/paper/run_llama3_8b.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting Mistral-7B (generalization)..."
JOB2=$(sbatch slurm_jobs/paper/run_mistral_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting LLaMA-2-7B (generalization)..."
JOB3=$(sbatch slurm_jobs/paper/run_llama2_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Qwen2-7B (generalization)..."
JOB4=$(sbatch slurm_jobs/paper/run_qwen2_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/paper_llama3_8b_${JOB1}.out"
echo "  tail -f logs/paper_mistral_7b_${JOB2}.out"
echo "  tail -f logs/paper_llama2_7b_${JOB3}.out"
echo "  tail -f logs/paper_qwen2_7b_${JOB4}.out"
echo ""
echo "Expected runtime: ~6-8 hours per job"
echo "Results will be in: results/paper/<model>/"
