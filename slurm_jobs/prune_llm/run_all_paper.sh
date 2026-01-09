#!/bin/bash
# ============================================================================
# SUBMIT ALL PAPER EXPERIMENTS
# ============================================================================
# This script submits all 4 paper experiments as separate SLURM jobs
# They will run in parallel if resources are available
#
# Output Directory Structure:
#   All results go to: /n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/
#   Each job creates a unique directory: {model}_paper_results_{timestamp}_{job_id}/
#       results/      - JSON results files
#       logs/         - experiment.log
#       figures/      - All visualizations
#       checkpoints/  - Model checkpoints
#       analysis/     - Post-analysis outputs
#
# Usage: 
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/prune_llm/run_all_paper.sh
# ============================================================================

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"

echo "=============================================="
echo "Submitting SCAR Paper Experiments"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Submit all jobs
echo "Submitting LLaMA-3.1-8B (main results)..."
JOB1=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/run_llama3_8b.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting Mistral-7B (generalization)..."
JOB2=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/run_mistral_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting LLaMA-2-7B (generalization)..."
JOB3=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/run_llama2_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Qwen2-7B (generalization)..."
JOB4=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/run_qwen2_7b.sh | awk '{print $4}')
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
echo "View SLURM logs:"
echo "  tail -f logs/paper_llama3_8b_${JOB1}.out"
echo "  tail -f logs/paper_mistral_7b_${JOB2}.out"
echo "  tail -f logs/paper_llama2_7b_${JOB3}.out"
echo "  tail -f logs/paper_qwen2_7b_${JOB4}.out"
echo ""
echo "Expected runtime: ~6-8 hours per job"
echo ""
echo "Results will be in:"
echo "  $OUTPUT_BASE/llama3_8b_paper_results_*_${JOB1}/"
echo "  $OUTPUT_BASE/mistral_7b_paper_results_*_${JOB2}/"
echo "  $OUTPUT_BASE/llama2_7b_paper_results_*_${JOB3}/"
echo "  $OUTPUT_BASE/qwen2_7b_paper_results_*_${JOB4}/"
