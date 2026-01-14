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

# NOTE: This is a *submission* script (it calls `sbatch ...` for the real jobs).
# Run it with `bash ...` from a login node. If you accidentally run it with `sbatch`,
# Slurm would normally create `slurm-<jobid>.out` in the repo root; we redirect that
# output to /tmp to avoid polluting the source tree.
#SBATCH --job-name=submit_scar_paper
#SBATCH --output=/tmp/%x_%j.out
#SBATCH --error=/tmp/%x_%j.err

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
# Ensure compute jobs can find the HuggingFace token/cache.
# If you ran `hf auth login` with HF_HOME under OUTPUT_BASE, this propagates it to all sbatch jobs.
export HF_HOME="${HF_HOME:-${OUTPUT_BASE}/huggingface_cache}"
mkdir -p "$HF_HOME" || true
SUBMIT_UNSTRUCTURED_BASELINES="${SUBMIT_UNSTRUCTURED_BASELINES:-0}"

echo "=============================================="
echo "Submitting SCAR Paper Experiments"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_BASE"
echo "Submit unstructured baseline reproductions: $SUBMIT_UNSTRUCTURED_BASELINES (set to 1 to enable)"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

# Submit all jobs
echo "Submitting LLaMA-3.1-8B (main results)..."
JOB1=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting Mistral-7B (generalization)..."
JOB2=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_mistral_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting LLaMA-2-7B (generalization)..."
JOB3=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama2_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Qwen2-7B (generalization)..."
JOB4=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_qwen2_7b.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

if [[ "$SUBMIT_UNSTRUCTURED_BASELINES" == "1" ]]; then
  echo ""
  echo "---- Paper-faithful unstructured baseline reproductions (LLaMA-3.1-8B) ----"
  echo "Submitting Wanda (unstructured)..."
  JOB5=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_wanda_unstructured.sh | awk '{print $4}')
  echo "  Job ID: $JOB5"

  echo "Submitting SparseGPT (unstructured + reconstruction)..."
  JOB6=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_sparsegpt_unstructured.sh | awk '{print $4}')
  echo "  Job ID: $JOB6"
fi

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
if [[ "$SUBMIT_UNSTRUCTURED_BASELINES" == "1" ]]; then
  echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5, $JOB6"
else
  echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
fi
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
if [[ "$SUBMIT_UNSTRUCTURED_BASELINES" == "1" ]]; then
  echo "  $OUTPUT_BASE/llama3_8b_paper_results_wanda_unstructured_*_${JOB5}/"
  echo "  $OUTPUT_BASE/llama3_8b_paper_results_sparsegpt_unstructured_*_${JOB6}/"
fi