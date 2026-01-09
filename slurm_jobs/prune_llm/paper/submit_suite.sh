#!/bin/bash
# ============================================================================
# SUBMIT FULL SCAR PAPER SUITE (main + controls/ablations)
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/prune_llm/paper/submit_suite.sh
#
# Output:
#   Uses OUTPUT_BASE (exported or defaulted below).
# ============================================================================

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"

echo "=============================================="
echo "Submitting SCAR Paper Suite"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

echo "---- Main results + generalization (4 models) ----"
export OUTPUT_BASE
bash slurm_jobs/prune_llm/run_all_paper.sh
echo ""

echo "---- Controls / ablations (Llama-3.1-8B) ----"
JOB_NP=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/paper/run_llama3_8b_noprotect.sh | awk '{print $4}')
echo "  noprotect/control: $JOB_NP"

JOB_PB=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/paper/run_llama3_8b_protect_baselines.sh | awk '{print $4}')
echo "  protect-baselines: $JOB_PB"

JOB_POSRED=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/paper/run_llama3_8b_positive_redundancy_array.sh | awk '{print $4}')
echo "  pos-redundancy array: $JOB_POSRED"

JOB_CALIB=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/prune_llm/paper/run_llama3_8b_calibration_array.sh | awk '{print $4}')
echo "  calibration array: $JOB_CALIB"

echo ""
echo "=============================================="
echo "All suite jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

