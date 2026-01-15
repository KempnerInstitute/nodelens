#!/bin/bash
# ============================================================================
# SUBMIT FULL SCAR PAPER SUITE (main + controls/ablations)
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/prune_llm/submit_suite.sh
#
# Output:
#   Uses OUTPUT_BASE (exported or defaulted below).
# ============================================================================

# NOTE: This is a *submission* script (it calls `sbatch ...` for the real jobs).
# Run it with `bash ...` from a login node. If you accidentally run it with `sbatch`,
# Slurm would normally create `slurm-<jobid>.out` in the repo root; we redirect that
# output to /tmp to avoid polluting the source tree.
#SBATCH --job-name=submit_scar_paper_suite
#SBATCH --output=/tmp/%x_%j.out
#SBATCH --error=/tmp/%x_%j.err

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
# Ensure compute jobs can find the HuggingFace token/cache.
# If you ran `hf auth login` with HF_HOME under OUTPUT_BASE, this propagates it to all sbatch jobs.
export HF_HOME="${HF_HOME:-${OUTPUT_BASE}/huggingface_cache}"
mkdir -p "$HF_HOME" || true

echo "=============================================="
echo "Submitting SCAR Paper Suite"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

echo "---- Main results + generalization (4 models) ----"
export OUTPUT_BASE
bash slurm_jobs/prune_llm/run_all_paper.sh
echo ""

echo "---- Controls / ablations (Llama-3.1-8B) ----"
JOB_NP=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_noprotect.sh | awk '{print $4}')
echo "  noprotect/control: $JOB_NP"

JOB_PB=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_protect_baselines.sh | awk '{print $4}')
echo "  protect-baselines: $JOB_PB"

JOB_POSRED=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_positive_redundancy_array.sh | awk '{print $4}')
echo "  pos-redundancy array: $JOB_POSRED"

JOB_CALIB=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME",HF_TOKEN=,HUGGINGFACE_HUB_TOKEN= slurm_jobs/prune_llm/run_llama3_8b_calibration_array.sh | awk '{print $4}')
echo "  calibration array: $JOB_CALIB"

echo ""
echo "=============================================="
echo "All suite jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

