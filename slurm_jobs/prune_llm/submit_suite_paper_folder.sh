#!/bin/bash
# ============================================================================
# SUBMIT FULL SCAR PAPER SUITE into OUTPUT_BASE/PAPER
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/prune_llm/submit_suite_paper_folder.sh
#
# Output:
#   Writes all new job dirs under: ${OUTPUT_BASE_ROOT}/PAPER/
# ============================================================================

# NOTE: This is a *submission* script (it calls `sbatch ...` for the real jobs).
# If you accidentally run it with `sbatch`, Slurm would normally create `slurm-<jobid>.out`
# in the repo root; we redirect that output to /tmp to avoid polluting the source tree.
#SBATCH --job-name=submit_scar_paper_suite_paper_folder
#SBATCH --output=/tmp/%x_%j.out
#SBATCH --error=/tmp/%x_%j.err

set -euo pipefail

OUTPUT_BASE_ROOT="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM}"
OUTPUT_BASE="${OUTPUT_BASE_ROOT}/PAPER"
export OUTPUT_BASE

# Ensure compute jobs can find the HuggingFace token/cache.
# IMPORTANT: keep HF_HOME in the *root* output base so we reuse the cache/token across PAPER reruns.
export HF_HOME="${HF_HOME:-${OUTPUT_BASE_ROOT}/huggingface_cache}"
mkdir -p "$HF_HOME" || true

echo "=============================================="
echo "Submitting SCAR Paper Suite (PAPER folder)"
echo "=============================================="
echo "OUTPUT_BASE_ROOT: $OUTPUT_BASE_ROOT"
echo "OUTPUT_BASE (runs): $OUTPUT_BASE"
echo "HF_HOME: $HF_HOME"
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

echo "---- Main results + generalization (4 models) ----"
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
echo "---- NEW: Sensitivity + stability sweeps (Llama-3.1-8B) ----"
JOB_RHO=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME" slurm_jobs/prune_llm/run_llama3_8b_rho_sweep_array.sh | awk '{print $4}')
echo "  ρ-sensitivity array: $JOB_RHO"

JOB_HALO=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME" slurm_jobs/prune_llm/run_llama3_8b_halo_sweep_array.sh | awk '{print $4}')
echo "  halo (K,η) sensitivity array: $JOB_HALO"

JOB_DOM=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME" slurm_jobs/prune_llm/run_llama3_8b_domain_stability_array.sh | awk '{print $4}')
echo "  domain stability array: $JOB_DOM"

JOB_CALIBSEED=$(sbatch --export=ALL,OUTPUT_BASE="$OUTPUT_BASE",HF_HOME="$HF_HOME" slurm_jobs/prune_llm/run_llama3_8b_calibseed_array.sh | awk '{print $4}')
echo "  within-domain calib-seed stability array: $JOB_CALIBSEED"

echo ""
echo "=============================================="
echo "All PAPER-folder suite jobs submitted"
echo "=============================================="
echo "Monitor with: squeue -u \$USER"
echo ""

