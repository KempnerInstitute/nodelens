#!/bin/bash
# ============================================================================
# SUBMIT FULL VISION PAPER: ONE ARRAY JOB (MAX 16 GPUs) + DEPENDENT BUILD JOB
# ============================================================================
# Usage:
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
#   bash slurm_jobs/vision_prune/submit_all_array.sh
#
# What this does:
# - Submits ONE array job that runs all suite + appendix runs
# - Caps concurrency to 16 tasks (== 16 GPUs, assuming 1 GPU per task)
# - Schedules build_artifacts.sh after the array completes (afterany)
# ============================================================================

set -euo pipefail

OUTPUT_BASE="${OUTPUT_BASE:-/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red}"

if [[ "$OUTPUT_BASE" != /* ]]; then
  echo "[error] OUTPUT_BASE must be an absolute path. Got: $OUTPUT_BASE"
  echo "[hint] Use: export OUTPUT_BASE=\"/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red\""
  exit 1
fi
mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "Submitting Vision Paper: ALL runs as ONE array"
echo "=============================================="
echo "OUTPUT_BASE: $OUTPUT_BASE"
echo ""

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
mkdir -p logs

export OUTPUT_BASE

# 21 tasks total (0..20). Concurrency cap: 16 GPUs max.
JOB_ALL=$(sbatch --array=0-20%16 --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/run_all_array.sh | awk '{print $4}')
echo "Array job (0-20%16): $JOB_ALL"

# Build job after the array finishes (even if some tasks fail).
JOB_BUILD=$(sbatch --dependency=afterany:${JOB_ALL} --export=ALL,OUTPUT_BASE="$OUTPUT_BASE" slurm_jobs/vision_prune/build_artifacts.sh | awk '{print $4}')
echo "Build all artifacts (afterany:${JOB_ALL}): $JOB_BUILD"

echo ""
echo "=============================================="
echo "Submitted."
echo "=============================================="
echo "Monitor:"
echo "  squeue -u $USER"
echo "  sacct -j ${JOB_ALL} --format=JobID,State,ExitCode,Elapsed"
echo ""

