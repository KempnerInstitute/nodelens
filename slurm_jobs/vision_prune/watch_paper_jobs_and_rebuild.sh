#!/bin/bash
# ============================================================================
# Watch Slurm job arrays and rebuild paper artifacts + PDF when finished.
# ============================================================================
# Usage (recommended):
#   cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
#   bash slurm_jobs/vision_prune/watch_paper_jobs_and_rebuild.sh \
#     --job-ids "56114536,56114539,56114540,56114541,56114543" \
#     --results-base "/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red" \
#     --paper-dir "/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/drafts/alignment_notes"
#
# Logs:
#   /tmp/watch_paper_jobs_and_rebuild.log
#   /tmp/pdflatex_alignment_red_watch.log
# ============================================================================

set -euo pipefail

JOB_IDS="56114536,56114539,56114540,56114541,56114543"
RESULTS_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"
PAPER_DIR="/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/drafts/alignment_notes"
POLL_SECS=90

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-ids) JOB_IDS="$2"; shift 2 ;;
    --results-base) RESULTS_BASE="$2"; shift 2 ;;
    --paper-dir) PAPER_DIR="$2"; shift 2 ;;
    --poll-secs) POLL_SECS="$2"; shift 2 ;;
    *) echo "[error] Unknown arg: $1" ; exit 2 ;;
  esac
done

echo "[watch] job ids: $JOB_IDS"
echo "[watch] results base: $RESULTS_BASE"
echo "[watch] paper dir: $PAPER_DIR"
echo "[watch] poll secs: $POLL_SECS"
echo "[watch] start: $(date)"

# Wait until NONE of the job ids appear in squeue.
while true; do
  if squeue -j "$JOB_IDS" -h 2>/dev/null | grep -q .; then
    echo "[watch] still running/pending: $(date)"
    sleep "$POLL_SECS"
    continue
  fi
  break
done

echo "[watch] all jobs finished: $(date)"
echo "[watch] rebuilding paper artifacts..."

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment
python drafts/alignment_notes/paper/scripts/build_all_artifacts.py \
  --results-base "$RESULTS_BASE" \
  --paper-dir "$PAPER_DIR"

echo "[watch] compiling PDF..."
cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error alignment_red.tex >/tmp/pdflatex_alignment_red_watch.log 2>&1 || (tail -n 160 /tmp/pdflatex_alignment_red_watch.log && exit 1)

echo "[watch] done: $(date)"

