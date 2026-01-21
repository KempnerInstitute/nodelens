#!/bin/bash
# Watcher: wait for AlexNet/ImageNet-100 jobs to finish, then rebuild artifacts
set -euo pipefail

JOB_ID="56192890"
RESULTS_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red/PAPER"
PAPER_DIR="/n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/drafts/alignment_notes"

echo "[watch] waiting for AlexNet job array $JOB_ID to finish..."
while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
    sleep 60
done

echo "[watch] job finished; rebuilding paper artifacts + pdf"

cd /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment

# Activate conda
eval "$(conda shell.bash hook)"
conda activate networkAlignmentAnalysis

# Rebuild artifacts
python drafts/alignment_notes/paper/scripts/build_all_artifacts.py \
    --results-base "$RESULTS_BASE" \
    --paper-dir "$PAPER_DIR"

# Generate professional figures
python drafts/alignment_notes/paper/scripts/generate_professional_figures.py \
    --results-base "$RESULTS_BASE" \
    --paper-dir "$PAPER_DIR"

# Compile PDF
cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode alignment_red.tex > /tmp/pdflatex_alexnet.log 2>&1 || true
pdflatex -interaction=nonstopmode alignment_red.tex > /tmp/pdflatex_alexnet2.log 2>&1 || true

echo "[watch] done"
