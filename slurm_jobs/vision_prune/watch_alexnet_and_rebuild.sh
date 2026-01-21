#!/bin/bash
# ==============================================================================
# Watch AlexNet jobs and automatically rebuild paper artifacts when done
# ==============================================================================
# Usage: ./watch_alexnet_and_rebuild.sh 56159638
# ==============================================================================

set -euo pipefail

JOB_ID="${1:-56159638}"
POLL_INTERVAL=60  # Check every 60 seconds
MAX_WAIT=18000    # Max 5 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PAPER_DIR="$REPO_ROOT/drafts/alignment_notes"
OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/alignment_red"

echo "============================================================"
echo "Watching AlexNet job array: $JOB_ID"
echo "Will rebuild paper artifacts when all array tasks complete"
echo "============================================================"
echo "Poll interval: ${POLL_INTERVAL}s, Max wait: ${MAX_WAIT}s"
echo ""

wait_time=0
while [ $wait_time -lt $MAX_WAIT ]; do
    # Check job status
    status=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | head -n 1 | tr -d ' ')
    
    # Count running/pending jobs
    running=$(squeue -j "$JOB_ID" --noheader 2>/dev/null | wc -l || echo "0")
    
    if [ "$running" -eq 0 ]; then
        echo ""
        echo "[$(date)] All jobs completed!"
        
        # Check for any failures
        failed=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | grep -c FAILED || echo "0")
        completed=$(sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null | grep -c COMPLETED || echo "0")
        
        echo "  Completed: $completed, Failed: $failed"
        
        if [ "$failed" -gt 0 ]; then
            echo "[WARN] Some jobs failed. Check logs at:"
            echo "  $SCRIPT_DIR/logs/vision_alexnet_*.err"
        fi
        
        if [ "$completed" -gt 0 ]; then
            echo ""
            echo "============================================================"
            echo "Rebuilding paper artifacts..."
            echo "============================================================"
            
            cd "$REPO_ROOT"
            
            # Activate conda
            eval "$(conda shell.bash hook)"
            conda activate networkAlignmentAnalysis
            
            # Rebuild artifacts
            echo "[1/4] Running build_all_artifacts.py..."
            python "$PAPER_DIR/paper/scripts/build_all_artifacts.py" \
                --output-base "$OUTPUT_BASE" \
                --paper-dir "$PAPER_DIR" \
                --prefer-paper-folder 2>&1 | tail -n 30 || true
            
            echo ""
            echo "[2/4] Generating professional figures..."
            python "$PAPER_DIR/paper/scripts/generate_professional_figures.py" \
                --results-base "$OUTPUT_BASE/PAPER" \
                --paper-dir "$PAPER_DIR" 2>&1 || true
            
            echo ""
            echo "[3/4] Generating kernel visualization..."
            python "$PAPER_DIR/paper/scripts/generate_kernel_visualization.py" \
                --results-base "$OUTPUT_BASE/PAPER" \
                --paper-dir "$PAPER_DIR" \
                --exp-prefix "alexnet_cifar10" 2>&1 || true
            
            echo ""
            echo "[4/4] Compiling LaTeX..."
            cd "$PAPER_DIR"
            pdflatex -interaction=nonstopmode alignment_red.tex > /dev/null 2>&1 || true
            bibtex alignment_red > /dev/null 2>&1 || true
            pdflatex -interaction=nonstopmode alignment_red.tex > /dev/null 2>&1 || true
            pdflatex -interaction=nonstopmode alignment_red.tex > /dev/null 2>&1 || true
            
            echo ""
            echo "============================================================"
            echo "Done! Paper PDF updated: $PAPER_DIR/alignment_red.pdf"
            echo "============================================================"
        fi
        
        break
    fi
    
    echo -n "."
    sleep $POLL_INTERVAL
    wait_time=$((wait_time + POLL_INTERVAL))
done

if [ $wait_time -ge $MAX_WAIT ]; then
    echo ""
    echo "[TIMEOUT] Maximum wait time reached. Jobs may still be running."
    echo "Check with: squeue -j $JOB_ID"
fi
