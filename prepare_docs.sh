#!/bin/bash
# prepare_docs.sh - Script to prepare documentation for MkDocs

echo "Preparing documentation for MkDocs..."

# Create documentation.md from DOCUMENTATION.md
cp doc/DOCUMENTATION.md docs/documentation.md

# Copy README files with renamed paths
cp doc/metrics/README.md docs/metrics/README.md
cp doc/experiment/README.md docs/experiment/README.md
cp doc/performance/README.md docs/performance/README.md
cp doc/api/README.md docs/api/README.md
cp doc/tensorized/README.md docs/tensorized/README.md

# Copy guides
cp doc/usage.md docs/usage.md
cp doc/pruning_modes.md docs/pruning_modes.md
cp doc/background.md docs/background.md
cp doc/ROADMAP.md docs/roadmap.md

# Copy summaries
mkdir -p docs/summaries
cp doc/summaries/DOCUMENTATION_SUMMARY.md docs/summaries/documentation_summary.md
cp doc/summaries/CODEBASE_CLEANUP_SUMMARY.md docs/summaries/codebase_cleanup_summary.md
cp doc/summaries/METRICS_REFACTORING_SUMMARY.md docs/summaries/metrics_refactoring_summary.md

# Fix internal links in copied files
find docs -type f -name "*.md" -exec sed -i 's|(../|(/|g' {} \;
find docs -type f -name "*.md" -exec sed -i 's|](/|](|g' {} \;
find docs -type f -name "*.md" -exec sed -i 's|.md)|)|g' {} \;
find docs -type f -name "*.md" -exec sed -i 's|.md#|#|g' {} \;

echo "Documentation prepared for MkDocs!" 