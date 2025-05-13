#!/bin/bash
# organize_readme_files.sh - Script to organize README files in the repository

echo "Organizing README files..."

# Create necessary directories
mkdir -p doc/metrics doc/performance doc/experiment doc/api doc/tensorized

# Move README files to appropriate locations

# Metrics README
if [ -f "src/alignment/README_metrics.md" ]; then
  echo "Copying metrics documentation..."
  cp src/alignment/README_metrics.md doc/metrics/README.md
fi

# Tensorized Dropout README
if [ -f "README_tensorized_dropout.md" ]; then
  echo "Moving tensorized dropout documentation..."
  cp README_tensorized_dropout.md doc/tensorized/README.md
fi

# If there's a duplicate in src/alignment, fix it
if [ -f "src/alignment/README_TENSORIZED_DROPOUT.md" ]; then
  echo "Updating duplicate tensorized dropout documentation..."
  cp src/alignment/README_TENSORIZED_DROPOUT.md doc/tensorized/IMPLEMENTATION.md
fi

# Update links in the main documentation
echo "Updating documentation links..."

# Create links from main doc to new locations
if [ -f "doc/DOCUMENTATION.md" ]; then
  # Update links in the main documentation
  sed -i 's|Metrics Documentation](metrics/README.md)|Metrics Documentation](metrics/README.md)|g' doc/DOCUMENTATION.md
  sed -i 's|Performance Documentation](performance/README.md)|Performance Documentation](performance/README.md)|g' doc/DOCUMENTATION.md
  sed -i 's|Experiment Documentation](experiment/README.md)|Experiment Documentation](experiment/README.md)|g' doc/DOCUMENTATION.md
fi

# Create documentation index
echo "Creating documentation index..."
cat > doc/README.md << 'EOF'
# Network Alignment Analysis Documentation

## Overview

Welcome to the Network Alignment Analysis documentation. This directory contains comprehensive documentation for the codebase.

## Documentation Structure

- [Main Documentation](DOCUMENTATION.md) - Overview of the entire codebase
- [Metrics](metrics/README.md) - Documentation for the metrics system
- [Experiment Framework](experiment/README.md) - Documentation for the experiment framework
- [Performance Optimizations](performance/README.md) - Documentation for performance optimizations
- [API Reference](api/README.md) - Comprehensive API reference
- [Tensorized Implementations](tensorized/README.md) - Documentation for tensorized implementations

## Guides

- [Usage Guide](usage.md) - How to use the codebase
- [Pruning Modes](pruning_modes.md) - Documentation for different pruning strategies
- [Background](background.md) - Background information on alignment analysis

## Links to Refactoring Summaries

- [Metrics Refactoring Summary](../METRICS_REFACTORING_SUMMARY.md)
- [Codebase Cleanup Summary](../CODEBASE_CLEANUP_SUMMARY.md)
EOF

echo "README organization complete!" 