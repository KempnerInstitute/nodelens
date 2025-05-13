# Network Alignment Analysis Documentation

This document provides a comprehensive index to all documentation for the Network Alignment Analysis codebase.

## Getting Started

- [README](README.md) - Main repository README with setup instructions
- [Usage Guide](doc/usage.md) - How to use the codebase
- [Installation Guide](README.md#installation) - How to install the package

## Core Documentation

- [Main Documentation](doc/DOCUMENTATION.md) - Overview of codebase architecture and capabilities
- [API Reference](doc/api/README.md) - Comprehensive API reference
- [Metrics System](doc/metrics/README.md) - Documentation for the metrics system
- [Experiment Framework](doc/experiment/README.md) - Documentation for the experiment framework
- [Performance Optimizations](doc/performance/README.md) - Documentation for performance optimizations
- [Tensorized Implementations](doc/tensorized/README.md) - Documentation for tensorized implementations

## Guides and References

- [Pruning Modes](doc/pruning_modes.md) - Documentation for different pruning strategies
- [Background](doc/background.md) - Background information on alignment analysis
- [Development Roadmap](doc/ROADMAP.md) - Future development plans

## Refactoring and Cleanup

- [Metrics Refactoring Summary](METRICS_REFACTORING_SUMMARY.md) - Summary of metrics system refactoring
- [Codebase Cleanup Summary](CODEBASE_CLEANUP_SUMMARY.md) - Summary of codebase cleanup

## Directory Structure

The codebase is organized into the following directories:

- `src/alignment/`: Core source code implementing alignment metrics and algorithms
  - `metrics.py`: Implementation of all alignment metrics
  - `utils/`: Utility functions for data handling, visualization, etc.
  - `experiment/`: Experiment framework for running alignment experiments
  - `networks/`: Network architectures and training utilities
  - `pruning/`: Implementation of various pruning strategies

- `tests/`: Unit and integration tests
- `scripts/`: Utility scripts for running experiments and analysis
- `benchmarks/`: Performance evaluation scripts
- `configs/`: Configuration files for experiments
- `results/`: Output directory for experiment results
- `doc/`: Documentation

## Additional Resources

- [CHANGELOG.md](CHANGELOG.md) - History of changes to the codebase
- [LICENSE](LICENSE) - MIT License for the project 