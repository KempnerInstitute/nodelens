# Alignment Scripts

This directory contains utility scripts for the Alignment library. These scripts provide various functionality for running experiments, demonstrations, and utilities.

## Contents

- **direct_pruning_test.py**: Tests pruning functionality directly without relying on the experiment infrastructure.
- **run_multi_strategy_experiment.py**: Runs experiments with multiple pruning strategies.
- **run_fixed_experiment.py**: Runs experiments with fixed configurations.
- **run_cascading_with_plots.py**: Runs cascading pruning experiments and generates plots.
- **run_cascading_test.py**: Tests cascading pruning methodology.
- **continual_mnist.py**: Implementation of continual learning on MNIST.
- **teacher_student.py**: Implementation of teacher-student neural network model.

## Shell Scripts

- **run_benchmark.sh**: Convenience script for running benchmarks.
- **run_cascading_test.sh**: Shell script for testing cascading pruning.

## Usage

Most scripts have command-line arguments for configuration:

```bash
python scripts/run_fixed_experiment.py --config configs/config_alignment_experiment.yaml
```

## Adding New Scripts

When adding new scripts, please follow these guidelines:

1. Include clear documentation at the top of the script about its purpose and usage
2. Add command-line arguments for configuration using argparse
3. Include appropriate error handling and logging
4. Update this README with information about the new script 