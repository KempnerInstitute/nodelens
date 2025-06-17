# Alignment Benchmarks

This directory contains benchmarking scripts for the Alignment library. These scripts are designed to assess the performance of various components of the codebase.

## Contents

- **benchmark_dropout_strategies.py**: Compares the performance of different dropout strategies (sequential vs. multi-strategy).
- **benchmark_network_training.py**: Benchmarks network training with different configurations.

## Usage

Most benchmark scripts can be run directly from the command line:

```bash
python benchmark_dropout_strategies.py --config configs/config_alignment_experiment.yaml
```

## Adding New Benchmarks

When adding new benchmarks, please follow these guidelines:

1. Include clear documentation within the script about what is being measured
2. Add command-line arguments for configuration
3. Include a simple way to output/visualize the benchmark results
4. Update this README with information about the new benchmark

## Results

Benchmark results should be saved to the `results/` directory, not committed directly to the repository unless they represent an important baseline. 