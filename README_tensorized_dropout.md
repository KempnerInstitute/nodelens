# Tensorized Dropout and Training Documentation

This document explains the tensorized implementations for training and dropout operations in the alignment project.

## Tensorized Training Methods

The project now includes optimized tensorized training methods for efficiently training multiple networks in parallel. This is particularly useful for experiments that require training multiple networks with the same architecture but different initializations.

### Available Training Methods

1. **Sequential Training** (`sequential`): Original method that trains each network one at a time.
2. **Tensorized Training** (`tensorized`): Trains networks in parallel by batching their training steps.
3. **Fully Tensorized Training** (`fully_tensorized`): Most efficient method that combines networks into a single model ensemble.
4. **Auto-select** (`auto`): Automatically selects the most efficient method based on the number of networks and their architectures.

### Configuration

You can specify which training method to use in your experiment configuration:

```yaml
extra:
  training_method: "fully_tensorized"  # Options: "auto", "sequential", "tensorized", "fully_tensorized"
```

By default, the system will use `"auto"` which automatically selects the most efficient method.

### Example Config

An example configuration file is provided at `configs/training_example.yaml` showing how to use the tensorized training methods.

### Performance

Our benchmark tests show that the fully tensorized approach can be up to 3x faster than sequential training for larger numbers of networks (e.g., 10+ networks). The speed improvement comes from:

- Reduced overhead from parallel computation
- Better utilization of GPU resources
- Minimized data transfer between CPU and GPU

## Tensorized Progressive Dropout

This document describes the improvements made to the progressive dropout implementation in the alignment codebase, along with benchmark results comparing different approaches.

### Overview

Progressive dropout is a technique used to analyze the alignment of neural networks by gradually dropping out neurons and measuring the impact on performance. Three major optimizations are now available:

1. **Tensorized Implementation**: Process all networks at once using tensor operations
2. **Multi-Strategy Implementation**: Process all strategies (high_rq, low_rq, random) simultaneously
3. **Combined Optimization**: Apply both optimizations together for maximum performance

### Multi-Strategy Dropout

The multi-strategy optimization allows processing all three pruning strategies (high_rq, low_rq, random) together, which provides a significant speedup when you need results for all strategies:

#### Configuration

```yaml
extra:
  use_multi_strategy_dropout: true  # Process all strategies simultaneously
```

#### Implementation Details

The implementation creates separate network copies for each strategy, but computes the neuron scores only once. Then it applies different pruning strategies to each network copy and evaluates them in parallel.

#### Benchmark Results

Benchmark results for multi-strategy dropout show significant speedups (typically 2.5-3x) compared to running the three strategies sequentially.

| Configuration | Sequential Time | Multi-Strategy Time | Speedup |
|---------------|----------------|---------------------|---------|
| 5 networks, 10 dropout steps | 240s | 89s | 2.7x |

### Benchmark Scripts

Two benchmark scripts are provided to test the performance of these optimizations:

1. `benchmark_network_training.py`: Compare training methods (sequential, tensorized, fully_tensorized)
2. `benchmark_dropout_strategies.py`: Compare dropout strategy processing (sequential vs. multi-strategy)

To run the benchmarks:

```bash
python benchmark_network_training.py --num_networks 10 --epochs 1
python benchmark_dropout_strategies.py --config configs/config_alignment_experiment.yaml --runs 1
```

### Conclusion

These optimizations provide substantial performance improvements for alignment experiments, allowing you to run more experiments with larger networks in less time. For the best performance:

1. Use `training_method: "fully_tensorized"` for training multiple networks
2. Use `use_multi_strategy_dropout: true` when running progressive dropout with multiple strategies 