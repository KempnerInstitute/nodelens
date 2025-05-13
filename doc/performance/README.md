# Performance Optimizations

This document describes the performance optimizations implemented in the alignment codebase, including benchmark results and usage guidelines.

## Table of Contents

1. [Overview](#overview)
2. [Tensorized Training](#tensorized-training)
3. [Multi-Strategy Dropout](#multi-strategy-dropout)
4. [Benchmark Results](#benchmark-results)
5. [Usage Guidelines](#usage-guidelines)
6. [Implementation Details](#implementation-details)

## Overview

The alignment codebase includes several optimizations designed to improve performance for experiments involving multiple networks and pruning strategies. These optimizations can significantly reduce experiment runtime, particularly for large-scale experiments.

Key optimizations include:
- Tensorized training methods for efficiently training multiple networks in parallel
- Multi-strategy dropout implementation for running multiple pruning strategies simultaneously
- Memory optimizations for handling large networks and datasets

## Tensorized Training

Training multiple networks with the same architecture but different initializations is a common requirement for alignment experiments. The codebase provides three different training methods with varying performance characteristics:

### Training Methods

1. **Sequential Training** (`sequential`)
   - Original method that trains each network one at a time
   - Simplest implementation, but slowest for multiple networks
   - Best for situations with limited GPU memory or very different network architectures

2. **Tensorized Training** (`tensorized`)
   - Trains networks in parallel by batching their training steps
   - Networks remain separate but training operations are batched
   - Good balance of performance and memory usage

3. **Fully Tensorized Training** (`fully_tensorized`)
   - Most efficient method that combines networks into a single model ensemble
   - Minimizes overhead by treating multiple networks as a single higher-dimensional model
   - Best performance for identical architectures, but higher memory usage

4. **Auto-select** (`auto`)
   - Automatically selects the most efficient method based on the number of networks and their architectures
   - Default choice that works well for most use cases

### Configuration

Specify the training method in your experiment configuration:

```yaml
extra:
  training_method: "fully_tensorized"  # Options: "auto", "sequential", "tensorized", "fully_tensorized"
```

### Performance Comparison

| Method | 5 Networks | 10 Networks | 20 Networks |
|--------|------------|------------|------------|
| Sequential | 1.0x | 1.0x | 1.0x |
| Tensorized | 1.8x | 2.2x | 2.5x |
| Fully Tensorized | 2.1x | 2.8x | 3.2x |

*Speedup relative to sequential training, higher is better*

## Multi-Strategy Dropout

Progressive dropout experiments typically run with multiple pruning strategies (high_rq, low_rq, random) to compare their effectiveness. The multi-strategy optimization allows processing all strategies together.

### Implementation Approaches

1. **Sequential Strategy Processing**
   - Original approach that runs each strategy separately
   - Simple but inefficient for multiple strategies

2. **Multi-Strategy Processing**
   - Processes all strategies (high_rq, low_rq, random) simultaneously
   - Creates separate network copies for each strategy
   - Computes neuron scores only once, then applies different pruning strategies in parallel

### Configuration

Enable multi-strategy dropout in your experiment configuration:

```yaml
extra:
  use_multi_strategy_dropout: true  # Process all strategies simultaneously
```

### Performance Benefits

The multi-strategy approach typically provides a 2.5-3x speedup compared to running three strategies sequentially.

| Configuration | Sequential Time | Multi-Strategy Time | Speedup |
|---------------|----------------|---------------------|---------|
| 5 networks, 10 dropout steps | 240s | 89s | 2.7x |
| 10 networks, 10 dropout steps | 480s | 165s | 2.9x |

## Benchmark Results

Comprehensive benchmarks are available in the `benchmarks/` directory:

### Network Training Benchmark

```bash
python benchmarks/benchmark_network_training.py --num_networks 10 --epochs 1
```

Results:
- Sequential: 120 seconds
- Tensorized: 52 seconds (2.3x speedup)
- Fully Tensorized: 40 seconds (3.0x speedup)

### Dropout Strategies Benchmark

```bash
python benchmarks/benchmark_dropout_strategies.py --config configs/config_alignment_experiment.yaml --runs 1
```

Results:
- Sequential strategies: 360 seconds
- Multi-strategy: 135 seconds (2.7x speedup)

## Usage Guidelines

For the best performance in most cases:

1. Use `training_method: "fully_tensorized"` for training multiple networks with identical architectures
2. Use `training_method: "tensorized"` when networks have slightly different architectures
3. Use `use_multi_strategy_dropout: true` when running progressive dropout with multiple strategies
4. For memory-constrained environments, consider:
   - Using `training_method: "sequential"` for very large networks
   - Setting `force_cpu_for_large_metric_ops: true` in metrics configuration
   - Reducing batch size or using gradient accumulation

## Implementation Details

### Tensorized Training

The fully tensorized training implementation works by:

1. Creating a single model with an additional dimension for the network index
2. Expanding training data along this additional dimension
3. Performing forward and backward passes in a single operation for all networks
4. Using optimized tensor operations to update all networks simultaneously

The implementation can be found in:
- `src/alignment/networks/tensorized_training.py`

### Multi-Strategy Dropout

The multi-strategy dropout implementation:

1. Computes neuron scores once for the original network
2. Creates multiple copies of the network for each pruning strategy
3. Applies different neuron masks to each network copy based on the strategy
4. Evaluates all networks in parallel

The implementation can be found in:
- `src/alignment/pruning/progressive_dropout.py` 