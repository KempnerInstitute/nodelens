# Tensorized Progressive Dropout Implementation

This document describes the improvements made to the progressive dropout implementation in the alignment codebase, along with benchmark results comparing different approaches.

## Overview

Progressive dropout is a technique used to analyze the alignment of neural networks by gradually dropping out neurons and measuring the impact on performance. The original implementation processed networks sequentially, which was inefficient for large numbers of networks. Two optimized implementations were developed:

1. **Batched approach**: Process networks in small batches
2. **Tensorized approach**: Process networks using tensor operations for maximum parallelism

## Implementation Details

### Key Changes

- Used proper separation of concerns:
  - Alignment calculations are delegated to the metrics module
  - Dropout implementation focuses on applying dropout and evaluating networks
  - Benchmarking code is separated into its own module

- Added three implementation strategies:
  - **Sequential**: Process one network at a time (original approach)
  - **Batched**: Process networks in small batches of configurable size
  - **Tensorized**: Process all networks at once using tensor operations

- Added detailed timing for performance analysis

### Timing Breakdown

The benchmark results show the time spent in each phase of the dropout process:

1. **Alignment computation**: Computing alignment values for each layer in each network
2. **Dropout indices computation**: Determining which neurons to drop based on alignment values
3. **Network evaluation**: Applying dropout and evaluating networks on the dataset

## Benchmark Results

Running with 8 networks, 5 data batches, and 1 run:

| Approach   | Average Time (s) | Speedup vs. Batched |
|------------|------------------|---------------------|
| Batched    | 169.90 ± 0.00    | 1.00x               |
| Tensorized | 32.03 ± 0.00     | 5.30x               |

### Detailed Timing Breakdown

| Operation                  | Batched (s) | Tensorized (s) |
|----------------------------|-------------|----------------|
| Alignment computation      | 6.44        | 6.43           |
| Dropout indices computation| 0.02        | 0.02           |
| Network evaluation         | 163.44      | 25.58          |
| Total time                 | 169.90      | 32.03          |

## Conclusion

The tensorized approach provides a significant speedup (5.3x) compared to the batched approach, with most of the improvement coming from the network evaluation phase. This makes it much more efficient for experiments involving large numbers of networks.

## Usage

To use the optimized progressive dropout implementation:

```python
from alignment.dropout import progressive_dropout

results = progressive_dropout(
    networks,
    dataset,
    dropout_fractions=np.linspace(0.1, 0.9, 9),
    metric=metric,
    device="cuda",
    use_tensorized=True  # Use the tensorized approach
)
```

To run the benchmark:

```bash
python src/alignment/experiments/benchmark_dropout.py --config configs/config_alignment_experiment.yaml \
    --approaches original batched tensorized \
    --num-networks 10 \
    --num-runs 3 \
    --num-batches 5
``` 