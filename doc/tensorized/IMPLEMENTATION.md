# Tensorized Dropout Implementation

## Overview

This document describes the implementation of the tensorized dropout approach in the alignment codebase and the proper organization of code components.

## Key Changes

1. **Implementation of Tensorized Dropout**:
   - Added three implementation strategies in `dropout.py`:
     - Sequential: Process networks one-by-one (original approach)
     - Batched: Process networks in small batches
     - Tensorized: Process all networks at once using tensor operations
   - Made tensorized approach the default in the main experiment code by setting `use_tensorized=True` in `alignment_experiments_preref.py`

2. **Code Organization**:
   - **Proper Separation of Concerns**:
     - Alignment calculations are delegated to the metrics module
     - Dropout implementation focuses on applying dropout and evaluating networks
     - `_compute_alignments` in dropout.py now properly leverages metrics for alignment calculations
     - Benchmarking code is separated into its own module `benchmark_dropout.py`

3. **Function Organization**:
   - `progressive_dropout`: Main entry point for dropout experiments
   - Helper functions with clear responsibilities:
     - `_compute_alignments`: Computes alignment values for each layer in each network
     - `_compute_dropout_indices`: Determines which neurons to drop based on alignment values
     - `_evaluate_networks_sequentially`: Sequential evaluation implementation
     - `_evaluate_networks_batched`: Batched evaluation implementation
     - `_evaluate_networks_tensorized`: Tensorized evaluation implementation

## Performance Benefits

The tensorized approach provides significant performance benefits:
- 5.3x speedup compared to the batched approach
- Most improvement comes from the network evaluation phase (6.4x faster)

## How to Use

The tensorized implementation is now the default in the main experiment code. To explicitly control the approach:

```python
# In alignment_experiments_preref.py:
results = progressive_dropout(
    networks,
    dataset,
    dropout_fractions=dropout_fractions,
    metric=self.metric,
    device=self.device,
    pruning_mode=pruning_mode,
    dropout_mode=dropout_mode,
    use_tensorized=True  # Set to False to use sequential or batched approach
)
```

For benchmarking different approaches:

```bash
python src/alignment/experiments/benchmark_dropout.py --config configs/config_alignment_experiment.yaml \
    --approaches original batched tensorized \
    --num-networks 10 \
    --num-runs 3 \
    --num-batches 5
```

## Implementation Details

The tensorized approach works by:
1. Computing alignment values for each layer in each network
2. Determining which neurons to drop based on alignment values
3. Processing all networks in parallel using tensor operations during evaluation

This approach minimizes loops and maximizes the use of GPU parallelism, resulting in significant performance improvements. 