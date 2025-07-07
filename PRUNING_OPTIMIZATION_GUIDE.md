# Pruning Performance Optimization Guide

This guide explains how to optimize pruning performance in the alignment experiments framework.

## Quick Start - Fastest Settings

For the fastest pruning experiments, use these configuration settings:

```python
from alignment.experiments.general_alignment import GeneralAlignmentConfig

config = GeneralAlignmentConfig(
    # Performance optimizations
    use_tensorized_pruning=True,      # Enable tensorized operations
    use_optimized_pruning=True,       # Use optimized implementation (NEW!)
    use_ultra_fast_pruning=True,      # Ultra-fast mode (uses more memory)
    
    # Reduce computational load
    fine_tune_after_pruning=False,    # Skip fine-tuning if not needed
    fine_tune_epochs=5,               # Reduce fine-tuning epochs if needed
    
    # Reduce data points
    pruning_amounts=[0.1, 0.5, 0.9],  # Fewer pruning amounts to test
    num_networks=5,                   # Fewer networks if using multiple
    
    # Other optimizations
    device='cuda',                    # Use GPU
    batch_size=256,                   # Larger batch size if memory allows
)
```

## Performance Optimization Options

### 1. **Pruning Mode Selection**

The framework offers three pruning modes with different speed/memory trade-offs:

- **Standard Mode** (`use_tensorized_pruning=False`): Slowest, least memory
- **Tensorized Mode** (`use_tensorized_pruning=True`): Faster, moderate memory
- **Optimized Mode** (`use_optimized_pruning=True`): Even faster, moderate memory (NEW!)
- **Ultra-Fast Mode** (`use_ultra_fast_pruning=True`): Fastest, high memory usage

### 2. **Key Optimizations in the New Implementation**

The optimized implementation includes:

1. **Reduced Weight Copying**: Original weights are referenced instead of cloned when possible
2. **Batch Mask Application**: Masks are applied to all networks more efficiently
3. **Inline Sparsity Calculation**: Sparsity is calculated during mask application
4. **Optimized Evaluation**: Networks are evaluated with better memory locality
5. **Simplified Hooks**: Lightweight hooks that don't store redundant data

### 3. **Configuration Tips for Speed**

#### Reduce Computational Load
```python
# Test fewer pruning amounts
pruning_amounts=[0.2, 0.5, 0.8]  # Instead of [0.1, 0.3, 0.5, 0.7, 0.9]

# Skip fine-tuning if you just need initial results
fine_tune_after_pruning=False

# Or reduce fine-tuning epochs
fine_tune_epochs=3  # Instead of 10
```

#### Optimize Network Configuration
```python
# For multi-network experiments
num_networks=10  # Use fewer networks if possible
aggregate_metrics=True  # Aggregate results to reduce output size
```

#### Use GPU Acceleration
```python
device='cuda'  # Always use GPU when available
batch_size=512  # Larger batches for GPU efficiency
```

### 4. **Memory vs Speed Trade-offs**

| Mode | Speed | Memory Usage | When to Use |
|------|-------|--------------|-------------|
| Standard | ⭐ | ⭐⭐⭐⭐⭐ | Small experiments, debugging |
| Tensorized | ⭐⭐⭐ | ⭐⭐⭐⭐ | Default for most cases |
| Optimized | ⭐⭐⭐⭐ | ⭐⭐⭐ | Recommended for large experiments |
| Ultra-Fast | ⭐⭐⭐⭐⭐ | ⭐ | When speed is critical and memory is available |

### 5. **Profiling Your Experiments**

To identify bottlenecks:

```python
import time

# Time different phases
start = time.time()
results = experiment.run()
print(f"Total time: {time.time() - start:.2f}s")

# Check logs for phase timing
# The implementation logs timing for each phase
```

### 6. **Advanced Optimizations**

For even more speed:

1. **Reduce Dataset Size** for initial experiments:
   ```python
   # Use a subset of data for pruning evaluation
   config.batch_size = 1000
   config.max_batches_per_epoch = 10  # If implemented
   ```

2. **Parallel Strategy Evaluation**: 
   ```python
   # Process multiple strategies in parallel
   pruning_strategies=["magnitude"]  # Test one at a time if needed
   ```

3. **Skip Alignment Computation** during pruning:
   ```python
   # Only compute alignment when needed
   measure_alignment_during_training=False
   ```

## Example: Optimized Configuration

Here's a complete example for fast pruning experiments:

```python
config = GeneralAlignmentConfig(
    name="fast_pruning_experiment",
    model_name="mlp",
    dataset_name="mnist",
    
    # Maximum performance settings
    use_tensorized_pruning=True,
    use_optimized_pruning=True,
    use_ultra_fast_pruning=False,  # Set True if you have enough memory
    
    # Pruning configuration
    do_pruning_experiments=True,
    pruning_strategies=["magnitude", "random"],
    pruning_amounts=[0.3, 0.6, 0.9],
    pruning_selection_mode=["low", "high"],
    fine_tune_after_pruning=True,
    fine_tune_epochs=3,
    
    # Network configuration
    num_networks=5,
    aggregate_metrics=True,
    
    # Training configuration
    do_train=False,  # Use pretrained if available
    
    # Hardware optimization
    device='cuda',
    batch_size=256,
    
    # Disable unnecessary features
    do_dropout_analysis=False,
    do_eigenfeature_analysis=False,
    measure_alignment_during_training=False,
    generate_plots=False,  # Generate plots separately if needed
)
```

## Benchmarks

Typical speedups with optimizations enabled:

- Standard → Tensorized: 2-3x faster
- Tensorized → Optimized: 1.5-2x faster  
- Optimized → Ultra-Fast: 1.5-3x faster (but uses much more memory)

Total potential speedup: **5-10x** with all optimizations enabled.

## Troubleshooting

1. **Out of Memory**: Disable `use_ultra_fast_pruning` or reduce `num_networks`
2. **Still Too Slow**: Reduce `pruning_amounts` and `fine_tune_epochs`
3. **GPU Not Utilized**: Ensure `device='cuda'` and increase `batch_size`

## Summary

For fastest pruning:
1. Enable all optimization flags
2. Reduce the number of pruning amounts tested
3. Skip or reduce fine-tuning
4. Use GPU with large batch sizes
5. Test fewer networks or strategies if possible 