# Quick Pruning Speedup Tips

## Fastest Configuration

```python
config = GeneralAlignmentConfig(
    # Enable all optimizations
    use_tensorized_pruning=True,
    use_optimized_pruning=True,
    use_ultra_fast_pruning=True,  # If you have memory
    
    # Reduce computation
    pruning_amounts=[0.3, 0.6, 0.9],  # Fewer amounts
    fine_tune_after_pruning=False,     # Skip if not needed
    fine_tune_epochs=3,                # Or reduce epochs
    
    # Hardware
    device='cuda',
    batch_size=512,
)
```

## Key Speedups

1. **New Optimized Mode**: `use_optimized_pruning=True` - 1.5-2x faster
2. **Ultra-Fast Mode**: `use_ultra_fast_pruning=True` - Up to 3x faster (high memory)
3. **Skip Fine-tuning**: `fine_tune_after_pruning=False` - Saves significant time
4. **Fewer Pruning Amounts**: Test 3-5 instead of many
5. **GPU + Large Batches**: Essential for speed

Total speedup possible: **5-10x** 