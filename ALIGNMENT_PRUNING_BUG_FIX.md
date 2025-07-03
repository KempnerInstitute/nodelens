# Alignment-Based Pruning Bug Fix

## Summary

A critical bug was discovered in the alignment-based pruning implementation that caused networks to catastrophically fail even with minimal pruning (5%). The issue was that **all layers were receiving the raw input data (e.g., MNIST images) instead of their actual layer inputs** when computing alignment scores for pruning decisions.

## The Problem

In `src/alignment/experiments/general_alignment.py`, when performing layer-wise alignment-based pruning:

1. Layer inputs were only captured for global pruning mode
2. For layer-wise pruning, if a layer's inputs weren't captured, the code fell back to using `sample_inputs` (the raw MNIST data)
3. This meant every layer beyond the first was computing alignment scores using 784-dimensional MNIST inputs instead of their actual inputs

### Example of the Issue

For a network with architecture: 784 → 512 → 256 → 128 → 64 → 32 → 16 → 10

- Layer 1 (784→512): Correctly received 784-dim MNIST inputs ✓
- Layer 2 (512→256): Incorrectly received 784-dim MNIST inputs instead of 512-dim outputs from Layer 1 ✗
- Layer 3 (256→128): Incorrectly received 784-dim MNIST inputs instead of 256-dim outputs from Layer 2 ✗
- And so on...

This caused:
- Dimension mismatches (seen in logs as "RQ: Dimension mismatch" warnings)
- Meaningless alignment scores
- Random pruning decisions
- Network accuracy dropping to ~10% (random chance) with just 5% pruning

## The Fix

The fix involves two changes in `general_alignment.py`:

### 1. Always Capture Layer Inputs for Alignment Pruning

**Before:**
```python
# For global alignment pruning, we need inputs for all layers
if pruning_config.global_pruning and strategy_name == "alignment":
    # Capture inputs with hooks...
```

**After:**
```python
# For alignment-based pruning, we ALWAYS need inputs for all layers
# (not just for global pruning)
# Use hooks to capture inputs for all layers
```

### 2. Don't Fall Back to Wrong Inputs

**Before:**
```python
if name in layer_inputs_dict:
    layer_inputs = layer_inputs_dict[name]
else:
    # Fallback to sample inputs (less accurate)
    layer_inputs = sample_inputs
```

**After:**
```python
if name in layer_inputs_dict:
    layer_inputs = layer_inputs_dict[name]
else:
    # This should not happen if hooks worked correctly
    logger.error(f"No captured inputs for layer {name} - this will cause incorrect pruning!")
    continue  # Skip this layer rather than using wrong inputs
```

## Impact

With this fix:
- Each layer receives its correct input tensors
- Alignment scores are computed properly
- Pruning decisions are based on actual neuron-input alignment
- Networks should maintain reasonable accuracy with small pruning amounts

## Testing the Fix

To verify the fix works:

1. Run an alignment pruning experiment with small sparsity levels (5%, 10%, etc.)
2. Check that there are no "dimension mismatch" warnings in the logs
3. Verify that accuracy doesn't catastrophically drop with minimal pruning
4. For a 576K parameter network, 5% pruning should have minimal impact on accuracy

## Files Modified

- `src/alignment/experiments/general_alignment.py`: Fixed input capture for layer-wise alignment pruning 