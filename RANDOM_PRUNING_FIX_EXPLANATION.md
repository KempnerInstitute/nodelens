# Random Pruning Fix Explanation

## The Issue

You observed that "random" pruning was performing much better than expected, which indicated a logic error in the implementation.

## What Was Wrong

The original implementation had a conceptual confusion between **pruning strategies** and **selection modes**:

1. **Pruning Strategies** determine WHAT metric to use for importance:
   - `magnitude`: Use weight magnitude as importance
   - `alignment`: Use alignment scores as importance  
   - `random`: Don't use any metric (pure random)

2. **Selection Modes** determine HOW to select weights based on importance:
   - `low`: Select weights with LOW importance scores
   - `high`: Select weights with HIGH importance scores
   - `random`: Select weights randomly (ignore importance)

### The Problem

The code was treating "random" as only a strategy, not as a selection mode. This meant:
- When using `strategy="random"`, it ALWAYS selected randomly (correct)
- When using `strategy="magnitude"` with `selection_mode="random"`, it was NOT selecting randomly (incorrect!)

This created an unfair comparison because:
- The "random" strategy was truly random
- The "magnitude" strategy with "random" selection was still using magnitude-based selection

## The Fix

I've updated the code to properly handle random selection for ALL strategies:

```python
# OLD LOGIC (incorrect):
if strategy_name == "random":
    # Always random, ignore selection_mode
    use_random_selection()
else:
    # Use importance scores with selection_mode
    use_importance_based_selection()

# NEW LOGIC (correct):
if strategy_name == "magnitude" and selection_mode == "random":
    # Special case: ignore magnitude, use random selection
    use_random_selection()
elif strategy_name == "random":
    # Pure random strategy: always random
    use_random_selection()
else:
    # Use importance scores with selection_mode
    use_importance_based_selection()
```

## What This Means

Now the comparisons are fair:

1. **Magnitude + Low**: Prune weights with LOW magnitude (keep high magnitude)
2. **Magnitude + High**: Prune weights with HIGH magnitude (keep low magnitude)
3. **Magnitude + Random**: Prune random weights (ignore magnitude completely)
4. **Random strategy**: Same as magnitude + random (for backwards compatibility)
5. **Alignment + Low/High/Random**: Same logic but using alignment scores

## Expected Results

With this fix, you should see:

- **Magnitude + Low**: Best performance (keeps important high-magnitude weights)
- **Magnitude + High**: Worst performance (removes important high-magnitude weights)
- **Magnitude + Random**: Medium performance (random selection)
- **Random strategy**: Same as Magnitude + Random

If random is still performing better than Magnitude + Low, it might indicate:
1. The network has redundancy and random pruning is sufficient
2. Magnitude might not be the best importance metric for your specific model
3. The network might benefit from the regularization effect of random pruning

## Configuration

In your YAML config, you can now properly test all combinations:

```yaml
algorithms: ["magnitude", "alignment"]  
selection_strategies: ["low", "high", "random"]
```

This will test:
- Magnitude with low selection (prune unimportant)
- Magnitude with high selection (prune important) 
- Magnitude with random selection (baseline)
- Alignment with low selection
- Alignment with high selection
- Alignment with random selection

## Additional Issue: Structured vs Unstructured Random Pruning

There's another important issue to consider: **random pruning doesn't properly handle structured (node/neuron) vs unstructured (weight) pruning**.

### The Problem with Current Random Implementation

When using structured pruning (e.g., pruning entire neurons/channels), the random strategy should:
- Select **entire nodes/neurons** randomly, not individual weights
- Respect the pruning granularity (node-level vs weight-level)

Currently, the random mask creation is too simplistic:

```python
def _create_random_mask(self, shape: torch.Size, amount: float) -> torch.Tensor:
    """Create a random pruning mask."""
    mask = torch.rand(shape) > amount
    return mask.float()
```

This creates a weight-wise random mask, even when structured pruning is requested!

### What Should Happen

For **structured pruning with random selection**:
1. Determine the number of nodes/neurons to prune based on the amount
2. Randomly select which nodes to prune
3. Create a mask that prunes ALL weights connected to selected nodes

For **unstructured pruning with random selection**:
1. Current behavior is correct - randomly select individual weights

### Example of Correct Structured Random Pruning

```python
def create_structured_random_mask(weights, amount, dim=0):
    """Create random mask for structured pruning."""
    # Get number of structures (e.g., neurons)
    num_structures = weights.shape[dim]
    num_to_prune = int(amount * num_structures)
    
    # Randomly select which structures to prune
    indices = torch.randperm(num_structures)[:num_to_prune]
    
    # Create mask (1 = keep, 0 = prune)
    mask = torch.ones(num_structures, dtype=torch.bool)
    mask[indices] = False
    
    # Expand mask to match weight dimensions
    if len(weights.shape) == 2:  # Linear layer
        if dim == 0:  # Prune output neurons
            mask = mask.unsqueeze(1).expand_as(weights)
        else:  # Prune input neurons
            mask = mask.unsqueeze(0).expand_as(weights)
    elif len(weights.shape) == 4:  # Conv layer
        # Expand for conv layers (out_channels, in_channels, H, W)
        mask = mask.view(-1, 1, 1, 1).expand_as(weights)
    
    return mask.float()
```

### Impact

This issue means that when comparing:
- **Alignment-based pruning** (which is inherently structured/neuron-based)
- **Random pruning** (which currently only does weight-wise random)

The comparison isn't fair because they're operating at different granularities. The alignment method prunes entire neurons while "random" is pruning individual weights.

### Recommendation

To ensure fair comparisons:
1. Fix the random pruning implementation to respect the `structured` flag
2. When comparing strategies, ensure they use the same granularity:
   - All structured (node-level) OR
   - All unstructured (weight-level)
3. Be explicit in your configuration about which type you want:
   ```yaml
   structured: true  # or false
   ```

## Final Fix: Default Dimension for Structured Pruning

After implementing the above fixes, there was still an issue where structured pruning wasn't working when `pruning_mode='random'`. The problem was in the base class `create_pruning_mask` method.

### The Hidden Bug

The `create_pruning_mask` method had this condition:
```python
if structured and dim is not None:
    # structured pruning code
else:
    # unstructured pruning code
```

But `dim` was never being passed! This meant that even with `structured=True`, it would fall through to unstructured pruning.

### The Solution

Changed the condition to default `dim=0` (output dimension) when not specified:
```python
if structured:
    # Default to dimension 0 (output dimension) for structured pruning
    if dim is None:
        dim = 0
    # ... rest of structured pruning code
```

Now structured pruning works correctly for all modes (low, high, random).

## Summary of All Fixes

1. **RandomPruning strategy**: Now generates neuron-level random scores when `structured=True`
2. **AlignmentPruning strategy**: 
   - Fixed 'high' mode to correctly prune highest-scoring neurons
   - Added proper 'random' mode handling
3. **Base class**: 
   - Added 'random' mode support
   - Fixed structured pruning to work without explicit `dim` parameter

With these fixes, all pruning strategies now correctly handle:
- **Structured pruning**: Removes entire neurons/channels
- **Unstructured pruning**: Removes individual weights
- **All selection modes**: low, high, and random work as expected 