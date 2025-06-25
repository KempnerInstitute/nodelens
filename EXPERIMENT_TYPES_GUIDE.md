# Pruning Experiment Types Guide

This guide explains the different ways to organize pruning experiments in the alignment framework.

## Overview

There are three concepts that are often confused:

1. **Pruning Scope** (removed) - Was about comparison granularity
2. **Experiment Types** - Different patterns for conducting pruning experiments
3. **Global Pruning Algorithm** - A specific pruning strategy

## Experiment Types

### 1. Standard Pruning (`standard_pruning`)

The most common approach - prunes all layers simultaneously.

**How it works:**
- Compute importance scores for each layer
- Prune each layer independently to its target sparsity
- All layers are pruned in one pass

**Configuration:**
```yaml
experiment_type: "standard_pruning"

pruning:
  algorithms: ["magnitude", "alignment", "hybrid"]
  structured: true  # For neuron pruning
  selection_mode: "low"
  sparsity_levels: [0.3, 0.5, 0.7]
  alignment_metric: "rayleigh_quotient"
```

**Use when:** You want to prune the entire network and evaluate the result.

### 2. Layer-Isolated Pruning (`layer_isolated_pruning`)

Studies each layer's individual importance by pruning one at a time.

**How it works:**
1. Prune layer 1 → evaluate → restore weights
2. Prune layer 2 → evaluate → restore weights
3. Continue for all layers

**Configuration:**
```yaml
experiment_type: "layer_isolated_pruning"

# Note: Different parameter names!
dropout_rates: [0.3, 0.5, 0.7]
pruning_metric: "rayleigh_quotient"
pruning_strategy: "low"  # Not "selection_mode"
exclude_classification_layer: true
```

**Use when:** You want to understand which layers are most/least important for performance.

### 3. Cascading Layer Pruning (`cascading_layer_pruning`)

Studies how pruning propagates through the network.

**How it works:**
1. Prune layer 1 → keep it pruned
2. Prune layer 2 (with layer 1 still pruned) → keep both pruned
3. Continue through all layers

**Configuration:**
```yaml
experiment_type: "cascading_layer_pruning"

cascade_direction: "forward"  # or "backward"
recompute_scores: true  # Recompute after each layer
dropout_rates: [0.3, 0.5, 0.7]
pruning_metric: "rayleigh_quotient"
pruning_strategy: "low"
```

**Options:**
- `cascade_direction`: Start from input ("forward") or output ("backward")
- `recompute_scores`: Whether to recompute importance after each layer

**Use when:** You want to study cumulative effects of pruning through the network.

## Global vs Layer-wise Pruning

This is different from experiment types! It's about how to compare importance:

### Layer-wise Pruning (Current Default)
- Each layer pruned to achieve its own target sparsity
- Example: Each layer pruned to 50% sparsity

### Global Pruning (Available as Algorithm)
- Compare all weights across all layers
- Prune the globally least important weights
- Some layers may be pruned more than others

**To use global pruning:**
```yaml
experiment_type: "standard_pruning"
pruning:
  algorithms: ["global_magnitude"]  # Special algorithm
  sparsity_levels: [0.5]
```

## Key Differences Summary

| Aspect | Standard | Layer-Isolated | Cascading |
|--------|----------|----------------|-----------|
| **Layers Pruned** | All at once | One at a time | Sequential |
| **Previous Pruning Kept** | N/A | No (restored) | Yes (cumulative) |
| **Purpose** | Regular pruning | Layer importance | Propagation effects |
| **Config Section** | `pruning:` | Root level | Root level |
| **Sparsity Parameter** | `sparsity_levels` | `dropout_rates` | `dropout_rates` |

## Common Confusion Points

### 1. "Scope" Parameter
The `scope` parameter that appeared in some configs **has been removed** because it was never implemented. Don't use:
```yaml
scope: "global"  # REMOVED - Don't use this
```

### 2. Parameter Names
Different experiment types use different parameter names:
- Standard pruning: Uses `pruning:` section with `selection_mode`
- Layer-isolated/Cascading: Use `pruning_strategy` at root level

### 3. Global Pruning
For true global pruning (comparing across all layers), use:
```yaml
algorithms: ["global_magnitude"]  # As an algorithm, not experiment type
```

## Example: Testing All Three Approaches

```bash
# Standard pruning
python run_experiment.py --config configs/unified_config.yaml \
    --experiment_type standard_pruning

# Layer-isolated analysis
python run_experiment.py --config configs/unified_config.yaml \
    --experiment_type layer_isolated_pruning \
    --dropout_rates "[0.3,0.5,0.7]"

# Cascading pruning
python run_experiment.py --config configs/unified_config.yaml \
    --experiment_type cascading_layer_pruning \
    --cascade_direction forward
```

## Best Practices

1. **For alignment-based pruning**: Always use `structured: true` since alignment computes neuron-level scores
2. **For layer importance studies**: Use layer-isolated pruning
3. **For understanding pruning propagation**: Use cascading pruning
4. **For production pruning**: Use standard pruning
5. **For global weight comparison**: Use `global_magnitude` algorithm

## Visual Summary

```
Standard Pruning:
Layer1: ████░░░░░░ (40% pruned)
Layer2: ████░░░░░░ (40% pruned)
Layer3: ████░░░░░░ (40% pruned)
All pruned simultaneously

Layer-Isolated:
Test 1: ████░░░░░░ ████████ ████████ (only layer 1)
Test 2: ████████ ████░░░░░░ ████████ (only layer 2)
Test 3: ████████ ████████ ████░░░░░░ (only layer 3)

Cascading:
Step 1: ████░░░░░░ ████████ ████████
Step 2: ████░░░░░░ ████░░░░░░ ████████
Step 3: ████░░░░░░ ████░░░░░░ ████░░░░░░
``` 