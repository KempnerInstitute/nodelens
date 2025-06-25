# Pruning Concepts in the Alignment Framework

## Overview

This document explains key pruning concepts and how they apply in the alignment framework.

## Structured vs Unstructured Pruning

### Unstructured Pruning (Default for magnitude-based)
- **What**: Prunes individual weights independently
- **Result**: Sparse weight matrices with zeros scattered throughout
- **Hardware**: Requires special sparse matrix operations for speedup
- **Example**: In a 100×50 weight matrix, might remove any 2500 weights

```
Before:                    After (50% pruned):
[[1.2, -0.8, 0.3],        [[1.2,  0.0, 0.3],
 [0.1, -2.5, 1.7],   →     [0.0, -2.5, 1.7],
 [-0.9, 0.2, -1.1]]        [0.0,  0.0, -1.1]]
```

### Structured Pruning (Default for alignment-based)
- **What**: Prunes entire neurons/channels as units
- **Result**: Smaller dense matrices (actually removes rows/columns)
- **Hardware**: Works efficiently on standard hardware
- **Example**: In a 100×50 weight matrix, might remove 50 entire neurons (rows)

```
Before (3 neurons):        After (neuron 2 pruned):
[[1.2, -0.8, 0.3],        [[1.2, -0.8, 0.3],
 [0.1, -2.5, 1.7],   →     [-0.9, 0.2, -1.1]]
 [-0.9, 0.2, -1.1]]        
```

## Why Alignment Defaults to Structured Pruning

Alignment metrics (like Rayleigh quotient) measure properties of **entire neurons**, not individual weights:

1. **Conceptual match**: Alignment measures how well a neuron's weights align with its inputs
2. **Natural granularity**: One alignment score per neuron
3. **Meaningful interpretation**: "This neuron is poorly aligned with its inputs"

## How Alignment-Based Pruning Works

### Step 1: Compute Alignment Scores
For each layer, alignment metrics compute one score per neuron:

```python
# Example: Linear layer [256 outputs, 128 inputs]
weights.shape  # [256, 128]
inputs.shape   # [batch_size, 128]

# Rayleigh quotient computes 256 scores (one per output neuron)
alignment_scores = metric.compute(inputs, weights)  # [256]
```

### Step 2: Select Neurons to Prune
Based on the scores and selection mode:

```python
# With selection_mode="low" and 50% pruning:
# - Sort neurons by alignment score
# - Mark bottom 128 neurons (50%) for removal
# - These are the neurons with lowest alignment to their inputs
```

### Step 3: Apply Structured Pruning
Remove entire neurons (all their weights):

```python
# Original: [256, 128] with all neurons
# After: Effectively [128, 128] with poorly-aligned neurons removed
```

## Common Confusion: "Scope" vs "Structured"

**What users often ask about:**
- "How do I prune neurons instead of weights?" → Use `structured: true`
- "What is pruning scope?" → This was a confusing parameter that has been removed
- "How do I compare neurons across layers?" → Currently not implemented; pruning is per-layer

**The key setting for neuron pruning:**
```yaml
pruning:
  algorithms: ["alignment"]
  structured: true  # This ensures entire neurons are pruned!
```

## Configuration Examples

### Unstructured Magnitude Pruning
```yaml
pruning:
  algorithms: ["magnitude"]
  structured: false  # Individual weights
  selection_mode: "low"  # Remove small weights
  sparsity_levels: [0.5]
```

### Structured Alignment Pruning
```yaml
pruning:
  algorithms: ["alignment"]
  alignment_metric: "rayleigh_quotient"
  structured: true  # Entire neurons (default for alignment)
  selection_mode: "low"  # Remove poorly aligned neurons
  sparsity_levels: [0.5]
```

### Structured Magnitude Pruning
```yaml
pruning:
  algorithms: ["magnitude"]  
  structured: true  # Prune neurons with smallest average magnitude
  selection_mode: "low"
  sparsity_levels: [0.5]
```

## Implementation Details

### For Linear Layers
- **Unstructured**: Prune individual elements in weight matrix
- **Structured**: Prune entire rows (output neurons)

### For Conv Layers
- **Unstructured**: Prune individual weights in filters
- **Structured**: Prune entire output channels (all weights in a filter)

## Alignment Metrics Available

1. **`rayleigh_quotient`**: Classic neuron-input alignment measure
2. **`mutual_information`**: Information shared between inputs and outputs
3. **`cka`**: Centered Kernel Alignment similarity
4. **`weight_cosine_similarity`**: Cosine similarity between weight vectors
5. **`gradient_similarity`**: Alignment based on gradient information

## Best Practices

1. **Use structured pruning for alignment-based methods** (automatic default)
2. **Use unstructured pruning for magnitude when you need maximum sparsity**
3. **Use structured pruning for any method when you need hardware efficiency**
4. **Hybrid approach**: Combine magnitude and alignment for balanced pruning

```yaml
pruning:
  algorithms: ["hybrid"]
  alignment_metric: "rayleigh_quotient"
  hybrid_alpha: 0.7  # 70% alignment, 30% magnitude
  structured: true   # Prune entire neurons
``` 