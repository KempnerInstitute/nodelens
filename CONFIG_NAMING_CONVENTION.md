# Pruning Configuration Naming Convention

This document explains the standardized naming convention for pruning configurations in the alignment framework.

## Overview

We've simplified and clarified the pruning configuration to eliminate confusion between "strategy", "strategies", "modes", etc.

## Clear Terminology

### 1. **Pruning Algorithms** (`algorithms`)
The actual pruning method used to compute importance scores.

**Options:**
- `"magnitude"` - Prune based on weight magnitude
- `"gradient"` - Prune based on gradient magnitude  
- `"fisher"` - Prune based on Fisher information
- `"alignment"` - Prune based on neuron-input alignment (specify metric separately)
- `"hybrid"` - Combine magnitude and alignment scores
- `"random"` - Random importance scores (Note: Consider using `selection_mode: "random"` instead)

**Config:**
```yaml
pruning:
  algorithms: ["magnitude", "alignment"]  # Can test multiple algorithms
```

### 2. **Selection Mode** (`selection_mode`)
Which weights to prune based on their importance scores.

**Options:**
- `"low"` - Prune weights with lowest importance scores
- `"high"` - Prune weights with highest importance scores
- `"random"` - Randomly select weights to prune

**Usage:**
- **Single mode**: Test one selection mode
- **Multiple modes**: Test and compare different selection modes

**Config:**
```yaml
pruning:
  # Single mode
  selection_mode: "low"
  
  # OR multiple modes for comparison
  selection_mode: ["low", "high", "random"]
```

### 3. **Sparsity Levels** (`sparsity_levels`)
The fraction of weights to prune (0.0 to 1.0).

**Config:**
```yaml
pruning:
  sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9]  # Test multiple sparsity levels
```

### `alignment_metric`
**Type:** `str`  
**Purpose:** Specifies which alignment metric to use for alignment-based pruning algorithms.

**Options:**
- `"rayleigh_quotient"` - Classic neuron-input alignment based on variance
- `"mutual_information_gaussian"` - Information theoretic alignment
- `"weight_cosine_similarity"` - Cosine similarity between weight vectors
- `"gradient_similarity"` - Alignment based on gradient information
- `"cka"` - Centered Kernel Alignment

**Used when:** `algorithms` includes `"alignment"` or `"hybrid"`

**Example:**
```yaml
algorithms: ["alignment", "hybrid"]
alignment_metric: "rayleigh_quotient"
```

### `hybrid_alpha`
**Type:** `float` (0.0 to 1.0)  
**Purpose:** Controls the weighting between magnitude and alignment scores in hybrid pruning.

**Values:**
- `0.0` - Pure magnitude pruning (no alignment)
- `0.5` - Equal weighting (50% magnitude, 50% alignment)
- `1.0` - Pure alignment pruning (no magnitude)

**Used when:** `algorithms` includes `"hybrid"`

**Example:**
```yaml
algorithms: "hybrid"
hybrid_alpha: 0.7  # 70% alignment, 30% magnitude
```

### `structured`
**Type:** `bool`  
**Purpose:** Controls whether to prune individual weights or entire neurons/channels.

**Critical for alignment-based pruning!**

**Values:**
- `false` - Unstructured pruning: removes individual weights (creates sparse matrices)
- `true` - Structured pruning: removes entire neurons/channels (creates smaller dense matrices)

**Why this matters for alignment:**
- Alignment metrics compute **one score per neuron**
- With `structured: true`, entire neurons are removed based on these scores
- This is the natural and recommended way to use alignment-based pruning

**Example:**
```yaml
# For alignment-based pruning
algorithms: ["alignment"]
structured: true  # Prune entire neurons (recommended!)

# For magnitude-based pruning
algorithms: ["magnitude"]
structured: false  # Prune individual weights (traditional approach)
```

### `sparsity_levels`
**Type:** `float` or `List[float]`  
**Purpose:** Specifies how much to prune (fraction of weights/neurons to remove).

**Values:** Between 0.0 and 1.0
- `0.1` = Prune 10%
- `0.5` = Prune 50%
- `0.9` = Prune 90%

**Can be a list** to test multiple sparsity levels.

**Example:**
```yaml
sparsity_levels: 0.5                      # Single level
sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9] # Multiple levels
```

## Removed/Deprecated Parameters

### `scope` (Removed)
**Why removed:** This parameter appeared in configs but was never implemented. It was intended to control whether pruning decisions were made globally across all layers or locally within each layer.

**Current behavior:** Pruning is always done per layer.

**If you were using:**
```yaml
scope: "global"  # This had no effect
```

**Just remove it.** Global pruning across layers is a planned future feature.

### `granularity` and `block_size` (Removed from main config)
These were overly specific and not implemented. Use `structured: true/false` instead.

## Complete Example

Here's a complete pruning configuration showing the new naming:

```yaml
pruning:
  # Which pruning algorithms to use
  algorithms: ["magnitude", "gradient", "fisher"]
  
  # Which weights to prune based on importance
  selection_mode: "low"  # Prune lowest importance weights
  
  # How much to prune
  sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9]
  
  # For alignment-based pruning
  alignment_metric: "rayleigh_quotient"
  hybrid_alpha: 0.5
  
  # Structured pruning (neurons vs weights)
  structured: false  # Set to true for neuron pruning
  
  # Fine-tuning
  fine_tune_after_pruning: true
  fine_tune_epochs: 10
```

This naming convention provides clarity and consistency across the pruning system.

## Migration from Old Names

For backward compatibility, the system still supports old parameter names:

| Old Name | New Name | Description |
|----------|----------|-------------|
| `strategy` | `algorithms` | Pruning method(s) |
| `amount` | `sparsity_levels` | How much to prune |
| `strategies` | `selection_mode` | Which weights to prune |
| `pruning_modes` | `selection_mode` | Which weights to prune |
| `fine_tune` | `fine_tune_after_pruning` | Whether to fine-tune |
| `fine_tune_lr` | `fine_tune_learning_rate` | Learning rate for fine-tuning |

## Code Usage

In Python code:
```python
from alignment.pruning.base import PruningConfig

# Clear and explicit
config = PruningConfig(
    amount=0.5,                    # 50% sparsity
    pruning_mode='low',           # Prune low importance weights
    global_pruning=True,          # Global scope
    iterative=False
)
```

## Benefits

1. **No ambiguity** between algorithms and selection modes
2. **Descriptive names** that clearly indicate purpose
3. **Consistent terminology** throughout codebase
4. **Backward compatible** with existing configs
5. **Self-documenting** configuration files 