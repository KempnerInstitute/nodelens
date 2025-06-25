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
- `"alignment"` or `"rayleigh_quotient"` - Prune based on neuron-input alignment
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

### 4. **Pruning Scope** (`scope`)
The granularity of pruning operations.

**Options:**
- `"global"` - Prune across entire model
- `"layer"` - Prune each layer independently
- `"channel"` - Prune entire channels
- `"filter"` - Prune entire filters

**Config:**
```yaml
pruning:
  scope: "global"
```

## Complete Example

```yaml
pruning:
  # What pruning method to use
  algorithms: ["magnitude", "gradient", "random"]
  
  # Which weights to prune (by importance)
  # Can be a single mode or list for comparison
  selection_mode: ["low", "high", "random"]
  
  # How much to prune
  sparsity_levels: [0.1, 0.3, 0.5, 0.7, 0.9]
  
  # Pruning granularity
  scope: "global"
  
  # Fine-tuning after pruning
  fine_tune_after_pruning: true
  fine_tune_epochs: 10
  fine_tune_learning_rate: 0.0001
```

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