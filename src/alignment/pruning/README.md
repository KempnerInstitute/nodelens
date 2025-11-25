# Pruning Module

Neural network pruning strategies.

## Strategies

### Magnitude-Based
- `MagnitudePruning` - Prune by weight magnitude
- `IterativeMagnitudePruning` - Gradual pruning with fine-tuning
- `GlobalMagnitudePruning` - Global cross-layer pruning

### Alignment-Based
- `AlignmentPruning` - Prune by alignment score
- `HybridPruning` - Combine magnitude and alignment

### Random
- `RandomPruning` - Random baseline

## Usage

```python
from alignment.pruning import MagnitudePruning, PruningConfig

config = PruningConfig(amount=0.5, structured=True)
strategy = MagnitudePruning(config)
mask = strategy.prune(layer, amount=0.5)
```

## Configuration

```python
config = PruningConfig(
    amount=0.5,              # Fraction to prune
    structured=False,        # Structured vs unstructured
    iterative=False,         # Single shot vs iterative
    global_pruning=False,    # Global vs layer-wise
)
```

## Structured vs Unstructured

- **Unstructured**: Remove individual weights (sparse matrices)
- **Structured**: Remove entire neurons/channels (dense matrices)
