# Pruning Module

Comprehensive neural network pruning strategies for alignment analysis.

## Quick Start

```python
from alignment.pruning import MagnitudePruning, PruningConfig

# Basic pruning
strategy = MagnitudePruning()
mask = strategy.prune(layer, amount=0.5)

# Configured pruning
config = PruningConfig(amount=0.7, structured=True, iterative=True)
strategy = MagnitudePruning(config)
```

## Available Strategies

### Magnitude-Based
- **MagnitudePruning** - Remove weights with smallest absolute values
- **IterativeMagnitudePruning** - Gradual pruning with fine-tuning
- **GlobalMagnitudePruning** - Global pruning across all layers

### Gradient-Based
- **GradientPruning** - Use gradient information for importance
- **FisherPruning** - Fisher information approximation
- **MomentumPruning** - Momentum-based stable pruning

### Alignment-Based
- **AlignmentPruning** - Prune based on neuron-input alignment (structured by default)
- **GlobalAlignmentPruning** - Global alignment-based pruning
- **HybridPruning** - Combine magnitude and alignment

### Random (Baselines)
- **RandomPruning** - Basic random pruning
- **LayerwiseRandomPruning** - Different sparsity per layer
- **BernoulliPruning** - Probabilistic pruning

## Pruning Experiments

Specialized experiments for analyzing pruning effects:

```python
from alignment.pruning.experiments import (
    GlobalDropoutExperiment,
    LayerIsolatedPruningExperiment,
    CascadingLayerPruningExperiment,
    EigenvectorDropoutExperiment
)

# Global pruning analysis
experiment = GlobalDropoutExperiment(config)
results = experiment.run()
```

## Configuration

```python
from alignment.pruning import PruningConfig

config = PruningConfig(
    amount=0.5,              # Fraction to prune
    structured=False,        # Structured vs unstructured
    iterative=False,         # Single shot vs iterative
    global_pruning=False,    # Global vs layer-wise
    iterations=1,            # Number of iterations
    fine_tune_epochs=0       # Fine-tuning epochs per iteration
)
```

## Custom Strategies

Create custom pruning strategies:

```python
from alignment.pruning.base import BasePruningStrategy

class CustomPruning(BasePruningStrategy):
    def compute_importance_scores(self, module, inputs=None, **kwargs):
        # Your custom importance computation
        return custom_importance_function(module.weight)
```

## Key Features

- **Structured/Unstructured**: Both pruning types supported
- **Iterative Pruning**: Gradual pruning with fine-tuning
- **Global Pruning**: Network-wide sparsity optimization
- **Alignment Integration**: Pruning based on alignment metrics
- **Comprehensive Experiments**: Ready-to-use experiment classes

## Structured vs Unstructured

```python
# Unstructured: Remove individual weights (sparse matrices)
config = PruningConfig(amount=0.5, structured=False)

# Structured: Remove entire neurons/channels (dense matrices)
config = PruningConfig(amount=0.5, structured=True)
```

See `STRUCTURED_PRUNING_STATUS.md` for implementation details. 