# Pruning Module

The pruning module provides a comprehensive set of strategies for neural network pruning, designed to work seamlessly with the alignment framework for analyzing how sparsity affects network alignment metrics.

## Overview

Neural network pruning reduces model size by removing parameters while maintaining performance. This module offers:

- **Multiple Pruning Strategies**: From simple magnitude-based to sophisticated gradient-based methods
- **Unified Interface**: All strategies follow the same API for easy experimentation
- **Integration with Alignment**: Designed to work with alignment metrics for comprehensive analysis
- **Flexible Configuration**: Support for structured/unstructured, iterative/one-shot, and global/local pruning

## Quick Start

```python
from alignment.pruning import MagnitudePruning, PruningConfig

# Basic pruning
strategy = MagnitudePruning()
mask = strategy.prune(layer, amount=0.5)  # Prune 50%

# Configured pruning
config = PruningConfig(
    amount=0.7,
    structured=True,
    iterative=True,
    iterations=10,
    fine_tune_epochs=5
)
strategy = MagnitudePruning(config)
```

## Available Strategies

### Magnitude-Based Pruning

#### MagnitudePruning
Basic magnitude pruning that removes weights with smallest absolute values.

```python
from alignment.pruning import MagnitudePruning

strategy = MagnitudePruning()
mask = strategy.prune(layer, amount=0.5)
```

#### IterativeMagnitudePruning
Gradually prunes weights over multiple iterations with fine-tuning.

```python
from alignment.pruning import IterativeMagnitudePruning, PruningConfig

config = PruningConfig(amount=0.9, iterations=10, fine_tune_epochs=5)
strategy = IterativeMagnitudePruning(config)

results = strategy.iterative_prune(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    criterion=nn.CrossEntropyLoss()
)
```

#### GlobalMagnitudePruning
Prunes weights globally across all layers to achieve network-wide sparsity.

```python
from alignment.pruning import GlobalMagnitudePruning

strategy = GlobalMagnitudePruning()
masks = strategy.prune_model(model, amount=0.7)
```

### Gradient-Based Pruning

#### GradientPruning
Uses gradient information to determine weight importance.

```python
from alignment.pruning import GradientPruning

# Using gradient magnitude
strategy = GradientPruning(mode='gradient')

# Using Taylor approximation (gradient * weight)
strategy = GradientPruning(mode='taylor')

# Need to compute gradients first
loss = criterion(model(inputs), targets)
loss.backward()

mask = strategy.prune(layer, amount=0.5)
```

#### FisherPruning
Uses Fisher information approximation for more stable importance estimates.

```python
from alignment.pruning import FisherPruning

strategy = FisherPruning()

# Accumulate Fisher information
for inputs, targets in dataloader:
    loss = criterion(model(inputs), targets)
    loss.backward()
    strategy.accumulate_fisher(model)
    optimizer.zero_grad()

# Prune based on accumulated information
masks = strategy.prune_model(model, amount=0.5)
```

#### MomentumPruning
Maintains momentum buffer for stable pruning decisions.

```python
from alignment.pruning import MomentumPruning

strategy = MomentumPruning(momentum=0.9)

for epoch in range(epochs):
    for inputs, targets in dataloader:
        loss = criterion(model(inputs), targets)
        loss.backward()
        strategy.update_momentum(model)
        optimizer.step()
        optimizer.zero_grad()

masks = strategy.prune_model(model, amount=0.5)
```

### Random Pruning (Baselines)

#### RandomPruning
Basic random pruning for baseline comparisons.

```python
from alignment.pruning import RandomPruning

strategy = RandomPruning(seed=42)  # For reproducibility
mask = strategy.prune(layer, amount=0.5)
```

#### LayerwiseRandomPruning
Random pruning with different sparsity per layer.

```python
from alignment.pruning import LayerwiseRandomPruning

layer_sparsity = {
    'features.0': 0.3,
    'features.3': 0.5,
    'classifier': 0.7
}

strategy = LayerwiseRandomPruning(layer_sparsity=layer_sparsity)
masks = strategy.prune_model(model)
```

#### BernoulliPruning
Probabilistic pruning where each weight has probability p of being pruned.

```python
from alignment.pruning import BernoulliPruning

strategy = BernoulliPruning(probability=0.5)
mask = strategy.prune(layer)
```

### Alignment-Based Pruning

#### AlignmentPruning
Prunes based on neuron-input alignment metrics (defaults to structured pruning).

```python
from alignment.pruning import AlignmentPruning

# Prune neurons with low alignment
strategy = AlignmentPruning(metric='rayleigh_quotient')

# Need inputs for alignment computation
inputs = torch.randn(batch_size, input_dim)
mask = strategy.prune(layer, inputs=inputs, amount=0.5)
```

#### GlobalAlignmentPruning
Global pruning based on alignment scores across all layers.

```python
from alignment.pruning import GlobalAlignmentPruning

strategy = GlobalAlignmentPruning(metric='rayleigh_quotient')

# Need inputs for each layer
layer_inputs = {}
# ... collect inputs with hooks ...

masks = strategy.prune_model(model, layer_inputs, amount=0.5)
```

#### HybridPruning
Combines magnitude and alignment information.

```python
from alignment.pruning import HybridPruning

# 70% alignment, 30% magnitude
strategy = HybridPruning(
    alignment_metric='rayleigh_quotient',
    alpha=0.7
)
mask = strategy.prune(layer, inputs=inputs, amount=0.5)
```

## Configuration

Use `PruningConfig` to configure pruning behavior:

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

## Pruning Experiments

The pruning module includes several experiments for analyzing how pruning affects model alignment:

### Progressive Pruning Experiment
Progressively increases pruning rates and tracks alignment metrics.

```python
from alignment.pruning.experiments import ProgressiveDropoutExperiment
from alignment.experiments import ExperimentConfig


config = ExperimentConfig(
    name="progressive_pruning",
    dropout_rates=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    dropout_structure='magnitude',  # or 'random', 'gradient'
    metrics=['rayleigh_quotient', 'mutual_information_gaussian']
)

experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()
```

### Cascading Layer Pruning
Analyzes the cascading effects of pruning layers sequentially.

```python
from alignment.pruning.experiments import CascadingLayerPruningExperiment

experiment = CascadingLayerPruningExperiment(config)
results = experiment.run()
```

### Layer-wise Pruning Analysis
Studies the impact of pruning individual layers in isolation.

```python
from alignment.pruning.experiments import LayerIsolatedPruningExperiment

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()
```

### Eigenvector-based Pruning
Uses eigenvector analysis to guide pruning decisions.

```python
from alignment.pruning.experiments import EigenvectorDropoutExperiment

experiment = EigenvectorDropoutExperiment(config)
results = experiment.run()
```

## Custom Pruning Strategies

Create custom strategies by extending `BasePruningStrategy`:

```python
from alignment.pruning.base import BasePruningStrategy

class CustomPruning(BasePruningStrategy):
    def compute_importance_scores(self, module, inputs=None, **kwargs):
        # Your custom importance computation
        importance = custom_importance_function(module.weight)
        return importance
```

## Utilities

### Get Strategy by Name
```python
from alignment.pruning import get_pruning_strategy

strategy = get_pruning_strategy('magnitude', amount=0.5)
```

### List Available Strategies
```python
from alignment.pruning import list_pruning_strategies

strategies = list_pruning_strategies()
print(strategies)
# ['magnitude', 'iterative_magnitude', 'global_magnitude', ...]
```

### Check Sparsity
```python
# Layer sparsity
sparsity = strategy.get_sparsity(layer)

# Model sparsity
total_params = sum(p.numel() for p in model.parameters())
zero_params = sum((p == 0).sum().item() for p in model.parameters())
model_sparsity = zero_params / total_params
```

## Best Practices

1. **Start Small**: Begin with 10-30% pruning to understand impact
2. **Use Iterative Pruning**: For high sparsity (>70%), use iterative pruning
3. **Fine-tune After Pruning**: Always fine-tune to recover performance
4. **Compare to Baselines**: Use random pruning as baseline
5. **Monitor Metrics**: Track both task performance and alignment metrics

## Examples

### Complete Pruning Pipeline
```python
from alignment.pruning import IterativeMagnitudePruning, PruningConfig
from alignment.metrics import get_metric

# Configure pruning
config = PruningConfig(
    amount=0.9,
    iterations=10,
    fine_tune_epochs=5
)

# Create strategy
strategy = IterativeMagnitudePruning(config)

# Track alignment during pruning
alignment_scores = []

def track_alignment(model):
    metric = get_metric('rayleigh_quotient')()
    scores = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            score = metric.compute(weights=module.weight)
            scores.append(score.mean().item())
    return np.mean(scores)

# Prune with custom tracking
initial_alignment = track_alignment(model)
results = strategy.iterative_prune(
    model=model,
    dataloader=train_loader,
    optimizer=optimizer,
    criterion=criterion
)
final_alignment = track_alignment(model)

print(f"Alignment change: {initial_alignment:.4f} -> {final_alignment:.4f}")
```

## Module Structure

```
pruning/
├── __init__.py          # Module exports and registry
├── base.py              # Base classes and interfaces
├── strategies/          # Pruning strategy implementations
│   ├── __init__.py
│   ├── magnitude.py     # Magnitude-based strategies
│   ├── gradient.py      # Gradient-based strategies
│   ├── random.py        # Random strategies
│   ├── alignment_based.py # Alignment-based strategies (with structured support)
│   └── cascading.py     # Cascading alignment strategy
├── experiments/         # Pruning experiments
│   ├── __init__.py
│   ├── progressive.py   # Progressive pruning analysis
│   ├── cascading_layer.py # Cascading layer-wise pruning
│   ├── layer_wise.py    # Layer-wise pruning analysis
│   └── eigenvector_based.py # Eigenvector-based pruning
├── STRUCTURED_PRUNING_STATUS.md # Status of structured pruning implementation
└── README.md           # This file
```

## Future Enhancements

- **Advanced Structured Patterns**: Block pruning, N:M sparsity patterns
- **Advanced Schedules**: Polynomial, exponential, and custom schedules
- **Hardware-Aware Pruning**: Optimize for specific hardware constraints
- **Lottery Ticket**: Find winning tickets in neural networks
- **Dynamic Pruning**: Adapt sparsity during training 

Note: Basic structured pruning (channel/neuron removal) is already implemented. See `STRUCTURED_PRUNING_STATUS.md` for details.

## Structured vs Unstructured Pruning

The module supports both structured and unstructured pruning:

```python
from alignment.pruning import PruningConfig

# Unstructured: Remove individual weights (default for magnitude)
config = PruningConfig(amount=0.5, structured=False)

# Structured: Remove entire neurons/channels (default for alignment)
config = PruningConfig(amount=0.5, structured=True)
```

**Key differences:**
- **Unstructured**: Creates sparse weight matrices, higher theoretical compression
- **Structured**: Removes entire neurons/channels, hardware-efficient, maintains dense matrices

See `STRUCTURED_PRUNING_STATUS.md` for implementation details. 