# Pruning Guide

This guide covers the pruning capabilities of the alignment framework, including different strategies, modes, and parallel execution options.

## Overview

The pruning module provides comprehensive pruning capabilities with:

- **Multiple Strategies**: Magnitude, gradient, random, and more
- **Pruning Modes**: Low (prune small values), high (prune large values), random
- **Parallel Execution**: Apply multiple modes simultaneously
- **GPU Optimization**: Tensorized operations for efficiency

## Basic Usage

### Simple Pruning

```python
from alignment.pruning import get_pruning_strategy, PruningConfig

# Basic magnitude pruning (prune small weights)
strategy = get_pruning_strategy('magnitude')
mask = strategy.prune(model.fc1, amount=0.5)

# Apply mask
model.fc1.weight.data *= mask
```

### Pruning Modes

```python
# Prune high-magnitude weights (adversarial)
config = PruningConfig(amount=0.5, pruning_mode='high')
strategy = get_pruning_strategy('magnitude', config=config)
mask = strategy.prune(model.fc1)

# Random pruning
config = PruningConfig(amount=0.5, pruning_mode='random')
strategy = get_pruning_strategy('random', config=config)
```

## Available Strategies

### Magnitude-Based

- **MagnitudePruning**: Basic magnitude pruning
- **IterativeMagnitudePruning**: Gradual pruning over iterations
- **GlobalMagnitudePruning**: Global threshold across layers

```python
# Iterative pruning
from alignment.pruning.strategies import IterativeMagnitudePruning

strategy = IterativeMagnitudePruning(
    initial_sparsity=0.1,
    final_sparsity=0.9,
    num_iterations=10
)

for i in range(10):
    mask = strategy.prune_iteration(model.fc1, iteration=i)
    # Apply mask and continue training...
```

### Gradient-Based

- **GradientPruning**: Based on gradient magnitudes
- **FisherPruning**: Using Fisher information
- **MomentumPruning**: Considering gradient momentum

```python
# Gradient-based pruning
from alignment.pruning.strategies import GradientPruning

# Requires gradients from backward pass
output = model(input_batch)
loss = criterion(output, targets)
loss.backward()

strategy = GradientPruning()
mask = strategy.prune(model.fc1, amount=0.5)
```

### Random

- **RandomPruning**: Uniform random selection
- **BernoulliPruning**: Probabilistic pruning

## Parallel Pruning

### Multiple Modes Simultaneously

```python
from alignment.pruning.strategies import ParallelModePruning

# Apply low, high, and random pruning in parallel
parallel_strategy = ParallelModePruning(
    modes=['low', 'high', 'random'],
    base_strategy='magnitude'
)

result = parallel_strategy.prune_parallel(model.fc1, amount=0.5)

# Access individual masks
low_mask = result.masks['low']
high_mask = result.masks['high']
random_mask = result.masks['random']

# Check sparsities
print(f"Sparsities: {result.sparsities}")

# Analyze overlaps
overlap = (low_mask * high_mask).sum() / low_mask.numel()
print(f"Low-High overlap: {overlap:.2%}")
```

### GPU-Optimized Tensorized Pruning

```python
from alignment.pruning.strategies import TensorizedPruning

# Compute multiple sparsity levels at once
tensorized = TensorizedPruning()

# Get tensor of shape [modes, sparsities, *weight_shape]
pruning_tensor = tensorized.compute_pruning_tensor(
    model.fc1,
    modes=['low', 'high'],
    amounts=[0.1, 0.3, 0.5, 0.7, 0.9]
)

# Analyze pruning patterns
analysis = tensorized.analyze_pruning_patterns(pruning_tensor)
print(f"Sparsity progression: {analysis['sparsity_progression']}")
```

## Configuration

### PruningConfig Options

```python
from alignment.pruning import PruningConfig

config = PruningConfig(
    amount=0.5,              # Sparsity level (0.5 = 50% pruned)
    pruning_mode='low',      # 'low', 'high', or 'random'
    structured=False,        # Structured vs unstructured
    scope='local',           # 'local' or 'global'
    iterative=False,         # Single shot vs iterative
    device='cuda'            # Computation device
)
```

### Global Configuration

```python
# Set default configuration
from alignment.pruning import set_default_config

set_default_config(
    pruning_mode='low',
    device='cuda'
)
```

## Advanced Usage

### Custom Pruning Strategy

```python
from alignment.pruning.base import BasePruningStrategy
import torch

class MyCustomPruning(BasePruningStrategy):
    """Custom pruning based on activation patterns."""
    
    def compute_importance_scores(self, module, inputs=None, **kwargs):
        # Custom importance computation
        if inputs is not None:
            # Use activation-based importance
            activations = module(inputs)
            importance = activations.std(dim=0)
        else:
            # Fallback to weight magnitude
            importance = module.weight.abs().sum(dim=1)
        return importance
    
    def create_pruning_mask(self, importance_scores, amount=None):
        # Use parent's masking logic
        return super().create_pruning_mask(importance_scores, amount)

# Register custom strategy
from alignment.pruning import register_pruning_strategy
register_pruning_strategy('custom', MyCustomPruning)
```

### Layer-wise Pruning

```python
def prune_model_layerwise(model, sparsity_dict):
    """Apply different sparsity to each layer."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and name in sparsity_dict:
            strategy = get_pruning_strategy('magnitude')
            mask = strategy.prune(module, amount=sparsity_dict[name])
            module.weight.data *= mask
```

### Pruning with Fine-tuning

```python
def prune_and_finetune(model, train_loader, val_loader, epochs=10):
    """Prune model and fine-tune."""
    # Initial evaluation
    initial_acc = evaluate(model, val_loader)
    
    # Prune
    strategy = get_pruning_strategy('magnitude')
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            mask = strategy.prune(module, amount=0.5)
            # Store mask for reapplying during training
            module.register_buffer('pruning_mask', mask)
    
    # Fine-tune with masks
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for batch in train_loader:
            # Forward pass
            loss = train_step(model, batch, optimizer)
            
            # Reapply masks after gradient update
            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, 'pruning_mask'):
                        module.weight *= module.pruning_mask
    
    final_acc = evaluate(model, val_loader)
    print(f"Accuracy: {initial_acc:.2%} -> {final_acc:.2%}")
```

## Visualization

### Using PruningVisualizer

```python
from alignment.analysis.visualization import PruningVisualizer

visualizer = PruningVisualizer()

# Compare different strategies
results = {
    'magnitude_low': {...},   # Results from low mode
    'magnitude_high': {...},  # Results from high mode
    'random': {...}           # Results from random
}

# Create comparison plots
fig = visualizer.plot_pruning_performance(
    results,
    metrics=['accuracy', 'loss'],
    save_path='pruning_comparison.png'
)

# Multi-seed analysis
seed_results = {...}  # Results from multiple runs
fig = visualizer.plot_multi_seed_results(
    seed_results,
    metric='accuracy',
    save_path='multi_seed_analysis.png'
)
```

## Best Practices

1. **Start Small**: Begin with low sparsity (10-30%) and increase gradually
2. **Layer Sensitivity**: Different layers have different pruning tolerance
3. **Fine-tune After Pruning**: Always fine-tune to recover performance
4. **Use Appropriate Metrics**: Monitor both sparsity and performance
5. **Consider Structure**: Structured pruning for hardware efficiency
6. **Validate Thoroughly**: Test on validation set, not training set

## Common Patterns

### Progressive Pruning
```python
sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
for sparsity in sparsities:
    # Prune
    masks = prune_model(model, sparsity)
    
    # Fine-tune
    finetune(model, train_loader, epochs=5)
    
    # Evaluate
    acc = evaluate(model, val_loader)
    print(f"Sparsity {sparsity}: Accuracy {acc:.2%}")
```

### Pruning Analysis Pipeline
```python
# 1. Train baseline model
model = train_model(...)

# 2. Compute alignment metrics
metrics_before = compute_metrics(model, data_loader)

# 3. Apply pruning
strategy = get_pruning_strategy('magnitude')
prune_model(model, strategy, amount=0.5)

# 4. Compute metrics after
metrics_after = compute_metrics(model, data_loader)

# 5. Analyze changes
analyze_metric_changes(metrics_before, metrics_after)
```

## Troubleshooting

### Common Issues

1. **Performance Degradation**: Reduce sparsity or use iterative pruning
2. **Memory Issues**: Use CPU offloading for large models
3. **Gradient Problems**: Ensure gradients are computed before gradient-based pruning
4. **Mask Persistence**: Store masks as buffers to survive model.to(device)

### Debugging Tips

```python
# Check actual sparsity achieved
def check_sparsity(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            sparsity = (module.weight == 0).float().mean()
            print(f"{name}: {sparsity:.2%}")

# Verify mask application
def verify_masks(model, masks):
    for name, mask in masks.items():
        module = dict(model.named_modules())[name]
        assert (module.weight * (1 - mask) == 0).all()
```

## See Also

- [Metrics Guide](metrics.md) - Computing alignment metrics
- [Experiments Guide](experiments.md) - Running pruning experiments
- [API Reference](../api/pruning.rst) - Detailed API documentation 