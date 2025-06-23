# Pruning Strategies Guide

This guide documents all pruning strategies available in the alignment framework and their use cases.

## Overview

Pruning is a technique for reducing neural network size by removing parameters while maintaining performance. The alignment framework provides several pruning strategies to analyze how network sparsity affects alignment metrics.

## Available Pruning Strategies

### 1. Magnitude-Based Pruning

**Location**: `utils/pruning.py::get_pruning_mask_magnitude()`

**Description**: Removes weights with the smallest absolute values.

**Theory**: Small magnitude weights contribute less to the network's output and can be removed with minimal impact.

**Usage**:
```python
from alignment.utils.pruning import PruningUtilities

mask = PruningUtilities.get_pruning_mask_magnitude(
    weights=layer.weight,
    amount=0.5,  # Prune 50% of weights
    structured=False,  # Unstructured pruning
    dim=None
)
PruningUtilities.apply_pruning_mask(layer, mask)
```

**Parameters**:
- `amount`: Fraction of weights to prune (0-1)
- `structured`: If True, prunes entire channels/filters
- `dim`: Dimension for structured pruning (0=output, 1=input)

### 2. Random Pruning

**Location**: `utils/pruning.py::get_pruning_mask_random()`

**Description**: Randomly removes weights regardless of their values.

**Theory**: Used as a baseline to compare against informed pruning strategies.

**Usage**:
```python
mask = PruningUtilities.get_pruning_mask_random(
    weights=layer.weight,
    amount=0.5,
    structured=False,
    dim=None
)
```

### 3. Gradient-Based Pruning

**Location**: `utils/pruning.py::get_pruning_mask_gradient()`

**Description**: Prunes weights with small gradient magnitudes.

**Theory**: Weights with small gradients have less impact on the loss function.

**Usage**:
```python
mask = PruningUtilities.get_pruning_mask_gradient(
    weights=layer.weight,
    gradients=layer.weight.grad,
    amount=0.5
)
```

**Requirements**: Requires gradient computation via backpropagation.

### 4. Sensitivity-Based Pruning

**Location**: `utils/pruning.py::get_pruning_mask_sensitivity()`

**Description**: Prunes weights that cause the smallest increase in loss when removed.

**Theory**: Directly measures each weight's impact on the loss function.

**Usage**:
```python
mask = PruningUtilities.get_pruning_mask_sensitivity(
    model=model,
    layer=layer,
    dataloader=dataloader,
    amount=0.5,
    loss_fn=nn.CrossEntropyLoss()
)
```

**Note**: Computationally expensive as it evaluates each weight individually.

### 5. Structured Pruning

**Location**: `utils/pruning.py::structured_pruning()`

**Description**: Removes entire channels, filters, or neurons.

**Theory**: Maintains regular tensor structure for hardware efficiency.

**Usage**:
```python
PruningUtilities.structured_pruning(
    layer=conv_layer,
    amount=0.3,
    dim=0,  # Prune output channels
    importance_scores=None  # Auto-computed if None
)
```

### 6. Iterative Magnitude Pruning

**Location**: `utils/pruning.py::iterative_magnitude_pruning()`

**Description**: Gradually prunes weights over multiple iterations with fine-tuning.

**Theory**: Allows network to adapt between pruning steps, maintaining better performance.

**Usage**:
```python
accuracies = PruningUtilities.iterative_magnitude_pruning(
    model=model,
    amount=0.9,  # Final sparsity
    iterations=10,
    dataloader=train_loader,
    fine_tune_epochs=5,
    optimizer=optimizer,
    loss_fn=criterion
)
```

### 7. Cascading Pruning

**Location**: `experiments/cascading.py::_cascading_prune()`

**Description**: Analyzes how pruning effects cascade through network layers.

**Theory**: Pruning early layers affects all subsequent layers; tracks these cascading effects.

**Usage**:
```python
from alignment.experiments import CascadingExperiment

experiment = CascadingExperiment(
    model=model,
    metrics=['rayleigh_quotient'],
    pruning_strategy='magnitude',
    cascade_analysis=True
)
```

## Pruning Schedules

Create pruning schedules for gradual sparsification:

```python
from alignment.utils.pruning import create_pruning_schedule

# Polynomial schedule (recommended)
schedule = create_pruning_schedule(
    initial_sparsity=0.0,
    final_sparsity=0.9,
    begin_step=1000,
    end_step=10000,
    frequency=100,
    schedule_type="polynomial"
)

# Get sparsity for current step
current_sparsity = schedule(step=5000)
```

**Schedule Types**:
- `linear`: Linear interpolation
- `polynomial`: Smooth cubic interpolation (default)
- `exponential`: Exponential decay

## Best Practices

### 1. Choosing a Strategy

- **Magnitude pruning**: Good default choice, simple and effective
- **Gradient-based**: When task-specific importance is crucial
- **Sensitivity-based**: For critical applications requiring optimal sparsity
- **Structured**: When hardware efficiency is important

### 2. Pruning Amount

- Start with small amounts (10-30%) for initial experiments
- Most networks can handle 50-70% sparsity with minimal accuracy loss
- 90%+ sparsity is possible but requires careful tuning

### 3. Iterative vs One-Shot

- **One-shot**: Fast, good for analysis
- **Iterative**: Better performance, allows adaptation

### 4. Fine-Tuning

Always fine-tune after pruning for best results:

```python
# Prune
mask = PruningUtilities.get_pruning_mask_magnitude(layer.weight, 0.5)
PruningUtilities.apply_pruning_mask(layer, mask)

# Fine-tune
for epoch in range(fine_tune_epochs):
    train(model, train_loader, optimizer, criterion)
```

## Utility Functions

### Check Sparsity
```python
# Layer sparsity
sparsity = PruningUtilities.get_sparsity(layer)

# Model sparsity
model_sparsity = PruningUtilities.get_model_sparsity(model)
```

### Remove Pruning (Make Permanent)
```python
# Makes pruning permanent and removes masks
PruningUtilities.remove_pruning(layer)
```

## Integration with Alignment Metrics

Pruning affects alignment metrics in various ways:

1. **Rayleigh Quotient**: May increase as unimportant directions are removed
2. **Mutual Information**: Can decrease if information pathways are disrupted
3. **Spectral Properties**: Eigenvalue distribution changes with sparsity

Example analysis:
```python
from alignment.experiments import CascadingExperiment

# Track how metrics change with pruning
experiment = CascadingExperiment(
    model=model,
    metrics=['rayleigh_quotient', 'mutual_information_gaussian', 'spectral_gap'],
    pruning_ratios=[0.0, 0.3, 0.5, 0.7, 0.9]
)
results = experiment.run(dataloader)

# Visualize metric changes vs sparsity
experiment.plot_results()
```

## Common Issues and Solutions

### Issue: Performance Degrades Significantly
**Solution**: Use iterative pruning with fine-tuning between steps

### Issue: Structured Pruning Removes Important Channels
**Solution**: Use custom importance scores based on your task

### Issue: Pruning Masks Not Persisting
**Solution**: Ensure hooks are properly registered, or make pruning permanent

### Issue: Memory Not Reduced After Pruning
**Solution**: Use structured pruning or sparse tensor formats 