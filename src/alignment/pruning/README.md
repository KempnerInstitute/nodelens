# Pruning Module

Neural network pruning strategies and infrastructure.

## Strategies

### Magnitude-Based
- `MagnitudePruning` - Prune by weight magnitude
- `IterativeMagnitudePruning` - Gradual pruning with fine-tuning
- `GlobalMagnitudePruning` - Global cross-layer pruning

### Alignment-Based
- `AlignmentPruning` - Prune by alignment score
- `HybridPruning` - Combine magnitude and alignment
- `GlobalAlignmentPruning` - Global alignment-based pruning
- `CascadingAlignmentPruning` - Sequential layer pruning with score recomputation

### Gradient-Based
- `GradientPruning` - Prune by gradient magnitude
- `FisherPruning` - Fisher information-based pruning
- `MomentumPruning` - Momentum-based pruning

### Eigenvector-Based
- `EigenvectorPruning` - PCA-based pruning (prune low-variance neurons)

### Movement-Based (Sanh et al. NeurIPS 2020)
- `MovementPruning` - Prune weights moving toward zero during training
- `AdaptiveMovementPruning` - Adaptive movement pruning with auto-tuned amounts

### Adaptive Sensitivity-Based
- `AdaptiveSensitivityPruning` - Layer-adaptive pruning based on sensitivity analysis

### Random
- `RandomPruning` - Random baseline
- `LayerwiseRandomPruning` - Per-layer random pruning
- `BernoulliPruning` - Bernoulli mask pruning

### Cluster-Aware
- `ClusterAwarePruning` - Cluster-based structured pruning
- `CompositePruning` - Composite pruning strategies

### LLM Baselines
- `WandaPruning` - Sun et al. 2023
- `SparseGPTPruning` - Frantar & Alistarh 2023

### Parallel/Advanced
- `ParallelModePruning` - Multiple modes simultaneously
- `TensorizedPruning` - Tensorized pruning operations
- `AsyncParallelPruning` - Async parallel pruning
- `ParallelBatchPruning` - Batch parallel pruning

## Usage

```python
from alignment.pruning import MagnitudePruning, PruningConfig

config = PruningConfig(amount=0.5, structured=True)
strategy = MagnitudePruning(config)
mask = strategy.prune(layer, amount=0.5)
```

### Eigenvector Pruning

```python
from alignment.pruning import EigenvectorPruning, PruningConfig

config = PruningConfig(amount=0.5, structured=True, pruning_mode='low')
strategy = EigenvectorPruning(config=config)

# Prune neurons with low eigenvalue contribution (low variance)
mask = strategy.prune(layer, inputs=activations)
```

### Movement Pruning

```python
from alignment.pruning import MovementPruning

strategy = MovementPruning()

# During training, track weight movement
for batch in train_loader:
    loss.backward()
    strategy.update_movement_history(model)  # Call before optimizer.step()
    optimizer.step()

# After training, prune weights moving toward zero
mask = strategy.prune(layer, amount=0.5)
```

### Adaptive Sensitivity Pruning

```python
from alignment.pruning import AdaptiveSensitivityPruning

strategy = AdaptiveSensitivityPruning(
    target_sparsity=0.7,
    metric='rayleigh_quotient',
    sensitivity_method='activation_variance',  # FAST - single forward pass
    min_amount=0.1,
    max_amount=0.9
)

# Compute layer sensitivities and prune adaptively
sensitivities = strategy.compute_all_sensitivities(
    model,
    layer_names,
    data_loader=val_loader  # For fast methods
)

# Print report showing per-layer sensitivity and pruning amounts
strategy.print_sensitivity_report()

# Apply adaptive pruning
masks = strategy.prune_adaptive(model, layer_names, eval_fn=None, inputs_per_layer=None)
```

#### Sensitivity Methods

| Method | Speed | Accuracy | Requirements |
|--------|-------|----------|--------------|
| `perturbation` | Slow | High | `eval_fn` |
| `masking` | Slow | High | `eval_fn` |
| `activation_variance` | **Fast** | Medium | `data_loader` |
| `gradient` | **Fast** | Medium | `data_loader` |
| `fisher` | Medium | Medium-High | `data_loader` |
| `weight_magnitude` | **Fastest** | Low | None |

**Recommendation**: Use `activation_variance` for a good speed/accuracy tradeoff.

### Cascading Pruning (Progressive)

```python
from alignment.pruning import CascadingAlignmentPruning, PruningConfig

config = PruningConfig(amount=0.5, structured=True)
strategy = CascadingAlignmentPruning(
    metric='rayleigh_quotient',
    direction='forward',  # or 'backward'
    config=config
)

# Prune layer by layer, recomputing scores after each
masks = strategy.prune_model(model, get_layer_inputs_fn)
```

## Using the Pipeline

```python
from alignment.pruning import run_pruning_pipeline, PruningPipelineOptions

options = PruningPipelineOptions(
    distribution="uniform",   # or "global_threshold"
    dependency_aware=True,    # Handle dependent layers
    min_amount=0.0,
    max_amount=0.95
)

result = run_pruning_pipeline(
    model=model,
    layer_scores=scores_dict,
    target_sparsity=0.5,
    selection_mode="low",
    options=options
)
```

## Configuration

```python
config = PruningConfig(
    amount=0.5,              # Fraction to prune
    structured=False,        # Structured vs unstructured
    iterative=False,         # Single shot vs iterative
    global_pruning=False,    # Global vs layer-wise
    pruning_mode='low',      # 'low', 'high', or 'random'
)
```

## Structured vs Unstructured

- **Unstructured**: Remove individual weights (sparse matrices)
- **Structured**: Remove entire neurons/channels (dense matrices)

## Module Organization

- `base.py` - Base classes (`BasePruningStrategy`, `PruningConfig`)
- `pipeline.py` - Shared pruning pipeline (`run_pruning_pipeline`)
- `dependency_aware.py` - Handle dependent layers (BatchNorm, etc.)
- `distribution.py` - Layer sparsity distribution strategies
- `strategies/` - All pruning strategy implementations
