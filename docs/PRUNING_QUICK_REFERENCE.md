# Pruning Quick Reference

## Pruning Modes

All pruning strategies now support three modes:

- **`low`** (default): Prune weights with lowest importance scores
- **`high`**: Prune weights with highest importance scores  
- **`random`**: Prune weights randomly

```python
from alignment.pruning import PruningConfig, get_pruning_strategy

# Prune low-magnitude weights
config = PruningConfig(amount=0.5, pruning_mode='low')
strategy = get_pruning_strategy('magnitude', config=config)

# Prune high-magnitude weights
config = PruningConfig(amount=0.5, pruning_mode='high')
strategy = get_pruning_strategy('magnitude', config=config)
```

## Parallel Pruning Strategies

### ParallelModePruning

Apply multiple pruning modes simultaneously:

```python
from alignment.pruning.strategies import ParallelModePruning

strategy = ParallelModePruning(
    modes=['low', 'high', 'random'],
    base_strategy='magnitude'
)

result = strategy.prune_parallel(layer, amount=0.5)

# Access masks
low_mask = result.masks['low']
high_mask = result.masks['high']
random_mask = result.masks['random']

# Combine masks
union = strategy.combine_masks(result.masks, method='union')
intersection = strategy.combine_masks(result.masks, method='intersection')
majority = strategy.combine_masks(result.masks, method='majority')
```

### TensorizedPruning

GPU-optimized computation of all pruning variations:

```python
from alignment.pruning.strategies import TensorizedPruning

strategy = TensorizedPruning()

# Returns tensor: [num_modes, num_amounts, *weight_shape]
pruning_tensor = strategy.compute_pruning_tensor(
    layer,
    modes=['low', 'high', 'random'],
    amounts=[0.1, 0.3, 0.5, 0.7, 0.9]
)

# Analyze patterns
analysis = strategy.analyze_pruning_patterns(pruning_tensor)
```

### AsyncParallelPruning

Prune multiple modules in parallel:

```python
from alignment.pruning.strategies import AsyncParallelPruning

strategy = AsyncParallelPruning()

modules = [model.layer1, model.layer2, model.layer3]
results = strategy.prune_modules_parallel(
    modules,
    amounts=[0.5, 0.6, 0.7],
    modes=['low', 'high'],
    max_workers=4
)
```

## Command Line Usage

Run MNIST pruning experiment with visualization:

```bash
# Using config file
python -m alignment.experiments.run --config configs/experiments/pruning_mnist.yaml

# Direct command
python -m alignment.pruning.experiments.run \
  --dataset mnist \
  --model lenet \
  --strategy magnitude \
  --mode low \
  --sparsity 0.9 \
  --visualize
```

## Example: Complete Analysis

```python
from alignment.pruning import PruningConfig
from alignment.pruning.strategies import ParallelModePruning
import matplotlib.pyplot as plt

# Setup
config = PruningConfig(amount=0.7)
strategy = ParallelModePruning(
    config=config,
    modes=['low', 'high', 'random']
)

# Apply parallel pruning
result = strategy.prune_parallel(model.conv1)

# Compare effects
print("Sparsity by mode:")
for mode, sparsity in result.sparsities.items():
    print(f"  {mode}: {sparsity:.2%}")

# Analyze weight distributions
for mode in ['low', 'high']:
    mask = result.masks[mode]
    kept_weights = model.conv1.weight[mask.bool()]
    pruned_weights = model.conv1.weight[~mask.bool()]
    
    print(f"\n{mode.capitalize()} pruning:")
    print(f"  Avg kept magnitude: {kept_weights.abs().mean():.4f}")
    print(f"  Avg pruned magnitude: {pruned_weights.abs().mean():.4f}")
```

## Available Strategies

### Basic Strategies
- `magnitude`: MagnitudePruning
- `iterative_magnitude`: IterativeMagnitudePruning
- `global_magnitude`: GlobalMagnitudePruning
- `gradient`: GradientPruning
- `fisher`: FisherPruning
- `momentum`: MomentumPruning
- `random`: RandomPruning
- `bernoulli`: BernoulliPruning

### Parallel Strategies
- `parallel_mode`: ParallelModePruning
- `tensorized`: TensorizedPruning
- `async_parallel`: AsyncParallelPruning

## Tips

1. **For analysis**: Use `ParallelModePruning` to compare different modes
2. **For efficiency**: Use `TensorizedPruning` for GPU-accelerated computation
3. **For large models**: Use `AsyncParallelPruning` to prune layers in parallel
4. **High vs Low**: 
   - Low pruning keeps important weights (traditional)
   - High pruning removes important weights (useful for analysis)
5. **Visualization**: Run `examples/pruning_parallel_demo.py` for visual comparisons 