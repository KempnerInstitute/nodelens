# Pruning Configuration Guide

## Overview

The alignment framework provides comprehensive pruning capabilities with multiple strategies, modes, and configuration options.

## Available Pruning Strategies

### 1. Magnitude-Based Strategies

- **`magnitude`**: Basic magnitude pruning - removes weights with smallest absolute values
- **`iterative_magnitude`**: Gradual pruning over multiple iterations with fine-tuning
- **`global_magnitude`**: Global pruning with threshold across entire model

### 2. Gradient-Based Strategies

- **`gradient`**: Prunes based on gradient magnitudes
- **`fisher`**: Uses Fisher information for importance scoring
- **`momentum`**: Considers gradient momentum over time

### 3. Random Strategies

- **`random`**: Uniform random pruning (baseline)
- **`bernoulli`**: Probabilistic pruning with specified probability

### 4. Structured Strategies

- **`structured`**: Channel/filter-wise pruning for convolutional layers
- **`block_structured`**: Prunes contiguous blocks of weights

### 5. Parallel Strategies

- **`parallel_mode`**: Applies multiple pruning modes simultaneously
- **`tensorized`**: GPU-optimized parallel pruning computation
- **`async_parallel`**: CPU-parallel pruning across modules

## PruningConfig Parameters

The `PruningConfig` class accepts the following parameters:

```python
from alignment.pruning import PruningConfig

config = PruningConfig(
    # Basic parameters
    amount=0.5,                    # Fraction of weights to prune (0.0-1.0)
    structured=False,              # Whether to use structured pruning
    pruning_mode='low',           # 'low' (default), 'high', or 'random'
    
    # Advanced parameters
    iterative_steps=10,           # Number of pruning iterations (for iterative strategies)
    granularity='weight',         # 'weight', 'channel', 'filter', or 'block'
    block_size=(4, 4),           # Block dimensions for block-structured pruning
    
    # Fine-tuning parameters
    fine_tune_epochs=5,           # Epochs to fine-tune after each pruning step
    fine_tune_lr=0.001,          # Learning rate for fine-tuning
    
    # Module selection
    modules_to_prune=None,       # List of module names to prune (None = all)
    exclude_modules=None,        # List of module names to exclude
    
    # Importance computation
    importance_metric='magnitude', # 'magnitude', 'gradient', 'fisher', 'taylor'
    compute_importance_on='train', # 'train', 'val', or specific dataloader
    
    # Scheduling
    schedule='constant',          # 'constant', 'linear', 'exponential', 'polynomial'
    initial_sparsity=0.0,        # Starting sparsity for scheduled pruning
    final_sparsity=0.9,          # Target sparsity for scheduled pruning
)
```

## Pruning Modes

The `pruning_mode` parameter controls which weights are pruned:

- **`'low'`** (default): Prunes weights with smallest absolute values
- **`'high'`**: Prunes weights with largest absolute values (useful for analysis)
- **`'random'`**: Prunes weights randomly regardless of their values

## Configuration Examples

### 1. Basic Magnitude Pruning (50% sparsity)

```yaml
pruning_strategy: magnitude
pruning_config:
  amount: 0.5
  structured: false
  pruning_mode: low
```

### 2. Iterative Pruning with Fine-tuning

```yaml
pruning_strategy: iterative_magnitude
pruning_config:
  amount: 0.9
  iterative_steps: 10
  fine_tune_epochs: 5
  fine_tune_lr: 0.0001
  schedule: polynomial
```

### 3. Structured Channel Pruning

```yaml
pruning_strategy: structured
pruning_config:
  amount: 0.3
  structured: true
  granularity: channel
  modules_to_prune: 
    - features.conv1
    - features.conv2
```

### 4. Gradient-Based Pruning

```yaml
pruning_strategy: gradient
pruning_config:
  amount: 0.6
  importance_metric: gradient
  compute_importance_on: train
```

### 5. Parallel Multi-Mode Pruning

```yaml
pruning_strategy: parallel_mode
pruning_config:
  amount: 0.5
  pruning_mode: low  # Will test low, high, and random in parallel
```

### 6. Block-Structured Pruning

```yaml
pruning_strategy: block_structured
pruning_config:
  amount: 0.4
  granularity: block
  block_size: [4, 4]
```

## Strategy-Specific Options

### MagnitudePruning
- No additional parameters beyond PruningConfig

### IterativeMagnitudePruning
- `iterative_steps`: Number of pruning iterations
- `schedule`: How to distribute pruning across iterations

### GradientPruning
- `num_batches`: Number of batches to accumulate gradients
- `normalize_gradients`: Whether to normalize gradient magnitudes

### FisherPruning
- `num_samples`: Number of samples for Fisher information estimation
- `damping`: Damping factor for numerical stability

### StructuredPruning
- `granularity`: Must be 'channel' or 'filter'
- `norm_type`: Norm to use for importance ('l1' or 'l2')

### ParallelModePruning
- Automatically tests all three pruning modes
- Returns results for each mode

## Module Selection

### Specifying Modules to Prune

```python
config = PruningConfig(
    amount=0.5,
    modules_to_prune=['layer1.conv1', 'layer2.conv1', 'fc']
)
```

### Excluding Modules

```python
config = PruningConfig(
    amount=0.5,
    exclude_modules=['fc', 'classifier']  # Don't prune final layers
)
```

## Importance Metrics

The framework supports various importance metrics:

1. **`'magnitude'`**: Weight magnitude (default)
2. **`'gradient'`**: Gradient magnitude
3. **`'fisher'`**: Fisher information
4. **`'taylor'`**: First-order Taylor expansion
5. **`'random'`**: Random importance (baseline)

## Scheduling Options

### Constant Schedule
Prunes the same amount at each step:
```python
schedule='constant'
```

### Linear Schedule
Linearly increases sparsity:
```python
schedule='linear'
initial_sparsity=0.0
final_sparsity=0.9
```

### Exponential Schedule
Exponentially increases sparsity:
```python
schedule='exponential'
initial_sparsity=0.1
final_sparsity=0.9
```

### Polynomial Schedule
Uses polynomial decay:
```python
schedule='polynomial'
power=3  # Polynomial power
```

## Integration with Experiments

### In GeneralAlignmentExperiment

```python
from alignment.experiments import GeneralAlignmentConfig

config = GeneralAlignmentConfig(
    name="pruning_experiment",
    model_name="resnet18",
    dataset_name="cifar10",
    
    # Enable pruning
    apply_pruning=True,
    pruning_strategy="iterative_magnitude",
    pruning_config={
        "amount": 0.9,
        "iterative_steps": 10,
        "fine_tune_epochs": 5,
        "schedule": "polynomial"
    },
    
    # Compute metrics on pruned model
    compute_metrics_on="pruned"
)
```

### Custom Pruning Pipeline

```python
from alignment.pruning import get_pruning_strategy, PruningConfig

# Create pruning configuration
pruning_config = PruningConfig(
    amount=0.5,
    pruning_mode='low',
    structured=False
)

# Get pruning strategy
strategy = get_pruning_strategy('magnitude', config=pruning_config)

# Apply pruning
masks = strategy.compute_masks(model)
strategy.apply_masks(model, masks)

# Get pruning statistics
stats = strategy.get_pruning_stats(model)
print(f"Overall sparsity: {stats['overall_sparsity']:.2%}")
```

## Best Practices

1. **Start Conservative**: Begin with lower pruning amounts (30-50%) and increase gradually
2. **Use Iterative Pruning**: For high sparsity (>80%), use iterative pruning with fine-tuning
3. **Exclude Critical Layers**: Don't prune batch normalization or the final classification layer
4. **Monitor Performance**: Track accuracy degradation at each pruning level
5. **Compare Strategies**: Test multiple strategies to find the best for your model/dataset
6. **Use Structured Pruning**: For deployment, structured pruning provides actual speedup

## Troubleshooting

### Common Issues

1. **High accuracy drop**: 
   - Reduce pruning amount
   - Use iterative pruning
   - Increase fine-tuning epochs

2. **Memory errors with parallel strategies**:
   - Reduce batch size
   - Use async_parallel instead of tensorized

3. **Structured pruning not working**:
   - Ensure you're pruning convolutional layers
   - Check that granularity is set correctly

4. **Custom modules not pruned**:
   - Verify module names with `model.named_modules()`
   - Check exclude_modules list

## Advanced Usage

### Custom Importance Functions

```python
def custom_importance(module, inputs, outputs):
    """Custom importance scoring function."""
    return torch.abs(module.weight) * torch.abs(outputs).mean()

strategy = MagnitudePruning(config)
strategy.importance_fn = custom_importance
```

### Combining Multiple Strategies

```python
# Prune different parts with different strategies
conv_strategy = get_pruning_strategy('structured', 
    config=PruningConfig(amount=0.3, granularity='channel'))
fc_strategy = get_pruning_strategy('magnitude',
    config=PruningConfig(amount=0.5))

# Apply to specific modules
conv_masks = conv_strategy.compute_masks(model, 
    modules=['features'])
fc_masks = fc_strategy.compute_masks(model,
    modules=['classifier'])
```

### Pruning Schedule Visualization

```python
from alignment.analysis.visualization import PruningVisualizer

viz = PruningVisualizer()
schedule = strategy.get_pruning_schedule()
viz.plot_pruning_schedule(schedule, save_path='schedule.png')
```

This guide covers all available pruning strategies and their configuration options in the alignment framework. 