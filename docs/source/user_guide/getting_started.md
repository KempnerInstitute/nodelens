# Getting Started

This guide will help you get started with the alignment framework.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU support)

### Install from Source

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
pip install -e .
```

### Verify Installation

```python
import alignment
print(alignment.__version__)
```

## Quick Start

### 1. Basic Metric Computation

```python
import torch
from alignment.metrics import get_metric

# Create sample data
inputs = torch.randn(32, 784)
weights = torch.randn(256, 784)

# Compute Rayleigh Quotient
RQMetric = get_metric('rayleigh_quotient')
rq = RQMetric()
scores = rq.compute(inputs=inputs, weights=weights)
print(f"RQ scores: mean={scores.mean():.4f}")
```

### 2. Model Wrapping

```python
import torch.nn as nn
from alignment.models import ModelWrapper

# Create a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Wrap it to track activations
wrapped_model = ModelWrapper(model)

# Forward pass
inputs = torch.randn(32, 784)
outputs, activations = wrapped_model.forward_with_activations(inputs)

# Access layer information
weights = wrapped_model.get_layer_weights()
print(f"Tracked layers: {wrapped_model.tracked_layers}")
```

### 3. Pruning

```python
from alignment.pruning import get_pruning_strategy, PruningConfig

# Configure pruning
config = PruningConfig(amount=0.5, pruning_mode='low')
strategy = get_pruning_strategy('magnitude', config=config)

# Apply to a layer
layer = model[0]  # First linear layer
mask = strategy.prune(layer)
layer.weight.data *= mask

print(f"Sparsity: {(mask == 0).float().mean():.2%}")
```

## Complete Example

Here's a complete workflow:

```python
import torch
import torch.nn as nn
from alignment.models import ModelWrapper
from alignment.metrics import get_metric
from alignment.pruning import get_pruning_strategy, PruningConfig

# 1. Create and wrap model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
wrapped_model = ModelWrapper(model)

# 2. Generate data
inputs = torch.randn(32, 784)

# 3. Forward pass
outputs, activations = wrapped_model.forward_with_activations(inputs)

# 4. Compute metrics
rq_metric = get_metric('rayleigh_quotient')()
weights = wrapped_model.get_layer_weights()

for layer_name in wrapped_model.tracked_layers:
    layer_inputs = activations[f"{layer_name}_input"]
    layer_weights = weights[layer_name]
    
    scores = rq_metric.compute(inputs=layer_inputs, weights=layer_weights)
    print(f"Layer {layer_name}: RQ mean={scores.mean():.4f}")

# 5. Apply pruning
strategy = get_pruning_strategy('magnitude')
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        mask = strategy.prune(module, amount=0.5)
        module.weight.data *= mask

print("\nPruning complete!")
```

## Running Examples

The `examples/` folder contains several demonstrations:

```bash
# Quick introduction
python examples/quick_demo.py

# Complete experiment workflow
python examples/standard_alignment_experiment.py

# Pruning strategies demonstration
python examples/pruning_strategies_demo.py
```

## Next Steps

1. **Explore Metrics**: See the [Metrics Guide](metrics.md) for available metrics
2. **Learn Pruning**: Read the [Pruning Guide](pruning.md) for advanced strategies
3. **Run Experiments**: Check the [Experiments Guide](experiments.md)
4. **Visualize Results**: See the [Visualization Guide](visualization.md)

## Common Issues

### Import Errors
- Ensure you're in the correct conda environment
- Verify installation: `pip show alignment`
- Check that you installed in editable mode (`-e`)

### CUDA/GPU Issues
- The framework automatically falls back to CPU
- To force CPU: `device = torch.device('cpu')`
- Check CUDA availability: `torch.cuda.is_available()`

### Memory Issues
- Reduce batch size for metric computation
- Use CPU offloading for large models
- Process layers sequentially instead of all at once

## Getting Help

- Check the [API Reference](../api/index.rst)
- Look at working examples in `examples/`
- Review test cases in `tests/`
- Open an issue on GitHub for bugs or questions 