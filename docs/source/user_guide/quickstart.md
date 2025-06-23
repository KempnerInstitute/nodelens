# Quick Start Guide

This guide will help you get started with the alignment framework.

## Basic Usage

### 1. Import Required Modules

```python
import torch
import torch.nn as nn
from alignment.core import ModelWrapper
from alignment.metrics import get_metric, METRIC_REGISTRY
```

### 2. Create or Load a Model

```python
# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Or load a pre-trained model
# model = torch.load('path/to/model.pth')
```

### 3. Wrap the Model

```python
# Wrap the model to track activations
wrapped_model = ModelWrapper(model)

# Get layer names
layer_names = wrapped_model.get_layer_names()
print("Available layers:", layer_names)
```

### 4. Compute Metrics

```python
# Create sample data
batch_size = 32
inputs = torch.randn(batch_size, 784)

# Extract activations
activations = wrapped_model.extract_activations(inputs)

# Compute Rayleigh Quotient for the first layer
rq_metric = get_metric("rayleigh_quotient")()
rq_scores = rq_metric.compute(
    inputs=inputs,
    weights=model[0].weight
)

print(f"RQ scores shape: {rq_scores.shape}")
print(f"Mean RQ score: {rq_scores.mean().item():.4f}")
```

## Running a Complete Analysis

Here's a complete example that analyzes multiple layers:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from alignment.core import ModelWrapper
from alignment.metrics import get_metric
from alignment.visualization import AlignmentVisualizer

# Create model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Create dummy dataset
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64)

# Wrap model
wrapped_model = ModelWrapper(model)

# Initialize metrics
metrics = {
    'rayleigh_quotient': get_metric('rayleigh_quotient')(),
    'mutual_information': get_metric('mutual_information_gaussian')()
}

# Collect results
results = {}

for batch_idx, (inputs, targets) in enumerate(dataloader):
    if batch_idx >= 5:  # Process only first 5 batches for demo
        break
    
    # Get activations
    activations = wrapped_model.extract_activations(inputs)
    
    # Compute metrics for each linear layer
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Linear):
            layer_results = {}
            
            # Get layer inputs (previous layer's output or original input)
            if i == 0:
                layer_input = inputs
            else:
                prev_layer_name = str(i-2)  # Account for ReLU layers
                layer_input = activations.get(prev_layer_name, inputs)
            
            # Compute metrics
            for metric_name, metric in metrics.items():
                scores = metric.compute(
                    inputs=layer_input,
                    weights=module.weight,
                    outputs=activations.get(str(i))
                )
                
                if name not in results:
                    results[name] = {}
                if metric_name not in results[name]:
                    results[name][metric_name] = []
                
                results[name][metric_name].append(scores.mean().item())

# Visualize results
visualizer = AlignmentVisualizer()

# Plot metric evolution
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (metric_name, ax) in enumerate(zip(metrics.keys(), axes)):
    for layer_name in results:
        scores = results[layer_name][metric_name]
        ax.plot(scores, label=f'Layer {layer_name}')
    
    ax.set_xlabel('Batch')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

## Using Configuration Files

For more complex experiments, you can use configuration files:

```python
from alignment.utils.batch_processing import BatchMetricProcessor

# Configure processor
processor = BatchMetricProcessor(
    metrics=['rayleigh_quotient', 'mutual_information_gaussian', 'spectral_gap'],
    batch_size=256,
    use_gpu=torch.cuda.is_available()
)

# Process entire dataset
all_results = processor.process(model, dataloader)

# Save results
import json
with open('results.json', 'w') as f:
    json.dump({k: v.tolist() for k, v in all_results.items()}, f)
```

## Next Steps

- Explore [available metrics](../api/metrics.md) for different analysis types
- Learn about [batch processing](batch_processing.md) for large-scale analysis
- See [visualization options](visualization.md) for creating plots and reports
- Check out [advanced examples](../examples/index.md) for complex use cases 