# Alignment Module User Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Metrics](#metrics)
6. [Model Wrapper](#model-wrapper)
7. [Experiments](#experiments)
8. [Advanced Usage](#advanced-usage)
9. [API Reference](#api-reference)

## Overview

The Alignment Module is a comprehensive framework for measuring and analyzing the alignment between neural network layers and their inputs/outputs. It provides:

- **20+ alignment metrics** including Rayleigh Quotient, Mutual Information, PID, and similarity measures
- **Model wrapping** for automatic activation and weight extraction
- **Pruning experiments** with various strategies
- **Distributed computing support** for multi-GPU training
- **Flexible architecture** for easy extension

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import torch
from alignment.models import ModelWrapper
from alignment.metrics.rayleigh import RayleighQuotient

# 1. Wrap your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
wrapped_model = ModelWrapper(model)

# 2. Get activations and weights
inputs = torch.randn(32, 784)
outputs, activations = wrapped_model.forward_with_activations(inputs)
weights = wrapped_model.get_layer_weights()

# 3. Compute alignment scores
rq_metric = RayleighQuotient()
for layer in wrapped_model.tracked_layers:
    scores = rq_metric.compute(
        inputs=activations[f"{layer}_input"],
        weights=weights[layer]
    )
    print(f"{layer}: mean RQ = {scores.mean():.4f}")
```

## Core Components

### 1. Model Wrapper

The `ModelWrapper` class automatically tracks activations and weights from your model:

```python
from alignment.models import ModelWrapper

# Auto-discover Linear and Conv layers
wrapped_model = ModelWrapper(model)

# Or specify layers manually
wrapped_model = ModelWrapper(
    model, 
    tracked_layers=['layer1.conv1', 'layer2.conv2']
)

# Configure preprocessing
wrapped_model = ModelWrapper(
    model,
    preprocessing_mode='unfold',  # For Conv layers
    unfold_kernel_size=3
)
```

### 2. Metrics Registry

Access all metrics through the registry:

```python
from alignment.metrics import METRIC_REGISTRY

# List available metrics
print(list(METRIC_REGISTRY.keys()))

# Get a metric class
RQMetric = METRIC_REGISTRY['rayleigh_quotient']
metric = RQMetric()
```

### 3. Experiments

Run structured experiments with automatic logging:

```python
from alignment.experiments import LayerIsolatedPruningExperiment
from alignment.experiments.configs import LayerIsolatedConfig

config = LayerIsolatedConfig(
    model=model,
    data_loader=train_loader,
    val_loader=val_loader,
    pruning_metric='rayleigh_quotient',
    dropout_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
    num_epochs=10
)

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()
```

## Metrics

### Rayleigh Quotient Metrics

#### Standard Rayleigh Quotient
```python
from alignment.metrics.rayleigh import RayleighQuotient

metric = RayleighQuotient(relative=True)
scores = metric.compute(inputs=inputs, weights=weights)
```

#### Alternative Rayleigh Quotient
```python
from alignment.metrics.rayleigh import RayleighQuotientAlternative

# Uses trace(C) as denominator instead of ||w||²
metric = RayleighQuotientAlternative()
scores = metric.compute(inputs=inputs, weights=weights)
```

### Information-Theoretic Metrics

#### Mutual Information
```python
from alignment.metrics.information import MutualInformationGaussian, MutualInformationBinning

# Gaussian approximation (fast)
mi_gaussian = MutualInformationGaussian()
scores = mi_gaussian.compute(outputs=outputs, target_outputs=targets)

# Binning approach (more accurate for non-Gaussian)
mi_binning = MutualInformationBinning(bins=10)
scores = mi_binning.compute(outputs=outputs, target_outputs=targets)
```

#### Conditional Mutual Information
```python
from alignment.metrics.information import ConditionalMutualInformation

cmi = ConditionalMutualInformation(bins=10, use_gaussian=False)
scores = cmi.compute(inputs=inputs, outputs=outputs, target_outputs=targets)
```

#### Partial Information Decomposition
```python
from alignment.metrics.information import SharedInformation, UniqueInformationX

# Requires BROJA_2PID module
si = SharedInformation(bins=10)
ui = UniqueInformationX(bins=10)

si_scores = si.compute(inputs=inputs, outputs=outputs)
ui_scores = ui.compute(inputs=inputs, outputs=outputs)
```

### Similarity Metrics

#### Weight Similarity
```python
from alignment.metrics.similarity import (
    WeightCosineSimilarity,
    WeightDotSimilarity,
    WeightEuclideanDistance
)

# Cosine similarity between weight vectors
cosine = WeightCosineSimilarity()
scores = cosine.compute(weights=weights)

# Euclidean distance (lower = more similar)
euclidean = WeightEuclideanDistance()
scores = euclidean.compute(weights=weights)
```

#### Node Metrics
```python
from alignment.metrics.similarity import NodeRedundancy, NodeCorrelation

# Input feature redundancy
redundancy = NodeRedundancy()
scores = redundancy.compute(inputs=inputs)

# Output correlation between neurons
correlation = NodeCorrelation(absolute=True)
scores = correlation.compute(outputs=outputs)
```

## Model Wrapper

### Basic Usage
```python
wrapped_model = ModelWrapper(model)

# Forward pass with activation collection
outputs, activations = wrapped_model.forward_with_activations(inputs)

# Access activations
for layer_name, layer_acts in activations.items():
    if layer_name.endswith('_input'):
        print(f"Input to {layer_name}: {layer_acts.shape}")
    elif layer_name.endswith('_output'):
        print(f"Output from {layer_name}: {layer_acts.shape}")

# Get weights
weights = wrapped_model.get_layer_weights()
```

### Preprocessing Modes

```python
# Flatten mode (default) - flattens spatial dimensions
wrapped = ModelWrapper(model, preprocessing_mode='flatten')

# Unfold mode - extracts patches from conv layers
wrapped = ModelWrapper(
    model, 
    preprocessing_mode='unfold',
    unfold_kernel_size=3,
    unfold_stride=1
)

# Patchwise mode - preserves spatial structure
wrapped = ModelWrapper(model, preprocessing_mode='patchwise')
```

### Pruning Support

```python
# Create pruning masks based on scores
scores = metric.compute(inputs, weights)
masks = {
    layer: scores < threshold 
    for layer, scores in layer_scores.items()
}

# Apply structured dropout
wrapped_model.apply_structured_dropout(masks)

# Make permanent
wrapped_model.make_pruning_permanent()

# Or restore
wrapped_model.restore_pruned_weights()
```

## Experiments

### Layer-Isolated Pruning
Prunes each layer independently:

```python
from alignment.experiments import LayerIsolatedPruningExperiment

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()

# Results include:
# - Accuracy at each pruning ratio
# - Layer-wise pruning statistics
# - Metric scores before/after pruning
```

### Progressive Dropout
Gradually increases pruning ratio:

```python
from alignment.experiments import ProgressiveDropoutExperiment

config = ProgressiveDropoutConfig(
    model=model,
    initial_dropout=0.1,
    final_dropout=0.9,
    num_steps=10
)

experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()
```

### Cascading Pruning
Prunes layers sequentially:

```python
from alignment.experiments import CascadingPruningExperiment

experiment = CascadingPruningExperiment(config)
results = experiment.run()
```

### Eigenvector-Based Pruning
Uses eigenvector analysis for pruning:

```python
from alignment.experiments import EigenvectorPruningExperiment

config = EigenvectorConfig(
    use_weight_eigenvectors=True,
    use_activation_eigenvectors=True
)

experiment = EigenvectorPruningExperiment(config)
results = experiment.run()
```

## Advanced Usage

### Custom Metrics

Create your own metrics by subclassing `BaseMetric`:

```python
from alignment.core.base import BaseMetric

class MyCustomMetric(BaseMetric):
    name = "my_custom_metric"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def compute(self, inputs=None, weights=None, outputs=None, **kwargs):
        # Your implementation
        scores = custom_computation(inputs, weights)
        return scores

# Register it
from alignment.metrics import METRIC_REGISTRY
METRIC_REGISTRY['my_custom_metric'] = MyCustomMetric
```

### Distributed Training

The framework supports distributed training:

```python
# Initialize distributed training
torch.distributed.init_process_group(backend='nccl')

# Metrics handle distributed computation automatically
metric = RayleighQuotient()
scores = metric.compute(inputs, weights)  # Aggregated across GPUs
```

### Checkpoint Management

Handle models with hooks properly:

```python
from alignment.utils.checkpoint import save_checkpoint, save_model_for_inference

# Save training checkpoint
save_checkpoint(
    model=wrapped_model.model,
    optimizer=optimizer,
    epoch=epoch,
    filepath='checkpoint.pt',
    additional_state={'best_acc': 0.95}
)

# Save for inference (removes hooks)
save_model_for_inference(
    model=wrapped_model.model,
    filepath='model_inference.pt',
    remove_hooks=True
)
```

### Performance Optimization

For large models or datasets:

```python
# Force CPU computation for large metric operations
metric = NodeRedundancy(force_cpu=True)

# Use fewer bins for faster PID computation
pid = SharedInformation(bins=5)  # Default is 10

# Process in batches
for batch in data_loader:
    scores = metric.compute(batch)
    # Accumulate results
```

## API Reference

### Core Classes

#### `ModelWrapper`
- `__init__(model, tracked_layers=None, preprocessing_mode='flatten', ...)`
- `forward_with_activations(inputs)` → `(outputs, activations)`
- `get_layer_weights()` → `Dict[str, Tensor]`
- `apply_structured_dropout(masks)`
- `make_pruning_permanent()`
- `restore_pruned_weights()`

#### `BaseMetric`
- `compute(inputs=None, weights=None, outputs=None, **kwargs)` → `Tensor`
- Class attributes: `name`, `requires_inputs`, `requires_weights`, `requires_outputs`

### Metric Classes

All metrics follow the same interface:

```python
metric = MetricClass(**init_params)
scores = metric.compute(
    inputs=input_tensor,      # If requires_inputs=True
    weights=weight_tensor,    # If requires_weights=True  
    outputs=output_tensor,    # If requires_outputs=True
    **additional_params
)
```

### Experiment Classes

All experiments follow:

```python
config = ExperimentConfig(**params)
experiment = ExperimentClass(config)
results = experiment.run()
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=/path/to/alignment/src:$PYTHONPATH
   ```

2. **PID Metrics Return Zeros**
   - Ensure BROJA_2PID module is in `src/alignment/external/`
   - Check logs for import warnings

3. **Memory Issues with Large Models**
   - Use `force_cpu=True` for metric computation
   - Process data in smaller batches
   - Use gradient checkpointing

4. **Checkpoint Loading Fails**
   - Use `save_model_for_inference()` for models with hooks
   - Load with `strict=False` for architecture mismatches

## Examples

See the `examples/` directory for:
- MNIST pruning example
- ResNet ImageNet pruning
- Custom metric implementation
- Distributed training example

## Contributing

1. Add new metrics in `src/alignment/metrics/`
2. Follow the `BaseMetric` interface
3. Add tests in `tests/unit/metrics/`
4. Update the registry in `metrics/__init__.py`
5. Document your metric in this guide

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alignment_module,
  title = {Neural Network Alignment Metrics Framework},
  author = {Your Team},
  year = {2024},
  url = {https://github.com/your-repo}
}
``` 