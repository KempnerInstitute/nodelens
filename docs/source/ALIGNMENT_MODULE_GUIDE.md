# Alignment Module User Guide

This guide provides comprehensive documentation for using the alignment module to analyze neural network representations.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Available Metrics](#available-metrics)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/alignment.git
cd alignment

# Install in development mode
pip install -e .

# Or install specific components
pip install -e ".[metrics,visualization]"
```

## Quick Start

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import get_metric

# Wrap your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
wrapped_model = ModelWrapper(model)

# Prepare data
inputs = torch.randn(100, 784)

# Extract activations
activations = wrapped_model.extract_activations(inputs)

# Compute metrics
rq_metric = get_metric("rayleigh_quotient")()
scores = rq_metric.compute(
    inputs=activations[0],
    weights=model[0].weight
)
```

## Core Concepts

### Model Wrapper

The `ModelWrapper` class provides a unified interface for extracting activations from any PyTorch model:

```python
from alignment.core import ModelWrapper

# Wrap any PyTorch model
wrapper = ModelWrapper(model)

# Extract activations from specific layers
activations = wrapper.extract_activations(
    inputs,
    layer_names=['layer1', 'layer2']
)
```

### Metrics

All metrics inherit from `BaseMetric` and follow a consistent interface:

```python
class BaseMetric:
    def compute(self, inputs=None, weights=None, outputs=None, **kwargs):
        """Compute metric scores."""
        pass
```

### Registry System

Metrics are registered and accessed through a central registry:

```python
from alignment.core.registry import METRIC_REGISTRY

# List available metrics
all_metrics = METRIC_REGISTRY.list()

# Get a specific metric
metric = METRIC_REGISTRY.get("mutual_information_gaussian")
```

## Available Metrics

The alignment module provides 36 metrics across 6 categories. For detailed mathematical descriptions, see [METRICS_REFERENCE.md](METRICS_REFERENCE.md).

### Categories

1. **Rayleigh Quotient Metrics** (3 metrics)
   - rayleigh_quotient
   - rq_alternative 
   - rq_patchwise

2. **Information-Theoretic Metrics** (14 metrics)
   - mutual_information_gaussian
   - mutual_information_binning
   - gaussian_mi_analytic
   - conditional_mutual_information
   - average_redundancy
   - layer_redundancy
   - mi_projection
   - pid_shared, pid_unique_x, pid_unique_y, pid_synergy
   - total_correlation
   - interaction_information
   - connected_information
   - synergistic_information

3. **Similarity Metrics** (7 metrics)
   - activation_cosine_similarity
   - weight_cosine_similarity
   - weight_dot_similarity
   - weight_euclidean_distance
   - node_correlation
   - node_redundancy
   - weight_activation_alignment

4. **Spectral Metrics** (8 metrics)
   - spectral_gap
   - spectral_norm_ratio
   - eigenvalue_entropy
   - spectral_clustering_score
   - eigenvalue_alignment
   - spectral_clustering
   - power_iteration
   - spectral_alignment

5. **Task-Specific Metrics** (8 metrics)
   - task_alignment
   - class_selectivity
   - feature_importance
   - representation_quality
   - classification_alignment
   - language_model_alignment
   - vision_task_alignment
   - reinforcement_learning_alignment

### Metric Properties

Each metric has properties indicating its data requirements:

```python
metric = get_metric("rayleigh_quotient")()
print(metric.requires_inputs)   # True
print(metric.requires_weights)  # True
print(metric.requires_outputs)  # False
```

## Advanced Features

### Batch Processing

For large-scale analysis:

```python
from alignment.utils.batch_processing import BatchMetricProcessor

processor = BatchMetricProcessor(
    metrics=['rayleigh_quotient', 'mutual_information_gaussian'],
    batch_size=1000,
    use_gpu=True
)

results = processor.process(model, dataloader)
```

### Visualization

Create comprehensive visualizations:

```python
from alignment.visualization import AlignmentVisualizer

visualizer = AlignmentVisualizer()
visualizer.plot_metric_distributions(results)
visualizer.create_report("alignment_report.html")
```

### Experiment Tracking

Track experiments with multiple backends:

```python
from alignment.utils.experiment_tracking import create_tracker

tracker = create_tracker("wandb", project="alignment-analysis")
tracker.log_metrics(results)
```

### Pruning Analysis

Analyze neuron importance for pruning:

```python
from alignment.utils.pruning import PruningUtilities

pruner = PruningUtilities(importance_metric="rayleigh_quotient")
importance_scores = pruner.compute_importance(model, dataloader)
pruned_model = pruner.prune(model, sparsity=0.5)
```

## API Reference

### Core Classes

#### ModelWrapper
```python
class ModelWrapper:
    def __init__(self, model: nn.Module)
    def extract_activations(self, inputs: torch.Tensor, layer_names: List[str] = None)
    def get_layer_names(self) -> List[str]
```

#### BaseMetric
```python
class BaseMetric(ABC):
    @abstractmethod
    def compute(self, inputs=None, weights=None, outputs=None, **kwargs) -> torch.Tensor
    
    @property
    def requires_inputs(self) -> bool
    
    @property
    def requires_weights(self) -> bool
    
    @property
    def requires_outputs(self) -> bool
```

### Utility Functions

#### Metric Creation
```python
def get_metric(name: str, **kwargs) -> BaseMetric:
    """Get a metric instance by name."""
    
def list_metrics() -> List[str]:
    """List all available metrics."""
```

#### Batch Processing
```python
def compute_metrics_parallel(
    metrics: List[str],
    inputs: torch.Tensor,
    weights: torch.Tensor,
    n_workers: int = 4
) -> Dict[str, torch.Tensor]:
    """Compute multiple metrics in parallel."""
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Use batch processing for large models
   - Reduce batch size
   - Use GPU acceleration when available

2. **NaN Values**
   - Check input data for NaN/Inf
   - Use regularization in metrics
   - Ensure sufficient data samples

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path includes the alignment module
   - Verify PyTorch installation

### Performance Tips

1. **GPU Acceleration**
   ```python
   # Enable GPU for metrics computation
   metric = get_metric("mutual_information_gaussian")(device='cuda')
   ```

2. **Parallel Processing**
   ```python
   # Use multiple workers
   results = compute_metrics_parallel(
       metrics=['rq', 'mi'],
       inputs=inputs,
       weights=weights,
       n_workers=8
   )
   ```

3. **Caching**
   ```python
   # Cache intermediate results
   from alignment.utils import ResultCache
   cache = ResultCache()
   ```

## Examples

### Complete Analysis Pipeline

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import METRIC_REGISTRY
from alignment.utils.batch_processing import BatchMetricProcessor
from alignment.visualization import AlignmentVisualizer

# 1. Setup model and data
model = load_pretrained_model()
dataloader = create_dataloader()

# 2. Configure metrics
metrics_config = {
    'rayleigh_quotient': {'relative': True},
    'mutual_information_gaussian': {'normalize': True},
    'spectral_gap': {'normalize': True}
}

# 3. Process in batches
processor = BatchMetricProcessor(
    metrics=list(metrics_config.keys()),
    metric_configs=metrics_config,
    use_gpu=True
)

results = processor.process(model, dataloader)

# 4. Visualize results
visualizer = AlignmentVisualizer()
visualizer.plot_metric_distributions(results)
visualizer.plot_layer_comparison(results)
visualizer.create_interactive_dashboard(results, "dashboard.html")

# 5. Export results
results.to_csv("alignment_scores.csv")
```

### Custom Metric Implementation

```python
from alignment.core.base import BaseMetric
from alignment.core.registry import register_metric

@register_metric("custom_metric")
class CustomMetric(BaseMetric):
    
    name = "custom_metric"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(self, param1=1.0):
        super().__init__()
        self.param1 = param1
    
    def compute(self, inputs, weights, outputs=None, **kwargs):
        # Your custom computation
        scores = torch.sum(inputs * weights.T, dim=1) * self.param1
        return scores
```

### Layer-wise Analysis

```python
# Analyze each layer separately
layer_results = {}

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        wrapper = ModelWrapper(model)
        acts = wrapper.extract_activations(data, [name])
        
        metric = get_metric("rayleigh_quotient")()
        scores = metric.compute(
            inputs=acts[name],
            weights=module.weight
        )
        
        layer_results[name] = scores

# Compare layers
visualizer.plot_layer_comparison(layer_results)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details. 