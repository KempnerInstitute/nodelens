# Alignment Metrics Framework - Refactored Architecture

## Overview

This refactored version of the alignment metrics framework provides a clean, modular, and scalable architecture for computing and analyzing alignment metrics in neural networks. The framework is designed for:

- **Multi-GPU/HPC Support**: Built-in support for distributed computing using PyTorch DDP
- **Extensibility**: Easy to add new metrics, models, and analysis methods
- **Performance**: Optimized computations with proper GPU/CPU memory management
- **Modularity**: Clear separation of concerns with focused, single-responsibility modules
- **Type Safety**: Full type annotations and protocol-based interfaces

## Architecture Overview

```
src/alignment_refactor/
├── core/                 # Core abstractions and base classes
│   ├── __init__.py
│   ├── base.py          # Base classes for metrics, models, experiments
│   ├── protocols.py     # Protocol definitions for interfaces
│   └── registry.py      # Central registry for metrics, models, etc.
├── metrics/             # Metric implementations
│   ├── __init__.py
│   ├── base.py          # Base metric classes and protocols
│   ├── rayleigh/        # Rayleigh Quotient-based metrics
│   │   ├── __init__.py
│   │   ├── rayleigh_quotient.py
│   │   ├── delta_alignment.py
│   │   └── normalized_rq.py
│   ├── information/     # Information-theoretic metrics
│   │   ├── __init__.py
│   │   ├── mutual_information.py
│   │   ├── pid.py      # Partial Information Decomposition
│   │   └── redundancy.py
│   └── similarity/      # Similarity-based metrics
│       ├── __init__.py
│       ├── cosine_similarity.py
│       ├── weight_similarity.py
│       └── node_correlation.py
├── models/              # Model wrappers and implementations
│   ├── __init__.py
│   ├── base.py          # Base model wrapper
│   ├── wrappers.py      # AlignmentNetwork wrapper
│   ├── architectures/   # Specific architectures
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   └── external.py  # External model integration
│   └── layers.py        # Custom layers (e.g., dropout variants)
├── data/                # Data loading and processing
│   ├── __init__.py
│   ├── base.py          # Base dataset classes
│   ├── loaders.py       # Data loader utilities
│   ├── datasets/        # Specific dataset implementations
│   │   ├── __init__.py
│   │   ├── mnist.py
│   │   ├── cifar.py
│   │   └── imagenet.py
│   └── processors.py    # Data preprocessing utilities
├── experiments/         # Experiment runners
│   ├── __init__.py
│   ├── base.py          # Base experiment class
│   ├── runners/         # Specific experiment runners
│   │   ├── __init__.py
│   │   ├── progressive_dropout.py
│   │   ├── layer_isolated.py
│   │   ├── cascading_pruning.py
│   │   └── eigenvector_analysis.py
│   └── callbacks.py     # Experiment callbacks and hooks
├── analysis/            # Analysis and visualization
│   ├── __init__.py
│   ├── aggregators.py   # Result aggregation utilities
│   ├── visualizers/     # Visualization modules
│   │   ├── __init__.py
│   │   ├── alignment_plots.py
│   │   ├── pruning_curves.py
│   │   └── layer_analysis.py
│   └── reporters.py     # Result reporting utilities
├── utils/               # General utilities
│   ├── __init__.py
│   ├── distributed.py   # Distributed computing utilities
│   ├── device.py        # Device management
│   ├── logging.py       # Logging configuration
│   ├── checkpoint.py    # Checkpointing utilities
│   └── math.py          # Mathematical utilities
└── configs/             # Configuration management
    ├── __init__.py
    ├── base.py          # Base configuration classes
    ├── experiment.py    # Experiment configurations
    └── templates/       # Configuration templates
        ├── progressive_dropout.yaml
        └── layer_analysis.yaml
```

## Key Design Principles

### 1. Protocol-Based Interfaces
All major components implement protocols (interfaces) to ensure consistency and enable easy extension:

```python
from typing import Protocol, Optional
import torch

class AlignmentMetric(Protocol):
    """Protocol for all alignment metrics."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def requires_inputs(self) -> bool: ...
    
    @property
    def requires_weights(self) -> bool: ...
    
    @property
    def requires_outputs(self) -> bool: ...
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor: ...
```

### 2. Distributed Computing Support
Built-in support for multi-GPU training and metric computation:

```python
# Automatic distributed reduction in metrics
class DistributedMetricComputer:
    def compute_metrics(self, model, data_loader, metrics):
        # Computes metrics in parallel across GPUs
        # Automatically handles reduction and synchronization
        pass
```

### 3. Lazy Loading and Memory Management
Efficient memory usage with automatic CPU offloading for large operations:

```python
class MemoryAwareMetric:
    def compute(self, inputs, weights, force_cpu_for_large_ops=True):
        if force_cpu_for_large_ops and inputs.numel() > threshold:
            # Automatically move to CPU for large matrix operations
            return self._compute_on_cpu(inputs, weights)
        return self._compute_on_device(inputs, weights)
```

### 4. Composable Components
All components are designed to be easily composed:

```python
# Example: Combining metrics with different aggregation strategies
metric = RayleighQuotient()
aggregator = LayerwiseAggregator(operation="mean")
analyzer = AlignmentAnalyzer(metric=metric, aggregator=aggregator)
```

## Migration Guide

### Converting Existing Code

1. **Metrics Migration**:
   ```python
   # Old style
   from alignment.alignment_metrics import AlignmentMetrics
   rq_scores = AlignmentMetrics.RQ(inputs, weights)
   
   # New style
   from alignment_refactor.metrics.alignment import RayleighQuotient
   metric = RayleighQuotient()
   rq_scores = metric.compute(inputs=inputs, weights=weights)
   ```

2. **Model Wrapping**:
   ```python
   # Old style
   from alignment.models import AlignmentNetwork
   net = AlignmentNetwork(model, alignment_layers=layers)
   
   # New style
   from alignment_refactor.models import ModelWrapper
   wrapper = ModelWrapper(model, tracked_layers=layers)
   ```

3. **Experiment Running**:
   ```python
   # Old style
   results = progressive_dropout(nets, dataset, alignment)
   
   # New style
   from alignment_refactor.experiments.runners import ProgressiveDropoutExperiment
   experiment = ProgressiveDropoutExperiment(config)
   results = experiment.run(models, dataset)
   ```

## Performance Optimizations

1. **Batched Metric Computation**: All metrics support batched computation for efficiency
2. **Automatic Mixed Precision**: Support for AMP in metric computations
3. **Smart Caching**: Activation caching to avoid redundant forward passes
4. **Parallel Data Loading**: Optimized data pipeline with prefetching
5. **Memory-Aware Operations**: Automatic CPU offloading for large matrices

## Extensibility

### Adding New Metrics

1. Create a new metric class implementing the `AlignmentMetric` protocol
2. Register it in the metric registry
3. The metric is automatically available in all experiments

```python
from alignment_refactor.core.registry import register_metric
from alignment_refactor.metrics.base import BaseMetric

@register_metric("my_custom_metric")
class MyCustomMetric(BaseMetric):
    def compute(self, inputs, weights, **kwargs):
        # Implementation
        pass
```

### Adding New Experiments

1. Inherit from `BaseExperiment`
2. Implement the required methods
3. Register the experiment type

```python
from alignment_refactor.experiments.base import BaseExperiment
from alignment_refactor.core.registry import register_experiment

@register_experiment("my_experiment")
class MyExperiment(BaseExperiment):
    def setup(self):
        # Setup code
        pass
    
    def run_iteration(self, iteration):
        # Run one iteration
        pass
```

## Configuration System

The new configuration system uses structured configs with validation:

```yaml
experiment:
  type: progressive_dropout
  name: "resnet18_imagenet_pruning"
  device: "cuda"
  use_ddp: true
  seed: 42

model:
  architecture: "torchvision_resnet18"
  pretrained: true
  
metrics:
  - name: "rayleigh_quotient"
    config:
      relative: true
      force_cpu_for_large_ops: true
  - name: "mutual_information"
    config:
      method: "gaussian"
      
pruning:
  mode: "layer_wise"
  dropout_range: [0.0, 0.9]
  steps: 40
  exclude_final_layer: true
```

## Testing

The refactored codebase includes comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=alignment_refactor tests/
```

## Future Enhancements

1. **Additional Metrics**: More information-theoretic and geometric metrics
2. **Advanced Pruning**: Structured pruning, channel pruning support
3. **Visualization Dashboard**: Interactive web-based visualization
4. **AutoML Integration**: Automatic hyperparameter optimization
5. **Model Zoo**: Pre-computed alignment metrics for popular models 