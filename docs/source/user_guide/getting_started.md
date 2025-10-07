# Getting Started with the Alignment Framework

This guide will help you get started with the Neural Network Alignment Framework, from installation to running your first experiments.

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment

# Install in development mode
pip install -e .
```

### Full Installation (with all dependencies)

```bash
pip install -e .[all]
```

This installs all optional dependencies including visualization tools and development utilities.

## Quick Start

### 1. Basic Example

The simplest way to start is with the quick demo:

```bash
python examples/quick_demo.py
```

This demonstrates:
- Creating and wrapping a model
- Computing alignment metrics
- Basic pruning operations

### 2. Standard Experiment

For a complete workflow including training, metrics, and pruning:

```bash
python examples/standard_alignment_experiment.py
```

This will:
- Train a model on MNIST
- Compute alignment metrics
- Test different pruning strategies
- Generate visualizations

Results are saved to `results/standard_experiment/`.

### 3. Comprehensive Experiment

For full control over all framework features:

```bash
# Quick test with minimal configuration
python examples/comprehensive_alignment_experiment.py --config configs/quick_test_config.yaml

# Full experiment with all options
python examples/comprehensive_alignment_experiment.py --config configs/comprehensive_alignment_config.yaml
```

## Key Concepts

### Model Wrapping

The framework uses model wrappers to track activations:

```python
from alignment.models import ModelWrapper

# Wrap any PyTorch model
wrapped_model = ModelWrapper(your_model)

# Forward pass with activation tracking
outputs, activations = wrapped_model.forward_with_activations(inputs)
```

### Computing Metrics

Access any of the 36+ metrics:

```python
from alignment.metrics import get_metric

# Get a metric class
RQMetric = get_metric('rayleigh_quotient')
metric = RQMetric()

# Compute scores
scores = metric.compute(inputs=activations, weights=layer_weights)
```

### Pruning

Apply various pruning strategies:

```python
from alignment.pruning import get_pruning_strategy, PruningConfig

# Configure pruning
config = PruningConfig(amount=0.5, pruning_mode='low')
strategy = get_pruning_strategy('magnitude', config=config)

# Apply to a layer
mask = strategy.prune(layer)
```

## Available Examples

1. **`quick_demo.py`** - Minimal introduction (no config needed)
2. **`standard_alignment_experiment.py`** - Complete workflow template
3. **`pruning_strategies_demo.py`** - Advanced pruning features
4. **`pruning_visualization_demo.py`** - Visualization capabilities
5. **`comprehensive_alignment_experiment.py`** - Full framework with YAML config

## Configuration System

The comprehensive experiment uses YAML configuration:

```yaml
# Example configuration structure
name: "my_experiment"
model_name: "resnet18"
dataset_name: "cifar10"

training_config:
  epochs: 20
  batch_size: 128
  learning_rate: 0.001

alignment_metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"

pruning_strategy: "magnitude"
pruning_config:
  amount: 0.5
```

See `configs/comprehensive_alignment_config.yaml` for all available options.

## Next Steps

1. **Explore Examples**: Start with `quick_demo.py` and work your way up
2. **Read the Metrics Guide**: Learn about available metrics in the [Metrics Reference](../METRICS_REFERENCE.md)
3. **Try Different Models**: The framework supports any PyTorch model
4. **Experiment with Pruning**: See the [Pruning Guide](pruning.md) for advanced strategies
5. **Analyze Results**: Use the built-in analysis tools for insights

## Common Patterns

### Running Multiple Experiments

```python
from alignment.experiments import GeneralAlignmentExperiment

# Run experiments with different configs
for config_path in ['config1.yaml', 'config2.yaml']:
    experiment = GeneralAlignmentExperiment.from_yaml(config_path)
    results = experiment.run()
```

### Custom Metrics

```python
from alignment.core import BaseMetric

class MyMetric(BaseMetric):
    def compute(self, inputs, weights):
        # Your metric computation
        return scores
```

### Batch Processing

```python
from alignment.dataops.processing import BatchMetricProcessor

processor = BatchMetricProcessor(metrics=['rq', 'mi'])
results = processor.process_dataset(dataloader, model)
```

## Getting Help

- Check the [API Reference](../api/index.rst) for detailed documentation
- See [Examples](../examples/index.rst) for more use cases
- Review the [Developer Guide](../developer_guide/architecture.rst) for extending the framework 