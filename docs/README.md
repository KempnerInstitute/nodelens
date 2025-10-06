# Alignment Framework Documentation

Neural network alignment analysis and intelligent pruning.

## Getting Started

- [Installation Guide](installation.md) - Setup and dependencies
- [Usage Guide](usage.md) - Running experiments with configuration files
- [User Guide](user_guide.md) - Comprehensive framework documentation

## Reference

- [API Reference](api_reference.md) - API documentation
- [Quick Reference](quick_reference.md) - Code examples
- [Configuration Template](../configs/template.yaml) - All configuration parameters

## Examples

Example configuration files in `configs/examples/`:

- `mnist_basic.yaml` - MLP analysis on MNIST
- `resnet_pruning.yaml` - ResNet pruning on CIFAR-10
- `llama3_scoring.yaml` - LLaMA scoring
- `llama3_pruning.yaml` - LLaMA pruning

Run experiments:
```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

## Quick Start

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(model)
rq = get_metric('rayleigh_quotient')

outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

scores = rq.compute(acts['layer_input'], weights['layer'])
```

## Building Documentation

```bash
cd docs
make html
# Open build/html/index.html
```
