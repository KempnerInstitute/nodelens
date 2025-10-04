# Alignment Framework Documentation

Neural Network Alignment Analysis & Intelligent Pruning

---

## Getting Started

- [Installation Guide](installation.md) - Setup and dependencies
- [Usage Guide](usage.md) - How to run experiments with YAML configs
- [User Guide](user_guide.md) - Complete usage documentation

---

## Reference

- [API Reference](api_reference.md) - Class and method documentation
- [Quick Reference](quick_reference.md) - Code examples and patterns
- [Configuration Reference](../configs/template.yaml) - All parameters with options

---

## Version Information

- [Changelog](changelog.md) - Release notes and version history

---

## Examples

Working example configurations:

- `configs/examples/mnist_basic.yaml` - Simple MNIST analysis
- `configs/examples/resnet_pruning.yaml` - ResNet pruning on CIFAR-10
- `configs/examples/llama3_scoring.yaml` - LLaMA-3 per-neuron scoring
- `configs/examples/llama3_pruning.yaml` - LLaMA-3 pruning

Run: `python scripts/run_experiment.py --config [path]`

---

## Quick Example

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(model)
rq = get_metric('rayleigh_quotient')

outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

scores = rq.compute(acts['layer_input'], weights['layer'])
```

---

## Building HTML Documentation

```bash
cd docs
make html
# Open build/html/index.html
```
