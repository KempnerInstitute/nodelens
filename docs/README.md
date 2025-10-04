# Alignment Framework Documentation

Neural Network Alignment Analysis & Intelligent Pruning

---

## Overview

The alignment framework provides tools for:

- Computing alignment metrics (Rayleigh Quotient, mutual information)
- Analyzing neuron redundancy and synergy
- Performing structured pruning with multiple strategies
- Training with alignment tracking
- Evaluating model performance

---

## Documentation

### Getting Started

- [Installation Guide](installation.md) - Setup and dependencies
- [User Guide](user_guide.md) - Complete usage documentation

### Reference

- [API Reference](api_reference.md) - Class and method documentation
- [Quick Reference](quick_reference.md) - Code examples and patterns
- [Configuration Guide](../configs/template_master_v2.yaml) - All configuration options

### Version Information

- [Changelog](changelog.md) - Release notes and version history

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

## Examples

See `examples/` directory for complete workflows.

---

## Building Documentation

```bash
cd docs
make html
# Open build/html/index.html
```

---

## License

See LICENSE file in repository root.
