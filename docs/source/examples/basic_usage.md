# Basic Usage Examples

## Getting Started

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import get_metric

# Create a model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Wrap the model
wrapped_model = ModelWrapper(model)

# Compute a metric
metric = get_metric('rayleigh_quotient')()
inputs = torch.randn(32, 784)
weights = model[0].weight
scores = metric.compute(inputs=inputs, weights=weights)
```

## See Also

- Full example: `examples/quick_demo.py`
- [Metrics Reference](../METRICS_REFERENCE.md) 