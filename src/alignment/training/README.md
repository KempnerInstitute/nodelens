# Training Module

Training utilities and callbacks.

## Components

- `BaseTrainer` - Training loop with metric tracking
- `AlignmentMetricsCallback` - Track alignment during training

## Usage

```python
from alignment.training.callbacks import AlignmentMetricsCallback

callback = AlignmentMetricsCallback(
    metrics={'rq': get_metric('rayleigh_quotient')},
    layers=['conv1'],
    frequency=100
)

# In training loop
callback.on_batch_end(wrapper, inputs, targets, step)
history = callback.get_history()
```
