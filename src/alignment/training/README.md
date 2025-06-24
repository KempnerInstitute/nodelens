# Training Module

This module provides training strategies and utilities for neural networks in the alignment framework.

## Overview

The training module contains various training strategies:
- **Base Training** (`base.py`): Standard training with comprehensive features
- **Multi-Network Training** (`multi_network.py`): Train multiple networks simultaneously

## Available Training Strategies

### 1. Base Training

The `base.py` module provides a flexible trainer with many features:

```python
from alignment.training import BaseTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    epochs=100,
    learning_rate=0.001,
    optimizer="adam",
    scheduler="cosine",
    early_stopping_patience=10,
    gradient_clip_val=1.0,
    device="cuda"
)

# Create trainer
trainer = BaseTrainer(model, config, loss_fn=nn.CrossEntropyLoss())

# Train
history = trainer.train(train_loader, val_loader)
```

### 2. Multi-Network Training

The `multi_network.py` module provides efficient simultaneous training of multiple networks:

```python
from alignment.training import train_networks_fully_tensorized

# Train multiple networks at once
networks = [create_model() for _ in range(5)]
trained_networks, history = train_networks_fully_tensorized(
    networks=networks,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    device="cuda"
)
```

## Features

### Base Trainer Features
- Multiple optimizers (Adam, SGD, AdamW)
- Learning rate schedulers (Cosine, Plateau, Step)
- Early stopping
- Gradient clipping
- Checkpoint saving
- Custom callbacks
- Metric tracking

### Multi-Network Training Features
- Efficient parallel training
- Shared data loading
- Synchronized updates
- Automatic architecture verification

## Integration with Alignment Metrics

The training module integrates seamlessly with alignment metrics:

```python
from alignment.training import BaseTrainer, TrainingConfig
from alignment.metrics import get_metric

# Define metric function
def alignment_metrics(outputs, targets):
    metric = get_metric("rayleigh_quotient")()
    score = metric.compute(outputs)
    return {"rayleigh": score}

# Train with metrics
trainer = BaseTrainer(model, config)
history = trainer.train(
    train_loader, 
    val_loader,
    metric_fn=alignment_metrics
)
```

## Custom Training Loops

You can extend the base trainer for custom behavior:

```python
from alignment.training import BaseTrainer

class AlignmentTrainer(BaseTrainer):
    def _train_epoch(self, train_loader, metric_fn=None):
        # Custom training logic
        # Can access self.model, self.optimizer, etc.
        pass
```

## Best Practices

1. **Use TrainingConfig** for consistent configuration
2. **Enable gradient clipping** for stability
3. **Use early stopping** to prevent overfitting
4. **Save checkpoints regularly**
5. **Monitor multiple metrics** during training

## Examples

### Advanced Configuration

```python
from alignment.training import BaseTrainer, TrainingConfig

config = TrainingConfig(
    epochs=200,
    learning_rate=0.01,
    optimizer="sgd",
    optimizer_kwargs={"momentum": 0.9, "weight_decay": 1e-4},
    scheduler="step",
    scheduler_kwargs={"step_size": 50, "gamma": 0.1},
    early_stopping_patience=20,
    gradient_clip_val=5.0,
    checkpoint_dir="checkpoints/experiment1"
)

trainer = BaseTrainer(model, config)
```

### Multi-Network with Callbacks

```python
def log_callback(trainer, epoch):
    print(f"Epoch {epoch}: LR = {trainer.optimizer.param_groups[0]['lr']}")

networks = [create_model() for _ in range(3)]
trained_networks, history = train_networks_fully_tensorized(
    networks=networks,
    train_loader=train_loader,
    epochs=50,
    callbacks=[log_callback]
)
```

## Future Enhancements

Planned additions to the training module:

1. **Distributed Training** (`distributed.py`)
   - Multi-GPU support
   - Multi-node training

2. **Adversarial Training** (`adversarial.py`)
   - Adversarial example generation
   - Robustness training

3. **Meta Learning** (`meta.py`)
   - MAML implementation
   - Few-shot learning

4. **Continual Learning** (`continual.py`)
   - EWC and similar methods
   - Task incremental learning

## Contributing

When adding new training strategies:

1. Inherit from a base trainer class
2. Support alignment metric tracking
3. Integrate with infrastructure module
4. Add comprehensive documentation
5. Include usage examples 