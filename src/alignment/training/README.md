# Training Module

Training strategies and utilities for neural networks in the alignment framework.

## Components

- **BaseTrainer** - Standard training with comprehensive features
- **Multi-Network Training** - Train multiple networks simultaneously
- **ExperimentTrainer** - Extended trainer for experiments

## Quick Usage

```python
from alignment.training import BaseTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    epochs=100,
    learning_rate=0.001,
    optimizer="adam",
    scheduler="cosine",
    early_stopping_patience=10
)

# Train model
trainer = BaseTrainer(model, config, loss_fn=nn.CrossEntropyLoss())
history = trainer.train(train_loader, val_loader)
```

## Multi-Network Training

```python
from alignment.training import train_networks_fully_tensorized

# Train multiple networks efficiently
networks = [create_model() for _ in range(5)]
trained_networks, history = train_networks_fully_tensorized(
    networks=networks,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10
)
```

## Key Features

- Multiple optimizers (Adam, SGD, AdamW)
- Learning rate schedulers
- Early stopping and gradient clipping
- Checkpoint saving and custom callbacks
- Integration with alignment metrics
- Efficient parallel training for multiple networks

## Custom Training

Extend the base trainer for custom behavior:

```python
from alignment.training import BaseTrainer

class CustomTrainer(BaseTrainer):
    def _train_epoch(self, train_loader, metric_fn=None):
        # Custom training logic
        pass
``` 