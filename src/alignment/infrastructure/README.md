# Infrastructure Module

This module provides the foundational infrastructure for the alignment framework, including distributed computing, storage management, and configuration utilities.

## Overview

The infrastructure module (formerly `utils`) contains all the supporting utilities that enable the core functionality of the alignment framework. It's organized into three main areas:

1. **Computing**: Distributed training, GPU optimization, and JIT compilation
2. **Storage**: Checkpointing and logging
3. **Configuration**: Configuration management and validation

## Structure

```
infrastructure/
├── computing/
│   ├── distributed.py     # Distributed training utilities
│   └── optimized/        # Optimization utilities
│       ├── gpu.py        # GPU memory optimization
│       └── jit.py        # JIT compilation
├── storage/
│   ├── checkpoint.py     # Model checkpointing
│   └── logging.py        # Logging utilities
├── configuration/
│   └── config.py         # Configuration management
├── __init__.py
└── README.md
```

## Computing Infrastructure

### Distributed Training

Support for multi-GPU and multi-node training:

```python
from alignment.infrastructure import setup_distributed, DistributedTrainer

# Setup distributed environment
setup_distributed()

# Create distributed trainer
trainer = DistributedTrainer(
    model=model,
    optimizer=optimizer,
    device_ids=[0, 1, 2, 3]
)

# Train with automatic gradient synchronization
trainer.train(dataloader, epochs=10)
```

### GPU Optimization

Memory optimization and profiling:

```python
from alignment.infrastructure import optimize_gpu_memory, get_gpu_memory_stats

# Optimize memory usage
optimize_gpu_memory(model)

# Monitor memory
stats = get_gpu_memory_stats()
print(f"Allocated: {stats['allocated_gb']:.2f} GB")
```

### JIT Compilation

Speed up inference with JIT compilation:

```python
from alignment.infrastructure import compile_model

# Compile model for faster inference
compiled_model = compile_model(model, example_input)
```

## Storage Infrastructure

### Checkpointing

Save and restore model states:

```python
from alignment.infrastructure import CheckpointManager

# Create checkpoint manager
ckpt_manager = CheckpointManager(
    directory="checkpoints/",
    keep_last=5,
    save_best=True
)

# Save checkpoint
ckpt_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={'loss': loss, 'accuracy': acc}
)

# Load checkpoint
state = ckpt_manager.load_best()
model.load_state_dict(state['model'])
```

### Logging

Structured logging for experiments:

```python
from alignment.infrastructure import setup_logging, get_logger

# Setup logging
setup_logging(
    log_file="experiment.log",
    level="INFO",
    format="detailed"
)

# Get logger
logger = get_logger(__name__)
logger.info("Starting experiment", extra={'epoch': 1, 'lr': 0.001})
```

## Configuration Infrastructure

### Configuration Management

Flexible configuration system:

```python
from alignment.infrastructure import load_config, ExperimentConfig

# Load configuration
config = load_config("config.yaml")

# Or create programmatically
config = ExperimentConfig(
    model=ModelConfig(
        architecture="resnet50",
        pretrained=True
    ),
    data=DataConfig(
        dataset="imagenet",
        batch_size=128
    ),
    metrics=["rayleigh_quotient", "mutual_information"]
)

# Validate configuration
config.validate()
```

## Integration Examples

### Complete Training Pipeline

```python
from alignment.infrastructure import (
    setup_distributed,
    CheckpointManager,
    setup_logging,
    load_config
)

# Load configuration
config = load_config("experiment.yaml")

# Setup infrastructure
setup_logging(config.logging)
setup_distributed()

# Create checkpoint manager
ckpt_manager = CheckpointManager(config.checkpoint)

# Training loop with infrastructure
for epoch in range(config.training.epochs):
    train_loss = train_epoch(model, dataloader)
    
    # Save checkpoint
    ckpt_manager.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={'loss': train_loss}
    )
    
    # Log progress
    logger.info(f"Epoch {epoch}: loss={train_loss:.4f}")
```

## Best Practices

1. **Always use configuration files** for reproducibility
2. **Enable checkpointing** for long-running experiments
3. **Use distributed training** for large models
4. **Monitor GPU memory** to optimize batch sizes
5. **Setup logging early** in your experiments

## Migration from Utils

If you're migrating from the old `utils` module:

```python
# Old
from alignment.utils.batch_processing import BatchMetricProcessor
from alignment.utils.distributed import setup_distributed

# New
from alignment.data.processing import BatchMetricProcessor
from alignment.infrastructure import setup_distributed
```

## Future Enhancements

- Cloud storage integration for checkpoints
- Advanced profiling tools
- Automatic hyperparameter optimization
- Resource monitoring and alerts 