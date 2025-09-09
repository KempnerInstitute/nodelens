# Infrastructure Module

Foundational utilities for distributed computing, storage, and configuration management.

## Components

- **Computing**: Distributed training, GPU optimization, JIT compilation
- **Storage**: Checkpointing and logging utilities
- **Configuration**: Configuration management and validation

## Quick Usage

```python
from alignment.infrastructure import (
    setup_distributed,
    CheckpointManager,
    setup_logging,
    load_config
)

# Setup infrastructure
setup_logging("experiment.log")
setup_distributed()

# Checkpointing
ckpt_manager = CheckpointManager("checkpoints/")
ckpt_manager.save(model=model, optimizer=optimizer, epoch=epoch)

# Configuration
config = load_config("config.yaml")
```

## Key Features

- **Distributed Training**: Multi-GPU and multi-node support
- **Memory Optimization**: GPU memory management and profiling
- **Checkpointing**: Automatic model state saving/loading
- **Logging**: Structured experiment logging
- **Configuration**: YAML-based configuration system

## Migration from Utils

The infrastructure module replaces the old `utils` module with better organization. 