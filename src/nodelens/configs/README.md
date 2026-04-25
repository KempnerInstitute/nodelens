# Configs Module

Configuration loading and management.

## Components

- `config_loader.py` - YAML configuration loading
- `ExperimentConfig` - Configuration dataclass

## Usage

```python
from nodelens.configs.config_loader import load_config

config = load_config("configs/examples/mnist_basic.yaml")
```
