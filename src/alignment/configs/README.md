# Configuration Module

Python utilities for loading and validating configuration files.

## Note

Configuration YAML files are located in the top-level `configs/` directory, not here.

This module contains only Python code for:
- Loading YAML configuration files (`config_loader.py`)
- Validating configuration parameters (`config_validator.py`)
- Utility functions for config manipulation

---

## Usage

```python
from alignment.configs import load_config

config = load_config('configs/examples/resnet_pruning.yaml')
```

---

## Configuration Files

See `../../configs/` directory for:
- `template.yaml` - Complete parameter reference
- `examples/` - Ready-to-use example configs

See `../../configs/README.md` for configuration documentation.
