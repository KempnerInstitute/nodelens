# Installation Guide

## Requirements

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA toolkit (optional, for GPU support)

## Installation Methods

### From Source (Recommended for Development)

Clone the repository and install in development mode:

```bash
git clone <repository-url>
cd alignment
pip install -e .
```

This installs the package in editable mode, allowing you to modify the code and see changes immediately.

### Installing with Extras

Install with additional dependencies for specific features:

```bash
# Install with visualization support
pip install -e ".[viz]"

# Install with development tools
pip install -e ".[dev]"

# Install with documentation building tools
pip install -e ".[docs]"

# Install everything
pip install -e ".[all]"
```

### From Git Repository

Install directly from the repository:

```bash
pip install git+<repository-url>
```

## Verifying Installation

Test that the installation was successful:

```python
import alignment
from alignment.core import ModelWrapper
from alignment.metrics import METRIC_REGISTRY

# List available metrics
print(METRIC_REGISTRY.list())
```

## Common Issues

### CUDA/GPU Issues

If you encounter CUDA-related errors:

1. Ensure PyTorch is installed with CUDA support:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify CUDA availability:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### Import Errors

If you get import errors:

1. Ensure you're in the correct environment
2. Check that the package is installed: `pip list | grep alignment`
3. Verify Python path includes the installation directory

## Next Steps

- See the [Quick Start Guide](quickstart.md) for basic usage
- Check out [Examples](../examples/index.md) for comprehensive demos
- Read the [API Reference](../api/index.md) for detailed documentation 