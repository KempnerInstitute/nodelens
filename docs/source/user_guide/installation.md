# Installation Guide

This guide provides instructions for installing and configuring the Neural Network Alignment framework.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (optional but recommended)

## Installation Methods

### Method 1: Development Installation (Recommended)

For development or if you want to modify the code:

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
cd src/alignment_refactor
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

### Method 2: Direct Installation

If you just want to use the framework:

```bash
# Install directly from the repository
pip install git+<repository-url>#subdirectory=src/alignment_refactor
```

### Method 3: Manual Installation

If you have downloaded the source code:

```bash
cd alignment/src/alignment_refactor
pip install .
```

## Dependencies

The main dependencies are:

- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Plotting and visualization
- **tqdm**: Progress bars
- **pyyaml**: YAML configuration support
- **wandb** (optional): Experiment tracking

### Installing Dependencies

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install torch torchvision numpy scipy matplotlib tqdm pyyaml
# Optional: pip install wandb
```

## Verifying Installation

To verify your installation:

```python
import alignment_refactor
from alignment_refactor.models import ModelWrapper
from alignment_refactor.metrics import RayleighQuotient

print("Installation successful!")
```

## GPU Support

To use GPU acceleration:

1. Ensure you have CUDA installed (check with `nvidia-smi`)
2. Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Common Issues

### Import Errors

If you encounter import errors:

1. Ensure you're in the correct directory
2. Check that the package is properly installed
3. Verify Python path includes the installation directory

### CUDA Errors

If you get CUDA-related errors:

1. Check CUDA version compatibility with PyTorch
2. Ensure GPU drivers are up to date
3. Try running with `CUDA_VISIBLE_DEVICES="" python script.py` to use CPU only

### Memory Issues

For large models or datasets:

1. Reduce batch size in configuration
2. Use gradient accumulation
3. Enable mixed precision training

## Next Steps

After installation, proceed to:
- [Quick Start Guide](quickstart.md) for basic usage
- [Experiments Guide](experiments.md) for running experiments
- [Configuration Guide](configuration.md) for customization 