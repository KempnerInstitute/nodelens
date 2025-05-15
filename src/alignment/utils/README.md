# Alignment Utilities

This directory contains utility modules for the alignment package, organized by functionality.

## Organization

The utilities are organized into the following modules:

- **core.py**: Core utilities for logging, timing, type conversion, etc.
- **math.py**: Mathematical utilities for matrix operations, linear algebra, etc.
- **plotting.py**: Plotting utilities for visualizing experimental results

## Usage

You can import specific functions from individual modules:

```python
from alignment.utils.core import timer, to_numpy
from alignment.utils.math import orthogonalize
from alignment.utils.plotting import plot_dropout_results
```

Or import commonly used functions directly from the utils package:

```python
from alignment.utils import timer, to_numpy, plot_dropout_results
```

## Backward Compatibility

For backward compatibility, the old `alignment.utils` and `alignment.plotting` modules are maintained but marked as deprecated. They import and re-export functions from the new modules.

## Plotting Functions

The `plotting.py` module contains these key functions:

- `plot_dropout_results`: Plot results from progressive dropout experiments
- `plot_experiment_summary`: Create a comprehensive summary of experiment results
- `plot_dropout_comparison`: Compare results from different dropout methods
- `log_plots_to_wandb`: Log plot images to Weights & Biases

## Core Utilities

The `core.py` module includes utilities like:

- `timer`: Context manager for timing operations
- `timed`: Decorator for timing function execution
- `to_numpy` and `to_tensor`: Convert between NumPy arrays and PyTorch tensors
- `setup_logging`: Configure logging with appropriate levels

## Math Utilities

The `math.py` module provides functions like:

- `orthogonalize`: Orthogonalize one matrix with respect to another
- `compute_correlation_matrix`: Compute correlations between matrices
- `matrix_angles`: Calculate principal angles between subspaces
- `project_to_subspace`: Project vectors onto subspaces 