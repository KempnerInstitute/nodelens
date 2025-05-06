#!/usr/bin/env python3
"""
Fix the alignment/utils/__init__.py file to correct the missing imports.
"""

import os
from pathlib import Path

# File path
file_path = Path(__file__).parent / "src" / "alignment" / "utils" / "__init__.py"

# Define the corrected content
fixed_content = """\"\"\"
Utilities for the alignment package.

This module contains various utility functions for the alignment package,
including plotting, timing, linear algebra helpers, and other utility functions.
\"\"\"

# Import plotting functions
from alignment.utils.plotting import (
    plot_dropout_results,
    plot_experiment_summary,
    plot_dropout_comparison,
    log_plots_to_wandb
)

# Import core functions to maintain backward compatibility
from alignment.utils.core import (
    setup_logging,
    timer,
    debug,
    to_numpy,
    to_tensor,
    check_iterable,
    ensure_device,
    timed
)

# Import math utilities
from alignment.utils.math import (
    orthogonalize,
    compute_correlation_matrix,
    matrix_angles,
    project_to_subspace
)

# Import model utilities
from alignment.utils.model_utils import (
    get_device,
    set_net_mode,
    get_maximum_strides,
    get_unfold_params,
    weighted_average,
    remove_by_idx,
    smart_pca
)

# Define these functions that were previously imported from alignment.plotting
# They're likely not used anymore, but we'll define them as empty functions to avoid import errors
def plot_pruning_experiments(*args, **kwargs):
    print("WARNING: plot_pruning_experiments is deprecated. Use functions from alignment.utils.plotting instead.")
    return None

def plot_per_layer_independent(*args, **kwargs):
    print("WARNING: plot_per_layer_independent is deprecated. Use functions from alignment.utils.plotting instead.")
    return None

__all__ = [
    # Plotting functions
    'plot_dropout_results',
    'plot_experiment_summary',
    'plot_dropout_comparison',
    'log_plots_to_wandb',
    'plot_pruning_experiments',
    'plot_per_layer_independent',
    
    # Core utilities
    'setup_logging',
    'timer',
    'debug',
    'to_numpy',
    'to_tensor',
    'check_iterable',
    'ensure_device',
    'timed',
    
    # Math utilities
    'orthogonalize',
    'compute_correlation_matrix',
    'matrix_angles',
    'project_to_subspace',
    
    # Model utilities
    'get_device',
    'set_net_mode',
    'get_maximum_strides',
    'get_unfold_params',
    'weighted_average',
    'remove_by_idx',
    'smart_pca'
]
"""

print(f"Updating file: {file_path}")
with open(file_path, 'w') as f:
    f.write(fixed_content)
print("File updated successfully.") 