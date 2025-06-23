"""
Utilities for the alignment package.

This module contains various utility functions for the alignment package,
including plotting, timing, linear algebra helpers, and other utility functions.
"""

# Import plotting functions
from alignment_refac1.utils.plotting import (
    plot_dropout_results,
    plot_experiment_summary,
    plot_dropout_comparison,
    log_plots_to_wandb
)

# Import core functions to maintain backward compatibility
from alignment_refac1.utils.core import (
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
from alignment_refac1.utils.math import (
    orthogonalize,
    compute_correlation_matrix,
    matrix_angles,
    project_to_subspace
)

# Import model utilities
from alignment_refac1.utils.model_utils import (
    get_device,
    set_net_mode,
    get_maximum_strides,
    get_unfold_params,
    weighted_average,
    remove_by_idx,
    smart_pca
)

# Removed deprecated plotting functions: plot_pruning_experiments, plot_per_layer_independent
# They were stubs printing warnings. Their functionality should be covered by other plotting utilities.

__all__ = [
    # Plotting functions
    'plot_dropout_results',
    'plot_experiment_summary',
    'plot_dropout_comparison',
    'log_plots_to_wandb',
    
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
