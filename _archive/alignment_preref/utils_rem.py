"""
Utilities for alignment research.

DEPRECATED: This module is kept for backward compatibility.
Please use alignment.utils.core and alignment.utils.math instead.

This module contains various utility functions for the alignment package.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import from the new modules for re-export
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

from alignment.utils.math import (
    orthogonalize,
    compute_correlation_matrix,
    matrix_angles,
    project_to_subspace
)

# Show deprecation warning on import
warnings.warn(
    "The alignment.utils module is deprecated and will be removed in a future version. "
    "Please use alignment.utils.core and alignment.utils.math instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the imported modules
__all__ = [
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
    'project_to_subspace'
]