"""
Metrics for measuring neural network alignment.
"""

from ..core.registry import METRIC_REGISTRY

# Import all metric modules to register them
from . import rayleigh
from . import information
from . import similarity

# Import new metric modules
try:
    from . import spectral
except ImportError:
    pass

try:
    from . import task_specific
except ImportError:
    pass

# For convenience, expose the registry
__all__ = ['METRIC_REGISTRY'] 