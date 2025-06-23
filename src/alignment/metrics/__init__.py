"""
Metrics for measuring neural network alignment.
"""

from ..core.registry import METRIC_REGISTRY

# Import all metric modules to register them
from . import rayleigh
from . import information
from . import similarity

# Import all metric modules to trigger registration via decorators
from . import spectral
from . import information
from . import task_specific

# For convenience, expose the registry
__all__ = ['METRIC_REGISTRY'] 