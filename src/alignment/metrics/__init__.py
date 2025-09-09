"""
Metrics for measuring neural network alignment.
"""

from ..core.registry import METRIC_REGISTRY

# Import all metric modules to register them
from . import rayleigh
from . import information
from .information import pairwise_gaussian  # Ensure registration side-effects
from . import similarity
from . import spectral
from . import task_specific
from .information import gaussian_pid  # Register gaussian PID synergy


def get_metric(name: str):
    """
    Get a metric class by name.
    
    Args:
        name: Name of the metric
        
    Returns:
        Metric class (not instance)
    """
    return METRIC_REGISTRY.get(name)


def list_metrics():
    """
    List all available metrics.
    
    Returns:
        List of metric names
    """
    return METRIC_REGISTRY.list()


# For convenience, expose the registry and functions
__all__ = ['METRIC_REGISTRY', 'get_metric', 'list_metrics'] 