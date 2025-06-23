"""
Neural Network Alignment Framework

A comprehensive framework for analyzing neural network representations
through information-theoretic metrics and alignment measures.
"""

__version__ = "0.1.0"

# Core functionality
from .core.base import BaseMetric
from .core.registry import METRIC_REGISTRY
from .models.wrappers import ModelWrapper

# Metrics
from .metrics import get_metric, list_metrics

# Utils
from .utils.batch_processing import BatchMetricProcessor
from .utils.pruning import PruningUtilities, PruningConfig
from .utils.experiment_tracking import create_tracker

# Visualization
try:
    from .analysis.visualization import AlignmentVisualizer
except ImportError:
    # Visualization dependencies may not be installed
    pass

__all__ = [
    # Core
    "ModelWrapper",
    "BaseMetric", 
    "METRIC_REGISTRY",
    
    # Metrics
    "get_metric",
    "list_metrics",
    
    # Utils
    "BatchMetricProcessor",
    "PruningUtilities",
    "PruningConfig",
    "create_tracker",
    
    # Visualization
    "AlignmentVisualizer",
    
    # Version
    "__version__"
] 