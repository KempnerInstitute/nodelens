"""
Neural Network Alignment Framework

A comprehensive framework for analyzing neural network representations
through information-theoretic metrics and alignment measures.
"""

__version__ = "0.1.0"

# Core functionality
from .core.base import BaseMetric
from .core.registry import METRIC_REGISTRY
from .core.wrappers import ModelWrapper

# Metrics
from .metrics import get_metric, list_metrics

# Data processing
from .data.processing import BatchMetricProcessor

# Pruning
from .pruning import PruningConfig, get_pruning_strategy

# Experiment tracking
from .experiments.tracking import create_tracker

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
    
    # Data processing
    "BatchMetricProcessor",
    
    # Pruning
    "PruningConfig",
    "get_pruning_strategy",
    
    # Experiment tracking
    "create_tracker",
    
    # Visualization
    "AlignmentVisualizer",
    
    # Version
    "__version__"
] 