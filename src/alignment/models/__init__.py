"""
Models package for alignment analysis.

This package contains neural network model implementations and utilities
for model instantiation, registration, and manipulation.
"""

# Import registry components
from alignment.models.registry import (
    register_model,
    create_model,
    get_model_constructor,
    get_available_models
)

# Import model implementations to ensure they're registered
from alignment.models.models import (
    MLP,
    CNN2P2,
    create_mlp,
    create_cnn2p2,
    create_alexnet,
    get_model_dataset_parameters,
    get_transform_parameters
)

# Import base classes and utilities
from alignment.models.base import AlignmentNetwork

# Export public API
__all__ = [
    # Registry
    "register_model",
    "create_model",
    "get_model_constructor",
    "get_available_models",
    
    # Models
    "MLP",
    "CNN2P2",
    "AlignmentNetwork",
    
    # Utilities
    "get_model_dataset_parameters",
    "get_transform_parameters"
] 