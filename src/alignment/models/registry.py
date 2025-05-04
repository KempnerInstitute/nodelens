"""
Model registry module for managing model creation and registration.

This module provides a centralized registry for all models used in the alignment
experiments, allowing for consistent model instantiation based on configuration.
"""

import logging
from typing import Dict, Type, Callable, Optional, Any, Union, List, Tuple

import torch
import torch.nn as nn

from alignment.config import ModelConfig
from alignment.models.base import AlignmentNetwork


logger = logging.getLogger(__name__)


# Registry of model constructors
_MODEL_REGISTRY: Dict[str, Callable[..., AlignmentNetwork]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model constructor in the global registry.
    
    Args:
        name: Unique identifier for the model
        
    Returns:
        Decorator function that registers the model constructor
    
    Example:
        @register_model("resnet18")
        def create_resnet18(**kwargs):
            return AlignmentNetwork(...)
    """
    def decorator(fn: Callable[..., AlignmentNetwork]) -> Callable[..., AlignmentNetwork]:
        if name in _MODEL_REGISTRY:
            logger.warning(f"Model {name} already registered, overwriting previous registration")
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model_constructor(model_name: str) -> Callable[..., AlignmentNetwork]:
    """
    Get the constructor function for a registered model.
    
    Args:
        model_name: Name of the registered model
        
    Returns:
        Constructor function for the specified model
        
    Raises:
        ValueError: If model_name is not registered
    """
    # Convert model name to lowercase for case-insensitive lookup
    model_name_lower = model_name.lower()
    
    # Create a case-insensitive registry lookup
    registry_lower = {k.lower(): v for k, v in _MODEL_REGISTRY.items()}
    
    if model_name_lower not in registry_lower:
        available_models = ', '.join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {available_models}")
    
    return registry_lower[model_name_lower]


def create_model(config: ModelConfig) -> AlignmentNetwork:
    """
    Create a model instance from configuration.
    
    Args:
        config: Model configuration containing model type and parameters
        
    Returns:
        Configured AlignmentNetwork instance
    """
    model_constructor = get_model_constructor(config.model_name)
    
    # Extract the necessary parameters for the model constructor
    model_params = {
        "dropout_rate": config.dropout_rate,
        "alignment_layers": config.alignment_layers
    }
    
    logger.info(f"Creating model '{config.model_name}' with dropout rate: {config.dropout_rate}")
    return model_constructor(**model_params)


def get_available_models() -> List[str]:
    """
    Get a list of all registered model names.
    
    Returns:
        List of registered model names
    """
    return sorted(_MODEL_REGISTRY.keys())