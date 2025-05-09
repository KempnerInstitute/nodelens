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
        config: ModelConfig instance (not a raw dict anymore for clarity here)
        
    Returns:
        Configured AlignmentNetwork instance
    """
    model_constructor = get_model_constructor(config.model_name)
    
    # Prepare a dictionary from ModelConfig to pass to the specific create_X function
    # The specific create_X function will then extract what it needs.
    # This avoids passing the whole ModelConfig object if create_X functions 
    # are designed to take simpler dicts, or we can pass config directly if they take ModelConfig.
    # The current refactor of create_mlp/cnn2p2 in models.py expects a dict.
    
    model_params_dict = {
        "model_name": config.model_name,
        "dropout_rate": config.dropout_rate, # Common
        "output_dim": config.output_dim,   # Common
        # MLP specific
        "input_dim": config.input_dim,
        "hidden_dims": config.hidden_dims,
        "activation": config.activation,
        # CNN2P2 specific
        "in_channels": config.in_channels,
        "conv_channels": config.conv_channels,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "paddings": config.paddings,
        "pool_kernel_size": config.pool_kernel_size,
        "pool_stride": config.pool_stride,
        "hidden_fc_dim": config.hidden_fc_dim,
        "example_input_hw": config.example_input_hw,
        # Include extra_model_params for any other values
        **config.extra_model_params 
    }
    
    # alignment_layers is a separate argument to the create_X functions
    alignment_layers_config = config.alignment_layers

    logger.info(f"Creating model '{config.model_name}' with dropout rate: {config.dropout_rate}")
    
    # Call the specific constructor with the dict of relevant model parameters and alignment_layers config
    return model_constructor(config_model=model_params_dict, alignment_layers=alignment_layers_config)


def get_available_models() -> List[str]:
    """
    Get a list of all registered model names.
    
    Returns:
        List of registered model names
    """
    return sorted(_MODEL_REGISTRY.keys())