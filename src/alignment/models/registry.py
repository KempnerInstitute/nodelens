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
    # cnn_mode is now part of AlignmentConfig, which is config.alignment
    # Default to "unfold" if not found, as AlignmentNetwork expects it.
    cnn_mode_config = "unfold" # Default
    if hasattr(config, 'alignment') and hasattr(config.alignment, 'cnn_mode'):
        cnn_mode_config = config.alignment.cnn_mode
    elif hasattr(config, 'extra') and hasattr(config.extra, 'cnn_mode'): # Fallback to check old ExtraConfig location
        logger.warning("cnn_mode found in config.extra, prefer config.alignment.cnn_mode")
        cnn_mode_config = config.extra.cnn_mode

    logger.info(f"Creating model '{config.model_name}' with dropout rate: {config.dropout_rate}, cnn_mode: {cnn_mode_config}")
    
    # The create_X functions (create_mlp, create_cnn2p2) in models.py return an AlignmentNetwork
    # They themselves now receive the base_model_params and alignment_layers.
    # The AlignmentNetwork constructor is called within those create_X functions.
    # We need to ensure cnn_mode is passed to AlignmentNetwork there.
    # So, the create_X functions need to be aware of cnn_mode.
    
    # Modification: Add cnn_mode_config to model_params_dict so create_X functions can access it
    # OR, pass it as a separate arg to create_X functions if they are modified to accept it.
    # Let's pass it in model_params_dict for now for simplicity, assuming create_X will pick it up
    # if it needs to pass it to AlignmentNetwork explicitly.
    # However, AlignmentNetwork __init__ takes cnn_mode directly.
    # The create_X functions in models.py already handle AlignmentNetwork creation.
    # They need to be passed cnn_mode.

    # Let's adjust create_X in models.py to accept cnn_mode and pass it to AlignmentNetwork
    # For now, this function (create_model in registry) just calls the registered constructor.
    # The registered constructors (create_mlp, create_cnn2p2) need to be updated.

    # The constructor (e.g., create_mlp) will now be called with:
    # model_constructor(config_model=model_params_dict, alignment_layers=alignment_layers_config, cnn_mode=cnn_mode_config)
    # This requires changing the signature of all registered model constructor functions.

    # Simpler: create_X functions get model_params_dict. They build base_model.
    # They then get cnn_mode from model_params_dict (if we put it there) or from config for AlignmentNetwork.
    # For now, we assume AlignmentNetwork gets cnn_mode from its own **kwargs if passed through create_X
    # or create_X explicitly passes it.
    
    # The create_X functions (create_mlp, create_cnn2p2 etc.) in models.py
    # already take **kwargs. We can pass cnn_mode via these kwargs, and they
    # can then pass it to AlignmentNetwork.
    # Let's add cnn_mode to the kwargs passed to the model_constructor
    model_creation_kwargs = model_params_dict # Start with model specific params
    # The create_X functions in models.py are registered and take `config_model: Dict` and `alignment_layers`.
    # They then create the base_model and wrap it with AlignmentNetwork.
    # The cnn_mode should be passed to the AlignmentNetwork constructor.
    # So, the create_X functions need access to cnn_mode.
    # We will pass it as part of the config_model dict to create_X functions.
    
    config_model_with_cnn_mode = model_params_dict.copy()
    config_model_with_cnn_mode['cnn_mode'] = cnn_mode_config 

    return model_constructor(config_model=config_model_with_cnn_mode, alignment_layers=alignment_layers_config)


def get_available_models() -> List[str]:
    """
    Get a list of all registered model names.
    
    Returns:
        List of registered model names
    """
    return sorted(_MODEL_REGISTRY.keys())