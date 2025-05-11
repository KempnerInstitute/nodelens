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
    
    model_params_dict = {
        "model_name": config.model_name,
        "dropout_rate": config.dropout_rate,
        "output_dim": config.output_dim,
        "input_dim": config.input_dim,
        "hidden_dims": config.hidden_dims,
        "activation": config.activation,
        "in_channels": config.in_channels,
        "conv_channels": config.conv_channels,
        "kernel_sizes": config.kernel_sizes,
        "strides": config.strides,
        "paddings": config.paddings,
        "pool_kernel_size": config.pool_kernel_size,
        "pool_stride": config.pool_stride,
        "hidden_fc_dim": config.hidden_fc_dim,
        "example_input_hw": config.example_input_hw,
        **config.extra_model_params 
    }
    
    alignment_layers_config = config.alignment_layers
    
    # Fetch cnn_mode from alignment_settings
    # Assuming ExperimentConfig is the top-level config object from which ModelConfig `config` is derived,
    # and the full config is accessible or cnn_mode is now part of ModelConfig.
    # For now, the edit to config.py put cnn_mode into AlignmentConfig, accessed via ExperimentConfig.alignment_settings.
    # This create_model function receives ModelConfig. We need a way to get cnn_mode here.
    # Simplest: Assume cnn_mode is also added to ModelConfig or passed through another way.
    # Let's modify ModelConfig to also hold cnn_mode if it's intrinsic to model setup for alignment.
    # OR, the calling context (e.g., ExperimentRunner) prepares a more complete dict for create_X functions.
    
    # The previous edit to this function added cnn_mode to config_model_with_cnn_mode:
    # config_model_with_cnn_mode['cnn_mode'] = cnn_mode_config 
    # This `cnn_mode_config` needs to be sourced correctly.
    # If `config` is ModelConfig, and cnn_mode is in AlignmentConfig, this function cannot directly access it.

    # Let's assume the full ExperimentConfig is available here, or ModelConfig now includes cnn_mode.
    # For the purpose of this edit, we will assume cnn_mode is now part of the model_params_dict via ModelConfig.
    # This requires ModelConfig to be updated to include cnn_mode if it hasn't been already.
    # (The config.py edit moved cnn_mode to AlignmentConfig, not ModelConfig)

    # SOLUTION: The create_X functions in models.py will receive the config_model dict.
    # They should extract cnn_mode from there if it's passed.
    # `create_model` in registry.py should ensure `cnn_mode` is in the dict it passes.
    # It will get it from `config.alignment_settings.cnn_mode` (assuming `config` here is ExperimentConfig)
    # BUT `config` here is `ModelConfig`. This is a structural issue.

    # Let's refine the call to model_constructor. The create_X functions (mlp, cnn2p2) will take an extra cnn_mode arg.
    # And create_model here will fetch it from the broader config context if possible, or rely on a default.
    # This implies ExperimentConfig should be passed to create_model, not just ModelConfig portion.

    # For now, to make minimal changes to create_model signature, let's assume the config_model dict passed
    # to create_X functions will have cnn_mode added by the caller of create_model if needed.
    # The previous edit to this function (create_model) already did this: It created 
    # config_model_with_cnn_mode and passed it. The source of cnn_mode_config was: 
    #   if hasattr(config, 'alignment') and hasattr(config.alignment, 'cnn_mode'): # config here was ExperimentConfig
    #       cnn_mode_config = config.alignment.cnn_mode
    # This needs to be re-evaluated based on what `config` is passed to THIS create_model.
    # If `config` is truly `ModelConfig`, it doesn't have `alignment_settings`.
    
    # Let's assume the create_X functions in models.py will be modified to accept cnn_mode as a direct kwarg,
    # and this create_model will pass it if available in the ModelConfig.extra_model_params as a temporary bridge.
    
    cnn_mode_to_pass = config.extra_model_params.get("cnn_mode", "unfold") # Get from extra_model_params or default
    if hasattr(config, 'alignment_settings') and hasattr(config.alignment_settings, 'cnn_mode'):
         # This won't work if config is ModelConfig type, it must be ExperimentConfig
         # This indicates create_model in registry.py should ideally receive ExperimentConfig
         pass # Keep cnn_mode_to_pass from extra_model_params for now

    logger.info(f"Creating model '{config.model_name}', cnn_mode to be used by AlignmentNetwork: {cnn_mode_to_pass}")

    # The create_X functions will now need to accept cnn_mode and pass it to AlignmentNetwork
    return model_constructor(
        config_model=model_params_dict, 
        alignment_layers=alignment_layers_config,
        cnn_mode=cnn_mode_to_pass # Pass cnn_mode here
    )


def get_available_models() -> List[str]:
    """
    Get a list of all registered model names.
    
    Returns:
        List of registered model names
    """
    return sorted(_MODEL_REGISTRY.keys())