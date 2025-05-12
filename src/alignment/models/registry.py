"""
Model registry module for managing model creation and registration.

This module provides a centralized registry for all models used in the alignment
experiments, allowing for consistent model instantiation based on configuration.
"""

import logging
from typing import Dict, Type, Callable, Optional, Any, Union, List, Tuple

import torch
import torch.nn as nn
import torchvision # NEW: Import torchvision

from alignment.config import ModelConfig, MLPParamsConfig, CNN2P2ParamsConfig, ExternalModelParamsConfig
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
    
    # Allow special prefixes for external models even if not explicitly registered
    if model_name_lower not in registry_lower and not (
        model_name_lower.startswith("torchvision_") or \
        model_name_lower.startswith("hf_") or \
        model_name_lower == "external"
    ):
        available_models = ', '.join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(f"Model '{model_name}' not found in registry. Available models: {available_models}")
    
    # If it's a registered model, return its constructor
    if model_name_lower in registry_lower:
    return registry_lower[model_name_lower]
    else:
        # It's an external model type (torchvision_, hf_, external)
        # create_model will handle the actual loading for these.
        return None # Signal to create_model that this is an external type to be handled there


def create_model(config: ModelConfig) -> AlignmentNetwork:
    """
    Create a model instance from configuration.
    
    Args:
        config: ModelConfig instance
        
    Returns:
        Configured AlignmentNetwork instance
    """
    model_name_lower = config.model_name.lower()
    base_model_instance: Optional[nn.Module] = None # Type hint for clarity
    alignment_layers_config = config.alignment_layers
    
    # Resolve cnn_mode for AlignmentNetwork wrapper from ModelConfig.cnn_mode
    # Default to "unfold" if not specified in ModelConfig or is None.
    cnn_mode_for_alignment_network = config.cnn_mode if config.cnn_mode is not None else "unfold"

    # --- Handle Model Creation based on model_name --- 
    if model_name_lower == "mlp":
        if not config.mlp_params:
            raise ValueError("model_name is MLP, but mlp_params are not configured in ModelConfig.")
        mlp_p = config.mlp_params
        # Call the registered MLP constructor (e.g., create_mlp from models.py)
        # It should expect common params + specific MLP params.
        model_constructor = get_model_constructor("MLP") # Case-sensitive, ensure registry matches
        base_model_instance = model_constructor(
            output_dim=config.output_dim,
            dropout_rate=config.dropout_rate, # Pass common dropout rate
            input_dim=mlp_p.input_dim,
            hidden_dims=mlp_p.hidden_dims,
            activation=mlp_p.activation,
            # Pass cnn_mode for AlignmentNetwork wrapper if create_mlp returns base model
            # OR if create_mlp itself instantiates AlignmentNetwork, it should use this.
            cnn_mode_for_wrapper=cnn_mode_for_alignment_network, 
            extra_params=config.extra_model_params
        )

    elif model_name_lower == "cnn2p2":
        if not config.cnn2p2_params:
            raise ValueError("model_name is CNN2P2, but cnn2p2_params are not configured in ModelConfig.")
        cnn_p = config.cnn2p2_params
        model_constructor = get_model_constructor("CNN2P2")
        base_model_instance = model_constructor(
            output_dim=config.output_dim,
            dropout_rate=config.dropout_rate,
            in_channels=cnn_p.in_channels,
            conv_channels=cnn_p.conv_channels,
            kernel_sizes=cnn_p.kernel_sizes,
            strides=cnn_p.strides,
            paddings=cnn_p.paddings,
            pool_kernel_size=cnn_p.pool_kernel_size,
            pool_stride=cnn_p.pool_stride,
            hidden_fc_dim=cnn_p.hidden_fc_dim,
            example_input_hw=cnn_p.example_input_hw,
            cnn_mode_for_wrapper=cnn_mode_for_alignment_network,
            extra_params=config.extra_model_params
        )

    elif model_name_lower.startswith("torchvision_") or \
         model_name_lower.startswith("hf_") or \
         model_name_lower == "external" or \
         config.external_params is not None:
        
        if not config.external_params:
            # __post_init__ in ModelConfig should have tried to infer this if model_name was specific.
            # If it's still None, then it's a configuration error.
            raise ValueError(
                f"External model '{config.model_name}' indicated, but external_params are missing in ModelConfig."
            )
        ext_p = config.external_params
        logger.info(f"Loading external model: {ext_p.name_or_path} from source: {ext_p.source}")

        if ext_p.source == "torchvision":
            try:
                weights_arg = torchvision.models.Weights.DEFAULT if ext_p.pretrained else None
                if not hasattr(torchvision.models, ext_p.name_or_path):
                    weights_enum_name = f"{ext_p.name_or_path.capitalize()}_Weights"
                    if not hasattr(torchvision.models, weights_enum_name) and not hasattr(torchvision.models, ext_p.name_or_path.lower()):
                        # Try lowercase for model name as a fallback for get_model
                        if not hasattr(torchvision.models, ext_p.name_or_path.lower()):
                            raise ValueError(f"Torchvision model/weights '{ext_p.name_or_path}' not found.")
                        else:
                            ext_p.name_or_path = ext_p.name_or_path.lower()
                
                base_model_instance = torchvision.models.get_model(ext_p.name_or_path, weights=weights_arg)
                logger.info(f"Loaded torchvision model '{ext_p.name_or_path}' with pretrained={ext_p.pretrained}")
                
                classifier_attr_names = ['fc', 'classifier'] 
                replaced_classifier = False
                for attr_name in classifier_attr_names:
                    if hasattr(base_model_instance, attr_name):
                        classifier_layer = getattr(base_model_instance, attr_name)
                        if isinstance(classifier_layer, nn.Linear):
                            if classifier_layer.out_features != config.output_dim:
                                in_features = classifier_layer.in_features
                                setattr(base_model_instance, attr_name, nn.Linear(in_features, config.output_dim))
                                logger.info(f"Replaced '{attr_name}' of {ext_p.name_or_path} for {config.output_dim} outputs.")
                                replaced_classifier = True; break
                        elif isinstance(classifier_layer, nn.Sequential):
                            for i in range(len(classifier_layer) - 1, -1, -1):
                                if isinstance(classifier_layer[i], nn.Linear):
                                    if classifier_layer[i].out_features != config.output_dim:
                                        in_features = classifier_layer[i].in_features
                                        classifier_layer[i] = nn.Linear(in_features, config.output_dim)
                                        logger.info(f"Replaced Linear in '{attr_name}' Sequential of {ext_p.name_or_path} for {config.output_dim} outputs.")
                                        replaced_classifier = True
                                    break 
                            if replaced_classifier: break
                if not replaced_classifier:
                    logger.warning(f"Classifier for {ext_p.name_or_path} not automatically replaced. Output dim may be {base_model_instance.fc.out_features if hasattr(base_model_instance, 'fc') else 'unknown'} vs configured {config.output_dim}.")

            except Exception as e:
                logger.error(f"Failed to load torchvision model '{ext_p.name_or_path}': {e}"); raise
        
        elif ext_p.source == "huggingface_transformers":
            try:
                from transformers import AutoModel 
                base_model_instance = AutoModel.from_pretrained(ext_p.name_or_path)
                logger.info(f"Loaded Hugging Face model '{ext_p.name_or_path}'.")
                logger.warning(f"HF model '{ext_p.name_or_path}' output adaptation to output_dim={config.output_dim} not auto-handled.")

            except ImportError:
                logger.error("transformers library not installed for Hugging Face models."); raise
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model '{ext_p.name_or_path}': {e}"); raise
        else:
            raise ValueError(f"Unsupported external_model_source: {ext_p.source}")

        if base_model_instance and ext_p.freeze_feature_extractor:
            logger.info(f"Freezing feature extractor for {ext_p.name_or_path}.")
            num_frozen_params = 0
            for param_name, param in base_model_instance.named_parameters():
                is_classifier_param = any(clf_name in param_name for clf_name in ['fc', 'classifier', 'pooler'])
                if not is_classifier_param and param.requires_grad:
                    param.requires_grad = False; num_frozen_params += param.numel()
                elif is_classifier_param: param.requires_grad = True
            logger.info(f"Froze {num_frozen_params} parameters.")

    else:
        # Fallback or error if model_name is not recognized
        raise ValueError(f"Unknown model_name '{config.model_name}' or missing specific parameter block (e.g., mlp_params, cnn2p2_params, external_params) in ModelConfig.")

    # --- Wrap with AlignmentNetwork --- 
    # If internal constructors (create_mlp, create_cnn2p2) return base nn.Module, wrap it here.
    # If they already return AlignmentNetwork, this check handles it.
    if isinstance(base_model_instance, AlignmentNetwork):
        # If already an AlignmentNetwork, ensure its cnn_mode is consistent or log warning
        if base_model_instance.cnn_mode != cnn_mode_for_alignment_network:
            logger.warning(
                f"Internal constructor for '{config.model_name}' returned AlignmentNetwork with cnn_mode='{base_model_instance.cnn_mode}', "
                f"but resolved cnn_mode_for_alignment_network is '{cnn_mode_for_alignment_network}'. Keeping constructor's version."
            )
        return base_model_instance 
    elif base_model_instance is not None:
        return AlignmentNetwork(
            base_model=base_model_instance,
            alignment_layer_names=alignment_layers_config,
            cnn_mode=cnn_mode_for_alignment_network # Pass the resolved cnn_mode
        )
    else:
        # This should not be reached if logic above is correct
        raise RuntimeError(f"Failed to instantiate or retrieve base_model for '{config.model_name}'")


def get_available_models() -> List[str]:
    """
    Get a list of all registered model names.
    
    Returns:
        List of registered model names
    """
    return sorted(_MODEL_REGISTRY.keys())