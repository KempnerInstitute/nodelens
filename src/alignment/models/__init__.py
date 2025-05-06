"""
Models package for alignment analysis.

This package contains neural network model implementations and utilities
for model instantiation, registration, and manipulation.
"""

import os
import logging
import torch
from typing import List, Dict, Any, Union, Optional

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

logger = logging.getLogger(__name__)

def load_model(model_path: str, device: Optional[torch.device] = None) -> AlignmentNetwork:
    """
    Load a saved model from a checkpoint file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model onto
        
    Returns:
        Loaded AlignmentNetwork model
        
    Raises:
        FileNotFoundError: If model_path doesn't exist
        ValueError: If model file is invalid
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    
    try:
        # Load the model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Check if we have model configuration in the checkpoint
        if 'model_config' in checkpoint:
            # Create a new model using the saved configuration
            model = create_model(checkpoint['model_config'])
            
            # Load the state dictionary
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume it's just a state dict
            model_name = os.path.basename(model_path).split('_')[0]
            logger.warning(f"No model config found in checkpoint, assuming model type: {model_name}")
            
            # Try to create the model using default parameters
            model = get_model_constructor(model_name)()
            model.load_state_dict(checkpoint)
        
        # Move to the specified device
        if device is not None:
            model = model.to(device)
        
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {str(e)}")

def load_model_family(family_dir: str, device: Optional[torch.device] = None) -> List[AlignmentNetwork]:
    """
    Load a family of models from a directory containing model checkpoints.
    
    Args:
        family_dir: Directory containing model checkpoints
        device: Device to load the models onto
        
    Returns:
        List of loaded AlignmentNetwork models
        
    Raises:
        FileNotFoundError: If family_dir doesn't exist
        ValueError: If no valid models found
    """
    if not os.path.exists(family_dir):
        raise FileNotFoundError(f"Model family directory {family_dir} not found")
    
    # Find all model files
    model_files = [f for f in os.listdir(family_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    if not model_files:
        raise ValueError(f"No model files found in {family_dir}")
    
    # Load each model
    models = []
    for model_file in model_files:
        model_path = os.path.join(family_dir, model_file)
        try:
            model = load_model(model_path, device)
            models.append(model)
            logger.info(f"Loaded model from {model_file}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_file}: {str(e)}")
    
    if not models:
        raise ValueError(f"No valid models could be loaded from {family_dir}")
    
    logger.info(f"Successfully loaded {len(models)} models from {family_dir}")
    return models

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
    
    # Loading functions
    "load_model",
    "load_model_family",
    
    # Utilities
    "get_model_dataset_parameters",
    "get_transform_parameters"
] 