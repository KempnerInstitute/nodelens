"""
Checkpoint utilities for saving models with hooks.
"""

import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    filepath: str,
    additional_state: Optional[Dict[str, Any]] = None,
    save_hooks: bool = False
) -> None:
    """
    Save a checkpoint, handling models with hooks gracefully.
    
    Args:
        model: The model to save
        optimizer: Optimizer state to save (optional)
        epoch: Current epoch number
        filepath: Path to save the checkpoint
        additional_state: Additional state to save
        save_hooks: Whether to attempt saving the full model (with hooks)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_state is not None:
        checkpoint.update(additional_state)
    
    if save_hooks:
        # Try to save the full model, but warn about hooks
        try:
            checkpoint['model'] = model
            logger.warning("Saving full model with hooks. This may not be loadable.")
        except Exception as e:
            logger.warning(f"Failed to save full model: {e}. Saving state_dict only.")
    
    try:
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def load_checkpoint(
    filepath: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a checkpoint, handling different checkpoint formats.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        map_location: Device mapping location
        
    Returns:
        Dictionary containing the loaded checkpoint data
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # Load model state if model is provided
    if model is not None and 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model state: {e}")
            # Try to load with strict=False
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.warning("Model state loaded with strict=False")
            except Exception as e2:
                logger.error(f"Failed to load model state even with strict=False: {e2}")
    
    # Load optimizer state if optimizer is provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")
    
    return checkpoint


def save_model_for_inference(
    model: torch.nn.Module,
    filepath: str,
    remove_hooks: bool = True
) -> None:
    """
    Save a model for inference, optionally removing hooks.
    
    Args:
        model: The model to save
        filepath: Path to save the model
        remove_hooks: Whether to remove hooks before saving
    """
    if remove_hooks and hasattr(model, '_forward_hooks'):
        # Store hooks temporarily
        hooks_backup = {}
        for name, module in model.named_modules():
            if hasattr(module, '_forward_hooks') and len(module._forward_hooks) > 0:
                hooks_backup[name] = dict(module._forward_hooks)
                module._forward_hooks.clear()
        
        # Save model without hooks
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model saved for inference (hooks removed): {filepath}")
        
        # Restore hooks
        for name, module in model.named_modules():
            if name in hooks_backup:
                module._forward_hooks.update(hooks_backup[name])
    else:
        # Save model as is
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model saved for inference: {filepath}") 