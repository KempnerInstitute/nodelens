"""
Core utility functions for the alignment package.

This module provides common utility functions used throughout the alignment package,
including timing, logging, device management, and type conversion utilities.
"""

import os
import time
import logging
import functools
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging with the specified log level.
    
    Args:
        log_level: The logging level to use (default: INFO)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None) -> None:
    """
    Context manager for timing operations.
    
    Args:
        description: Description of the operation being timed
        logger: Logger to use for output (if None, uses print)
    
    Example:
        with timer("Data loading"):
            data = load_data()
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    
    message = f"{description} completed in {elapsed:.2f} seconds"
    if logger:
        logger.info(message)
    else:
        print(message)


def debug(msg: str) -> None:
    """
    Print a debug message with timestamp.
    
    Args:
        msg: Message to print
    """
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}")


def to_numpy(x: Union[Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a tensor to a numpy array.
    
    Args:
        x: Tensor to convert
        
    Returns:
        Numpy array
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return x


def to_tensor(x: Union[Tensor, np.ndarray], device: Optional[torch.device] = None) -> Tensor:
    """
    Convert a numpy array to a tensor.
    
    Args:
        x: Array to convert
        device: Device to put the tensor on
        
    Returns:
        PyTorch tensor
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if device is not None and isinstance(x, Tensor):
        x = x.to(device)
    return x


def check_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable (and not a string).
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is iterable and not a string
    """
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def ensure_device(device: Union[str, torch.device]) -> torch.device:
    """
    Ensure that the device is a torch.device object.
    
    Args:
        device: Device specification
        
    Returns:
        torch.device object
    """
    if isinstance(device, str):
        return torch.device(device)
    return device


T = TypeVar("T")

def timed(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """
    Decorator to time a function and return the result and elapsed time.
    
    Args:
        func: Function to time
        
    Returns:
        Tuple of (function result, elapsed time in seconds)
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[T, float]:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    
    return wrapper


def _create_mask_from_indices(shape, indices_to_zero, device):
    """
    Create a binary mask tensor with zeros at specified indices and ones elsewhere.
    
    Args:
        shape: Shape of the mask tensor
        indices_to_zero: Indices where the mask should be zero
        device: Device to create the mask on
        
    Returns:
        Binary mask tensor
    """
    # Ensure indices_to_zero is a tensor for indexing
    if not isinstance(indices_to_zero, torch.Tensor):
        # Handle cases where indices_to_zero might be a list of single-element tensors or Python numbers
        if indices_to_zero and isinstance(indices_to_zero[0], torch.Tensor) and indices_to_zero[0].numel() == 1:
            indices_to_zero = torch.tensor([idx.item() for idx in indices_to_zero], device=device, dtype=torch.long)
        elif indices_to_zero and isinstance(indices_to_zero[0], int):
            indices_to_zero = torch.tensor(indices_to_zero, device=device, dtype=torch.long)
        # If it's already a tensor but not on the right device, move it
        elif isinstance(indices_to_zero, torch.Tensor) and indices_to_zero.device != device:
            indices_to_zero = indices_to_zero.to(device)
        # If it's an empty list, create an empty long tensor
        elif not indices_to_zero:
             indices_to_zero = torch.tensor([], device=device, dtype=torch.long)
        # Add other checks if necessary, or raise error for unsupported types

    mask = torch.ones(shape[0], device=device) # Mask for the 0-th dimension
    if indices_to_zero.numel() > 0: # Check if the tensor is not empty
        # Ensure indices are within bounds for the 0-th dimension
        valid_indices = indices_to_zero[(indices_to_zero >= 0) & (indices_to_zero < shape[0])]
        if valid_indices.numel() > 0:
            mask[valid_indices] = 0.0
    
    # Expand mask to match the full tensor shape if needed (e.g., for 2D weights, 4D conv filters)
    if len(shape) > 1:
        # Reshape mask to be (shape[0], 1, 1, ...) to allow broadcasting
        mask_shape_for_expansion = [shape[0]] + [1] * (len(shape) - 1)
        mask = mask.view(mask_shape_for_expansion)
        # Expand to the target shape
        mask = mask.expand(shape)
    
    return mask 