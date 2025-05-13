"""
Core utility functions for the alignment package.

This module provides common utility functions used throughout the alignment package,
including timing, logging, device management, and type conversion utilities.
"""

import os
import time
import logging
import functools
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union, Dict
from contextlib import contextmanager
import sys

import numpy as np
import torch
from torch import Tensor

# --- NEW: More robust setup_logging --- 
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Store if handlers have been added to root to avoid duplication if called multiple times
_logging_handlers_added = False

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_file_path: Optional[str] = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    date_fmt: str = DEFAULT_DATE_FORMAT,
    force_debug: bool = False # Flag to explicitly force DEBUG level
):
    global _logging_handlers_added
    config = config or {}
    
    # Determine overall logging level
    log_level_str = "DEBUG" if force_debug else str(config.get("log_level", "INFO")).upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)
    if not isinstance(numeric_log_level, int): # Fallback if string is invalid
        logging.warning(f"Invalid log_level '{log_level_str}'. Defaulting to INFO.")
        numeric_log_level = logging.INFO

    root_logger = logging.getLogger() # Get the root logger
    
    # Set level on root logger *once* or update if new level is lower
    # This ensures all loggers inherit at least this level.
    if not root_logger.hasHandlers() or root_logger.level > numeric_log_level :
        root_logger.setLevel(numeric_log_level)

    # Add handlers only once to avoid duplication if setup_logging is called multiple times by different parts
    if not _logging_handlers_added:
        # Clear any pre-existing handlers from other libraries on first setup if necessary
        # For robust clean slate: 
        # for handler in root_logger.handlers[:]:
        #     root_logger.removeHandler(handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, date_fmt))
        # Console can have its own level, e.g., INFO, while file is DEBUG
        # If force_debug, make console DEBUG too, otherwise use the determined numeric_log_level
        console_handler.setLevel(logging.DEBUG if force_debug else numeric_log_level) 
        root_logger.addHandler(console_handler)

        # File Handler
        if log_file_path:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' to overwrite for each run
            file_handler.setFormatter(logging.Formatter(log_format, date_fmt))
            # File handler always gets DEBUG if force_debug, else numeric_log_level
            file_handler.setLevel(logging.DEBUG if force_debug else numeric_log_level) 
            root_logger.addHandler(file_handler)
        
        _logging_handlers_added = True

    # Log setup completion using a specific logger for this module
    # This message will now reflect the actual effective level of the root logger.
    setup_complete_logger = logging.getLogger(__name__) 
    if _logging_handlers_added or force_debug: # Log if first time or if forcing debug
        setup_complete_logger.info(
            f"Logging setup/reconfigured. Effective Root Level: {logging.getLevelName(root_logger.getEffectiveLevel())}. "
            f"File: {log_file_path if log_file_path else 'None'}."
        )
# --- End NEW setup_logging ---


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


def _create_mask_from_indices(shape: Tuple[int, ...], indices_to_zero: Union[List[int], torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Create a binary mask tensor with zeros at specified indices for the 0-th dimension,
    and ones elsewhere. The mask is then expanded to the full shape.
    
    Args:
        shape: Shape of the target mask tensor (e.g., layer.weight.shape).
        indices_to_zero: A 1D List or Tensor of unique indices in the 0-th dimension to be zeroed out.
        device: The torch.device to create the mask on.
        
    Returns:
        A binary mask tensor of the given shape.
    """
    if not shape:
        raise ValueError("Shape cannot be empty for creating a mask.")

    dim0_size = shape[0]
    mask_dim0 = torch.ones(dim0_size, device=device, dtype=torch.bool) # Start with a boolean mask for efficiency

    if isinstance(indices_to_zero, list):
        if not indices_to_zero: # Empty list
            pass # Mask remains all ones
        else:
            # Ensure indices are valid and within bounds before attempting to cast to tensor or index
            valid_indices = [idx for idx in indices_to_zero if 0 <= idx < dim0_size]
            if valid_indices:
                indices_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
                mask_dim0[indices_tensor] = False
    elif isinstance(indices_to_zero, torch.Tensor):
        if indices_to_zero.numel() > 0:
            # Ensure tensor is 1D and long/int type
            if indices_to_zero.dim() > 1:
                indices_to_zero = indices_to_zero.view(-1)
            if indices_to_zero.dtype not in [torch.long, torch.int]:
                indices_to_zero = indices_to_zero.long()
            
            # Filter out-of-bounds indices
            valid_mask = (indices_to_zero >= 0) & (indices_to_zero < dim0_size)
            valid_indices_tensor = indices_to_zero[valid_mask]
            if valid_indices_tensor.numel() > 0:
                mask_dim0[valid_indices_tensor] = False
    else:
        raise TypeError(f"indices_to_zero must be a list or a torch.Tensor, got {type(indices_to_zero)}")

    # Convert boolean mask to float (0.0 or 1.0)
    float_mask_dim0 = mask_dim0.float()
    
    if len(shape) > 1:
        # Reshape for broadcasting: (shape[0], 1, 1, ...)
        mask_shape_for_expansion = [dim0_size] + [1] * (len(shape) - 1)
        expanded_mask = float_mask_dim0.view(mask_shape_for_expansion).expand(shape)
        return expanded_mask
    else:
        return float_mask_dim0 