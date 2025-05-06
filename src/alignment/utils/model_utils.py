"""
Model-specific utility functions for the alignment package.

This module provides utility functions specifically for neural network models,
including functions for manipulating layers, weights, and activations.
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from alignment.utils.core import to_numpy, check_iterable

logger = logging.getLogger(__name__)


def get_device(tensor_or_module: Union[Tensor, nn.Module]) -> torch.device:
    """
    Get the device of a tensor or module.
    
    Args:
        tensor_or_module: Tensor or module to get device from
        
    Returns:
        torch.device object
    """
    if isinstance(tensor_or_module, torch.Tensor):
        return tensor_or_module.device
    elif isinstance(tensor_or_module, nn.Module):
        return next(tensor_or_module.parameters()).device
    else:
        raise TypeError(f"Expected torch.Tensor or nn.Module, got {type(tensor_or_module)}")


def set_net_mode(net: nn.Module, train: bool) -> None:
    """
    Set the network to train or eval mode.
    
    Args:
        net: Network to set mode for
        train: Whether to set to train mode (True) or eval mode (False)
    """
    if train:
        net.train()
    else:
        net.eval()


def get_maximum_strides(h_in: int, w_in: int, layer: nn.Conv2d) -> Tuple[int, int]:
    """
    Calculate the maximum output dimensions after applying a convolutional layer.
    
    Args:
        h_in: Input height
        w_in: Input width
        layer: Convolutional layer
        
    Returns:
        Tuple of (output height, output width)
    """
    # Extract layer parameters
    kernel_size = layer.kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    stride = layer.stride
    if isinstance(stride, int):
        stride = (stride, stride)
    
    padding = layer.padding
    if isinstance(padding, int):
        padding = (padding, padding)
    
    dilation = layer.dilation
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Calculate output dimensions
    h_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
    w_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
    
    return h_out, w_out


def get_unfold_params(layer: nn.Conv2d) -> dict:
    """
    Get parameters for nn.functional.unfold from a convolutional layer.
    
    Args:
        layer: Convolutional layer
        
    Returns:
        Dictionary of parameters for unfold
    """
    unfold_params = {}
    if hasattr(layer, 'stride'):
        unfold_params['stride'] = layer.stride
    if hasattr(layer, 'padding'):
        unfold_params['padding'] = layer.padding
    if hasattr(layer, 'dilation'):
        unfold_params['dilation'] = layer.dilation
    
    return unfold_params


def weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Compute a weighted average of values.
    
    Args:
        values: List of values to average
        weights: List of weights for each value
        
    Returns:
        Weighted average
    """
    if len(values) != len(weights):
        raise ValueError(f"Length mismatch: values {len(values)}, weights {len(weights)}")
    
    if not values:
        return 0.0
        
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)  # Unweighted average if weights sum to 0
        
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def remove_by_idx(tensor: torch.Tensor, indices: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Remove elements from a tensor along a specific dimension by their indices.
    
    Args:
        tensor: Input tensor
        indices: Indices to remove
        dim: Dimension along which to remove indices
        
    Returns:
        Tensor with elements removed
    """
    mask = torch.ones(tensor.size(dim), device=tensor.device, dtype=torch.bool)
    mask[indices] = False
    return tensor.index_select(dim, mask.nonzero().squeeze())


def smart_pca(matrix: Union[np.ndarray, torch.Tensor], n_components: Optional[int] = None, 
             centered: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform PCA on a matrix, returning eigenvalues and eigenvectors.
    
    Args:
        matrix: Input matrix (features × samples)
        n_components: Number of components to keep (None for all)
        centered: Whether to center the data before computing covariance
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    is_tensor = isinstance(matrix, torch.Tensor)
    device = matrix.device if is_tensor else None
    
    # Convert to numpy if tensor
    if is_tensor:
        matrix_np = to_numpy(matrix)
    else:
        matrix_np = matrix
    
    # Center data if requested
    if centered:
        matrix_np = matrix_np - matrix_np.mean(axis=1, keepdims=True)
    
    # Compute covariance matrix
    cov = np.dot(matrix_np, matrix_np.T) / (matrix_np.shape[1] - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components if specified
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        eigenvalues = torch.from_numpy(eigenvalues)
        eigenvectors = torch.from_numpy(eigenvectors)
        if device is not None:
            eigenvalues = eigenvalues.to(device)
            eigenvectors = eigenvectors.to(device)
    
    return eigenvalues, eigenvectors 