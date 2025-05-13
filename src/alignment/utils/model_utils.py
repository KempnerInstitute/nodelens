"""
Model-specific utility functions for the alignment package.

This module provides utility functions specifically for neural network models,
including functions for manipulating layers, weights, and activations.
"""

import logging
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from alignment.utils.core import to_numpy, check_iterable

logger = logging.getLogger(__name__)


# Moved from dropout.py to break circular import
def _normalize_device(device):
    """
    Normalize device specification to avoid mismatches between 'cuda' and 'cuda:0'.

    Args:
        device: Device specification

    Returns:
        Normalized device
    """
    if isinstance(device, str) and device == "cuda":
        return torch.device("cuda:0")
    elif isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


# Moved from dropout.py to break circular import
def _ensure_model_on_device(model, device):
    """
    Ensure all model parameters are on the specified device.

    Args:
        model: PyTorch model
        device: Target device
    """
    normalized_device = _normalize_device(device)  # Ensure target device is normalized
    current_param_device = None

    try:
        current_param_device = next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, nothing to do
        return

    # Check if all parameters are on the same device
    all_params_on_same_device = True
    for param in model.parameters():
        if param.device != current_param_device:
            all_params_on_same_device = False
            break

    if not all_params_on_same_device:
        # If parameters are on mixed devices, move the whole model
        model.to(normalized_device)
    elif current_param_device != normalized_device:
        # If all parameters are on one device, but it's not the target normalized device
        model.to(normalized_device)


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
    if hasattr(layer, "stride"):
        unfold_params["stride"] = layer.stride
    if hasattr(layer, "padding"):
        unfold_params["padding"] = layer.padding
    if hasattr(layer, "dilation"):
        unfold_params["dilation"] = layer.dilation

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


def smart_pca(
    matrix: Union[np.ndarray, torch.Tensor], n_components: Optional[int] = None, centered: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def enhance_cnn_configuration(config):
    """
    Enhance CNN configuration with specific parameters based on model type.

    This function detects CNN-type models and applies optimized settings.

    Args:
        config: The experiment configuration object

    Returns:
        Modified configuration with CNN-specific enhancements
    """
    # Detect CNN models from model name
    cnn_model_types = ["CNN", "CNN2P2", "ResNet", "VGG", "AlexNet"]
    is_cnn_model = any(cnn_type in config.model.model_name for cnn_type in cnn_model_types)

    if is_cnn_model:
        logger.info(f"Detected CNN model: {config.model.model_name}")

        # Set appropriate CNN mode on config.model.cnn_mode
        if "ResNet" in config.model.model_name or "VGG" in config.model.model_name:
            # More complex architectures benefit from batch_patch_combined
            config.model.cnn_mode = "batch_patch_combined"
            logger.info(f"Setting model.cnn_mode to 'batch_patch_combined' for {config.model.model_name}")
        else:
            # Simpler CNN architectures use the standard unfold approach
            # Ensure it's set if it was None, otherwise respect existing value if not ResNet/VGG
            if config.model.cnn_mode is None: 
                config.model.cnn_mode = "unfold"
            logger.info(f"Ensuring model.cnn_mode is set for {config.model.model_name} (current: '{config.model.cnn_mode}')")

    return config


# --- Functions to be moved from dropout.py ---
def _flatten_layer_weights_for_node(layer: nn.Module) -> torch.Tensor:
    """
    Flatten layer weights to standardized format for alignment computations.
    Handles various layer types including Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d.
    Args:
        layer: PyTorch module with weights
    Returns:
        Flattened weights with shape [out_channels, flattened_dims]
    """
    w = layer.weight.data
    if w.dim() == 2:
        return w  # Linear
    elif w.dim() == 3:
        return w.view(w.size(0), -1)  # Conv1d / ConvTranspose1d
    elif w.dim() == 4:
        return w.view(w.size(0), -1)  # Conv2d / ConvTranspose2d
    elif w.dim() == 5:
        return w.view(w.size(0), -1)  # Conv3d / ConvTranspose3d
    else:
        logger.warning(f"Unexpected weight shape {w.shape} in _flatten_layer_weights_for_node, returning flattened anyway.")
        return w.view(w.size(0), -1)


def process_cnn_weights(model, layer_idx: int, pruning_strategy: str = "standard") -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Process CNN weights according to architecture-specific requirements.
    This function handles special cases for different CNN architectures, such as
    skip connections in ResNet or dense connections in DenseNet.
    Args:
        model: The neural network model (typically AlignmentNetwork wrapping a base_model).
        layer_idx: Index of the layer (in model.alignment_layers) to process.
        pruning_strategy: Strategy for pruning ("standard", "uniform", "structure-aware").
    Returns:
        Tuple of (processed_weights_flat, layer_metadata_dict).
    """
    # Use parentheses for the multi-line conditional statement for clarity and parser compatibility
    if not (
        hasattr(model, "alignment_layers")
        and hasattr(model, "alignment_names")
        and layer_idx < len(model.alignment_layers)
        and layer_idx  # Check layer_idx is within bounds for alignment_layers
        < len(model.alignment_names)  # Check layer_idx is also within bounds for alignment_names (should be same length)
    ):
        # Improved error message or handling for more specific issues
        if not hasattr(model, "alignment_layers") or not hasattr(model, "alignment_names"):
            raise ValueError("Model is missing alignment_layers or alignment_names attributes.")
        if layer_idx >= len(model.alignment_layers) or layer_idx >= len(model.alignment_names):
            raise ValueError(f"layer_idx {layer_idx} is out of bounds for alignment_layers/names (length {len(model.alignment_layers)}). ")
        # Fallback for any other unexpected case that makes the condition false
        raise ValueError("Invalid model structure or layer_idx for processing CNN weights.")

    layer = model.alignment_layers[layer_idx]
    layer_name = model.alignment_names[layer_idx]

    is_conv = isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))

    if not is_conv:
        return layer.weight.data, {"type": "linear", "name": layer_name}

    w_flat = _flatten_layer_weights_for_node(layer)
    base_model = getattr(model, "base_model", None)
    model_name = type(base_model).__name__ if base_model else "UnknownBaseModel"

    metadata = {
        "type": "conv_standard",
        "name": layer_name,
        "in_channels": getattr(layer, "in_channels", None),
        "out_channels": getattr(layer, "out_channels", None),
        "kernel_size": getattr(layer, "kernel_size", None),
    }

    if "ResNet" in model_name:  # Simple check, might need more robust ResNet block identification
        is_residual = any(n_part in layer_name.lower() for n_part in ["shortcut", "downsample", "skip"])
        if is_residual and pruning_strategy == "structure-aware":
            metadata.update({"type": "conv_residual", "requires_dimensional_matching": True})
    elif "DenseNet" in model_name:  # Simple check
        metadata.update({"type": "conv_dense", "all_outputs_required": True})

    return w_flat, metadata


# --- End of functions moved from dropout.py ---
