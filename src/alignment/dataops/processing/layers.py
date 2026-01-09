"""
Layer preprocessing utilities for different neural network architectures.

This module provides preprocessing strategies for various layer types,
including convolutional, linear, and other specialized layers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LayerPreprocessor(ABC):
    """Abstract base class for layer-specific preprocessing."""

    @abstractmethod
    def preprocess(self, activation: torch.Tensor, layer: nn.Module, is_input: bool = True, **kwargs: Any) -> torch.Tensor:
        """
        Preprocess activation tensor for a specific layer type.

        Args:
            activation: The activation tensor to preprocess
            layer: The layer module (for accessing parameters)
            is_input: Whether this is input activation (True) or output (False)
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed activation tensor
        """
        pass

    @abstractmethod
    def get_output_shape(self, input_shape: Tuple[int, ...], layer: nn.Module) -> Tuple[int, ...]:
        """Get the expected output shape after preprocessing."""
        pass


class LinearPreprocessor(LayerPreprocessor):
    """Preprocessor for linear/fully-connected layers."""

    def preprocess(self, activation: torch.Tensor, layer: nn.Module, is_input: bool = True, **kwargs: Any) -> torch.Tensor:
        """
        Preprocess linear layer activations.

        For linear layers, we simply ensure the tensor is 2D.
        """
        if activation.ndim > 2:
            # Flatten all dimensions except batch
            return activation.reshape(activation.shape[0], -1)
        return activation

    def get_output_shape(self, input_shape: Tuple[int, ...], layer: nn.Module) -> Tuple[int, ...]:
        """Get output shape for linear layer preprocessing."""
        batch_size = input_shape[0]
        flattened_size = 1
        for dim in input_shape[1:]:
            flattened_size *= dim
        return (batch_size, flattened_size)


class CNNPreprocessor(LayerPreprocessor):
    """Preprocessor for convolutional layers with multiple modes."""

    def __init__(self, mode: str = "unfold"):
        """
        Initialize CNN preprocessor.

        Args:
            mode: Preprocessing mode
                - "unfold": Unfold spatial dimensions and flatten
                - "patchwise": Keep patches separate [batch, features, patches]
                - "batch_patch_combined": Combine batch and patch dimensions
        """
        self.mode = mode
        if mode not in ["unfold", "patchwise", "batch_patch_combined"]:
            raise ValueError(f"Unknown CNN preprocessing mode: {mode}")

    def preprocess(self, activation: torch.Tensor, layer: nn.Module, is_input: bool = True, **kwargs: Any) -> torch.Tensor:
        """
        Preprocess convolutional layer activations based on mode.

        Args:
            activation: Input tensor [batch, channels, height, width]
            layer: Conv2d layer module
            is_input: Whether this is input or output activation
            **kwargs: Additional parameters (e.g., unfold parameters)

        Returns:
            Preprocessed tensor based on mode
        """
        if not isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            raise ValueError(f"Expected Conv layer, got {type(layer)}")

        if activation.ndim != 4 and isinstance(layer, nn.Conv2d):
            raise ValueError(f"Expected 4D tensor for Conv2d, got {activation.ndim}D")

        if self.mode == "unfold":
            return self._unfold_mode(activation, layer, is_input)
        elif self.mode == "patchwise":
            return self._patchwise_mode(activation, layer, is_input)
        elif self.mode == "batch_patch_combined":
            return self._batch_patch_combined_mode(activation, layer, is_input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _unfold_mode(self, activation: torch.Tensor, layer: nn.Module, is_input: bool) -> torch.Tensor:
        """
        Unfold mode: unfold spatial dimensions and flatten.

        Returns:
            Tensor of shape [batch_size * num_patches, features]
        """
        if isinstance(layer, nn.Conv2d):
            if activation.ndim != 4:
                raise ValueError(f"Expected 4D tensor for Conv2d, got {activation.ndim}D")

            b, c, h, w = activation.shape

            if is_input:
                # Unfold based on the layer's kernel parameters so feature dimension matches weight flattening
                unfold_params = self._get_unfold_params(layer)
                unfolded = torch.nn.functional.unfold(activation, kernel_size=layer.kernel_size, **unfold_params)
                # [b, features, num_patches] -> [b*num_patches, features]
                unfolded = unfolded.transpose(1, 2).contiguous()
                return unfolded.view(-1, unfolded.size(2))

            # Output: treat each spatial location as a sample (node = output channel)
            # [b, c, h, w] -> [b*h*w, c]
            return activation.permute(0, 2, 3, 1).reshape(-1, c)

        if isinstance(layer, nn.Conv1d):
            if activation.ndim != 3:
                raise ValueError(f"Expected 3D tensor for Conv1d, got {activation.ndim}D")

            b, c, l = activation.shape

            if is_input:
                # Use 2D unfold trick on [b, c, 1, l] to respect stride/padding/dilation
                x4 = activation.unsqueeze(2)  # [b, c, 1, l]
                k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
                p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
                unfolded = torch.nn.functional.unfold(
                    x4,
                    kernel_size=(1, k),
                    dilation=(1, d),
                    padding=(0, p),
                    stride=(1, s),
                )  # [b, c*k, num_patches]
                unfolded = unfolded.transpose(1, 2).contiguous()
                return unfolded.view(-1, unfolded.size(2))  # [b*num_patches, c*k]

            # Output: [b, c, l] -> [b*l, c]
            return activation.permute(0, 2, 1).reshape(-1, c)

        raise ValueError(f"Expected Conv layer, got {type(layer)}")

    def _patchwise_mode(self, activation: torch.Tensor, layer: nn.Module, is_input: bool) -> torch.Tensor:
        """
        Patchwise mode: keep patches separate for patch-wise analysis.

        Returns:
            Tensor of shape [batch_size, features, num_patches]
        """
        if isinstance(layer, nn.Conv2d):
            if activation.ndim != 4:
                raise ValueError(f"Expected 4D tensor for Conv2d, got {activation.ndim}D")

            b, c, h, w = activation.shape

            if is_input:
                # Unfold to get kernel patches
                unfold_params = self._get_unfold_params(layer)
                unfolded = torch.nn.functional.unfold(activation, kernel_size=layer.kernel_size, **unfold_params)
                return unfolded  # [b, features, patches]

            # Output: reshape spatial dims to patches (node = output channel)
            return activation.reshape(b, c, h * w)  # [b, c, patches]

        if isinstance(layer, nn.Conv1d):
            if activation.ndim != 3:
                raise ValueError(f"Expected 3D tensor for Conv1d, got {activation.ndim}D")

            b, c, l = activation.shape

            if is_input:
                # Unfold 1D input into kernel patches: [b, c*k, patches]
                x4 = activation.unsqueeze(2)  # [b, c, 1, l]
                k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
                p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
                d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
                unfolded = torch.nn.functional.unfold(
                    x4,
                    kernel_size=(1, k),
                    dilation=(1, d),
                    padding=(0, p),
                    stride=(1, s),
                )
                return unfolded  # [b, c*k, patches]

            # Output: already [b, c, l] = [b, c, patches]
            return activation

        raise ValueError(f"Expected Conv layer, got {type(layer)}")

    def _batch_patch_combined_mode(self, activation: torch.Tensor, layer: nn.Module, is_input: bool) -> torch.Tensor:
        """
        Batch-patch combined mode: combine batch and patch dimensions.

        This treats all patches from all samples as one large dataset.

        Returns:
            Tensor of shape [batch_size * num_patches, features]
        """
        # This is essentially the same as unfold mode
        return self._unfold_mode(activation, layer, is_input)

    def _get_unfold_params(self, layer: nn.Module) -> Dict[str, Any]:
        """Extract unfold parameters from a convolutional layer."""
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            return {"dilation": layer.dilation, "padding": layer.padding, "stride": layer.stride}
        return {}

    def get_output_shape(self, input_shape: Tuple[int, ...], layer: nn.Module) -> Tuple[int, ...]:
        """Get expected output shape after preprocessing."""
        if isinstance(layer, nn.Conv2d):
            if len(input_shape) != 4:
                raise ValueError(f"Expected 4D input shape for Conv2d, got {len(input_shape)}D")
            b, c, h, w = input_shape

            # Output spatial size (PyTorch conv2d formula; floor division)
            k_h, k_w = layer.kernel_size
            s_h, s_w = layer.stride
            p_h, p_w = layer.padding
            d_h, d_w = layer.dilation
            out_h = (h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
            out_w = (w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
            num_patches = max(0, out_h) * max(0, out_w)
            features = c * k_h * k_w

            if self.mode in {"unfold", "batch_patch_combined"}:
                return (b * num_patches, features)
            if self.mode == "patchwise":
                return (b, features, num_patches)

            raise ValueError(f"Unknown mode: {self.mode}")

        if isinstance(layer, nn.Conv1d):
            if len(input_shape) != 3:
                raise ValueError(f"Expected 3D input shape for Conv1d, got {len(input_shape)}D")
            b, c, l = input_shape
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
            d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
            out_l = (l + 2 * p - d * (k - 1) - 1) // s + 1
            num_patches = max(0, out_l)
            features = c * k

            if self.mode in {"unfold", "batch_patch_combined"}:
                return (b * num_patches, features)
            if self.mode == "patchwise":
                return (b, features, num_patches)

            raise ValueError(f"Unknown mode: {self.mode}")

        raise ValueError(f"Expected Conv layer, got {type(layer)}")


class AttentionPreprocessor(LayerPreprocessor):
    """Preprocessor for attention/transformer layers."""

    def __init__(self, aggregate_heads: bool = True):
        """
        Initialize attention preprocessor.

        Args:
            aggregate_heads: Whether to aggregate attention heads
        """
        self.aggregate_heads = aggregate_heads

    def preprocess(self, activation: torch.Tensor, layer: nn.Module, is_input: bool = True, **kwargs: Any) -> torch.Tensor:
        """
        Preprocess attention layer activations.

        For multi-head attention, can either keep heads separate
        or aggregate them.
        """
        # Handle different attention tensor formats
        if activation.ndim == 4:  # [batch, heads, seq_len, dim]
            if self.aggregate_heads:
                # Average across heads
                return activation.mean(dim=1)
            else:
                # Reshape to [batch, seq_len, heads * dim]
                b, h, s, d = activation.shape
                return activation.permute(0, 2, 1, 3).reshape(b, s, h * d)
        elif activation.ndim == 3:  # [batch, seq_len, dim]
            return activation
        else:
            # Flatten if needed
            return activation.reshape(activation.shape[0], -1)

    def get_output_shape(self, input_shape: Tuple[int, ...], layer: nn.Module) -> Tuple[int, ...]:
        """Get output shape for attention preprocessing."""
        if len(input_shape) == 4:  # Multi-head format
            b, h, s, d = input_shape
            if self.aggregate_heads:
                return (b, s, d)
            else:
                return (b, s, h * d)
        else:
            return input_shape


# Registry of preprocessors for different layer types
PREPROCESSOR_REGISTRY = {
    nn.Linear: LinearPreprocessor,
    nn.Conv2d: CNNPreprocessor,
    nn.Conv1d: CNNPreprocessor,
    nn.MultiheadAttention: AttentionPreprocessor,
}


def get_preprocessor(layer: nn.Module, mode: Optional[str] = None, **kwargs: Any) -> LayerPreprocessor:
    """
    Get appropriate preprocessor for a given layer.

    Args:
        layer: The neural network layer
        mode: Optional mode for the preprocessor (e.g., "unfold" for CNN)
        **kwargs: Additional preprocessor parameters

    Returns:
        Appropriate LayerPreprocessor instance
    """
    layer_type = type(layer)

    # Find the appropriate preprocessor class
    preprocessor_class = None
    for registered_type, preprocessor in PREPROCESSOR_REGISTRY.items():
        if isinstance(layer, registered_type):
            preprocessor_class = preprocessor
            break

    if preprocessor_class is None:
        # Default to linear preprocessor for unknown types
        logger.warning(f"No specific preprocessor for {layer_type}, using LinearPreprocessor")
        return LinearPreprocessor()

    # Instantiate with appropriate parameters
    if preprocessor_class == CNNPreprocessor and mode is not None:
        return preprocessor_class(mode=mode, **kwargs)
    elif preprocessor_class == AttentionPreprocessor:
        return preprocessor_class(**kwargs)
    else:
        return preprocessor_class()


def preprocess_layer_activations(
    activations: Dict[str, torch.Tensor],
    layer_modules: Dict[str, nn.Module],
    mode: Optional[str] = None,
    layer_specific_modes: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess activations for multiple layers.

    Args:
        activations: Dict mapping layer names to activation tensors
        layer_modules: Dict mapping layer names to layer modules
        mode: Global preprocessing mode (e.g., "unfold" for all CNN layers)
        layer_specific_modes: Optional dict of layer-specific modes
        **kwargs: Additional preprocessing parameters

    Returns:
        Dict of preprocessed activations
    """
    preprocessed = {}
    layer_specific_modes = layer_specific_modes or {}

    for layer_name, activation in activations.items():
        # Handle input/output activations (name ends with _input or _output)
        is_input = layer_name.endswith("_input")
        is_output = layer_name.endswith("_output")
        if is_input:
            actual_layer_name = layer_name[:-6]  # Remove "_input" suffix
        elif is_output:
            actual_layer_name = layer_name[:-7]  # Remove "_output" suffix
        else:
            actual_layer_name = layer_name

        # Get the layer module
        if actual_layer_name not in layer_modules:
            logger.warning(f"Layer {actual_layer_name} not found in modules, skipping preprocessing")
            preprocessed[layer_name] = activation
            continue

        layer_module = layer_modules[actual_layer_name]

        # Determine preprocessing mode for this layer
        layer_mode = layer_specific_modes.get(actual_layer_name, mode)

        # Get appropriate preprocessor
        preprocessor = get_preprocessor(layer_module, mode=layer_mode, **kwargs)

        # Preprocess the activation
        try:
            preprocessed[layer_name] = preprocessor.preprocess(activation, layer_module, is_input=is_input, **kwargs)
        except Exception as e:
            logger.error(f"Error preprocessing {layer_name}: {e}")
            # Fall back to original activation
            preprocessed[layer_name] = activation

    return preprocessed
