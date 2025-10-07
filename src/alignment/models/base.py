"""
Base classes for model wrappers.

This module extends the core base model with additional functionality
specific to the models module.
"""

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from alignment.core.base import BaseModel

from .hooks import HookManager

# Conditional import for layer detector (graceful fallback)
try:
    from alignment.core.layer_detector import detect_trackable_layers

    HAS_LAYER_DETECTOR = True
except ImportError:
    HAS_LAYER_DETECTOR = False

logger = logging.getLogger(__name__)


class BaseModelWrapper(BaseModel):
    """
    Extended base class for model wrappers with additional utilities.

    This class provides common functionality for all model wrappers,
    including layer discovery, weight extraction, and activation preprocessing.
    """

    def __init__(
        self,
        model: nn.Module,
        tracked_layers: Optional[List[str]] = None,
        track_inputs: bool = True,
        track_outputs: bool = True,
        flatten_activations: bool = True,
        **config: Any,
    ):
        """
        Initialize the model wrapper.

        Args:
            model: PyTorch model to wrap
            tracked_layers: List of layer names to track (None = auto-discover)
            track_inputs: Whether to track layer inputs
            track_outputs: Whether to track layer outputs
            flatten_activations: Whether to flatten activations to 2D
            **config: Additional configuration
        """
        super().__init__(model, tracked_layers, **config)
        self.track_inputs = track_inputs
        self.track_outputs = track_outputs
        self.flatten_activations = flatten_activations

        # Initialize HookManager for automatic lifecycle management
        self.hook_manager = HookManager()

        # Auto-discover trackable layers if not specified
        if not self._tracked_layers:  # Check if empty list instead of None
            self._tracked_layers = self._discover_layers()
            logger.info(f"Auto-discovered {len(self._tracked_layers)} trackable layers")
            # Re-register hooks for discovered layers
            if self._tracked_layers:
                self._register_hooks()

    def _discover_layers(self) -> List[str]:
        """
        Auto-discover layers that can be tracked for alignment.

        Uses generic LayerDetector for model-agnostic detection.

        Returns:
            List of layer names suitable for tracking
        """
        if HAS_LAYER_DETECTOR:
            # Use generic detector (model-agnostic)
            trackable_layers = detect_trackable_layers(self._model, min_neurons=1)

            logger.info(f"Used generic LayerDetector (model-agnostic)")

        else:
            # Fallback to simple type-based detection
            logger.warning("LayerDetector not available, using simple detection")

            trackable_layers = []
            trackable_types = (nn.Linear, nn.Conv2d, nn.Conv1d)

            for name, module in self._model.named_modules():
                if isinstance(module, trackable_types):
                    # Skip layers without learnable parameters
                    if hasattr(module, "weight") and module.weight is not None:
                        trackable_layers.append(name)

        return trackable_layers

    def _create_activation_hook(self, layer_name: str):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            # Store both input and output if requested
            if self.track_inputs and input is not None:
                if isinstance(input, tuple):
                    input = input[0]
                self._activation_cache[f"{layer_name}_input"] = input.detach()

            if self.track_outputs and output is not None:
                if isinstance(output, tuple):
                    output = output[0]
                # Store under both legacy and explicit output keys for compatibility
                out = output.detach()
                self._activation_cache[layer_name] = out
                self._activation_cache[f"{layer_name}_output"] = out

        return hook

    def get_layer_weights(self, layers: Optional[List[str]] = None, flatten: bool = True, include_bias: bool = False) -> Dict[str, torch.Tensor]:
        """
        Get weights for specified layers.

        Args:
            layers: Layers to get weights for (None = all tracked)
            flatten: Whether to flatten weights to 2D
            include_bias: Whether to include bias terms

        Returns:
            Dictionary mapping layer names to weight tensors
        """
        layers = layers or self._tracked_layers
        weights = {}

        for name, module in self._model.named_modules():
            if name in layers:
                if hasattr(module, "weight") and module.weight is not None:
                    weight = module.weight.detach()

                    # Flatten if requested
                    if flatten and weight.ndim > 2:
                        # For Conv layers: [out_channels, in_channels, k_h, k_w] -> [out_channels, in_features]
                        weight = weight.reshape(weight.shape[0], -1)

                    weights[name] = weight

                    # Add bias if requested
                    if include_bias and hasattr(module, "bias") and module.bias is not None:
                        weights[f"{name}_bias"] = module.bias.detach()

        return weights

    def preprocess_activations(self, activations: Dict[str, torch.Tensor], mode: str = "flatten") -> Dict[str, torch.Tensor]:
        """
        Basic activation preprocessing.

        For advanced preprocessing (CNN modes, attention, etc.),
        use the data processing module: alignment.data.processing.preprocess_layer_activations

        Args:
            activations: Raw activations from hooks
            mode: Preprocessing mode ("flatten" or "none")

        Returns:
            Preprocessed activations
        """
        if mode == "none":
            return activations

        processed = {}
        for name, activation in activations.items():
            if mode == "flatten" and activation.ndim > 2:
                # Simple flattening to [batch_size, features]
                processed[name] = activation.reshape(activation.shape[0], -1)
            else:
                processed[name] = activation

        return processed

    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific layer.

        Args:
            layer_name: Name of the layer

        Returns:
            Dictionary with layer information
        """
        module = dict(self._model.named_modules()).get(layer_name)

        if module is None:
            return {"error": f"Layer '{layer_name}' not found"}

        info = {
            "name": layer_name,
            "type": type(module).__name__,
            "trainable_params": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in module.parameters()),
        }

        # Add layer-specific info
        if hasattr(module, "weight") and module.weight is not None:
            info["weight_shape"] = list(module.weight.shape)

        if isinstance(module, nn.Linear):
            info["in_features"] = module.in_features
            info["out_features"] = module.out_features

        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            info["in_channels"] = module.in_channels
            info["out_channels"] = module.out_channels
            info["kernel_size"] = module.kernel_size
            info["stride"] = module.stride
            info["padding"] = module.padding

        return info

    def get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Get a specific layer module by name."""
        for name, module in self._model.named_modules():
            if name == layer_name:
                return module
        return None

    def apply_structured_dropout(self, dropout_masks: Dict[str, torch.Tensor], mode: str = "multiplicative", permanent: bool = False) -> None:
        """
        Apply structured dropout to specified layers.

        Args:
            dropout_masks: Binary masks for each layer
            mode: How to apply masks ("multiplicative", "zeroing")
            permanent: Whether to permanently modify weights
        """
        for layer_name, mask in dropout_masks.items():
            module = dict(self._model.named_modules()).get(layer_name)

            if module is None or not hasattr(module, "weight"):
                logger.warning(f"Cannot apply dropout to layer '{layer_name}'")
                continue

            if mode == "multiplicative":
                # Apply mask during forward pass
                if not permanent:
                    # Store original weight and apply mask temporarily
                    if not hasattr(self, "_original_weights"):
                        self._original_weights = {}
                    self._original_weights[layer_name] = module.weight.data.clone()

                    # Apply mask based on layer type
                    if isinstance(module, nn.Linear):
                        # Mask entire neurons (rows)
                        module.weight.data *= mask.unsqueeze(1)
                    elif isinstance(module, nn.Conv2d):
                        # Mask entire filters
                        module.weight.data *= mask.view(-1, 1, 1, 1)
                else:
                    # Permanently zero out weights
                    with torch.no_grad():
                        if isinstance(module, nn.Linear):
                            module.weight.data *= mask.unsqueeze(1)
                        elif isinstance(module, nn.Conv2d):
                            module.weight.data *= mask.view(-1, 1, 1, 1)

            elif mode == "zeroing":
                # Set masked weights to exactly zero
                with torch.no_grad():
                    if isinstance(module, nn.Linear):
                        module.weight.data[~mask.bool()] = 0
                    elif isinstance(module, nn.Conv2d):
                        module.weight.data[~mask.bool()] = 0

    def restore_weights(self) -> None:
        """Restore original weights after temporary dropout."""
        if hasattr(self, "_original_weights"):
            for layer_name, original_weight in self._original_weights.items():
                module = dict(self._model.named_modules()).get(layer_name)
                if module is not None:
                    module.weight.data = original_weight
            delattr(self, "_original_weights")

    def forward_with_activations(
        self, inputs: torch.Tensor, layers: Optional[List[str]] = None, **kwargs  # Allow additional kwargs for compatibility
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with automatic activation capture using HookManager.

        Args:
            inputs: Input tensor
            layers: Layers to capture (None = all tracked layers)
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Tuple of (model_output, activations_dict)

        Example:
            >>> outputs, acts = wrapper.forward_with_activations(x)
            >>> conv1_input = acts['conv1_input']
            >>> conv1_output = acts['conv1_output']
        """
        layers = layers or self._tracked_layers

        # Use HookManager's context manager for automatic cleanup
        with self.hook_manager.temporary_hooks(self._model, layers, track_inputs=self.track_inputs, track_outputs=self.track_outputs) as activations:
            outputs = self._model(inputs)

        # activations are automatically cleaned up after context
        return outputs, activations.copy()

    def capture_activations_safe(self, inputs: torch.Tensor, layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Safely capture activations with guaranteed cleanup.

        This is a convenience method that just returns activations
        without model outputs.

        Args:
            inputs: Input tensor
            layers: Layers to capture

        Returns:
            Dictionary of activations
        """
        _, activations = self.forward_with_activations(inputs, layers)
        return activations

    def __del__(self):
        """Ensure HookManager cleanup on object destruction."""
        if hasattr(self, "hook_manager"):
            self.hook_manager.cleanup()
        super().__del__()
