"""
Base classes for model wrappers.

This module extends the core base model with additional functionality
specific to the models module.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from collections import OrderedDict
import logging

from alignment.core.base import BaseModel

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
        cnn_mode: str = "unfold",
        **config: Any
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model: PyTorch model to wrap
            tracked_layers: List of layer names to track (None = auto-discover)
            track_inputs: Whether to track layer inputs
            track_outputs: Whether to track layer outputs
            flatten_activations: Whether to flatten activations to 2D
            cnn_mode: How to preprocess CNN layers ("unfold", "patchwise", "batch_patch_combined")
            **config: Additional configuration
        """
        super().__init__(model, tracked_layers, **config)
        self.track_inputs = track_inputs
        self.track_outputs = track_outputs
        self.flatten_activations = flatten_activations
        self.cnn_mode = cnn_mode
        
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
        
        Returns:
            List of layer names suitable for tracking
        """
        trackable_layers = []
        trackable_types = (nn.Linear, nn.Conv2d, nn.Conv1d)
        
        for name, module in self._model.named_modules():
            if isinstance(module, trackable_types):
                # Skip layers without learnable parameters
                if hasattr(module, 'weight') and module.weight is not None:
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
                self._activation_cache[layer_name] = output.detach()
        
        return hook
    
    def get_layer_weights(
        self,
        layers: Optional[List[str]] = None,
        flatten: bool = True,
        include_bias: bool = False
    ) -> Dict[str, torch.Tensor]:
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
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.detach()
                    
                    # Flatten if requested
                    if flatten and weight.ndim > 2:
                        # For Conv layers: [out_channels, in_channels, k_h, k_w] -> [out_channels, in_features]
                        weight = weight.reshape(weight.shape[0], -1)
                    
                    weights[name] = weight
                    
                    # Add bias if requested
                    if include_bias and hasattr(module, 'bias') and module.bias is not None:
                        weights[f"{name}_bias"] = module.bias.detach()
        
        return weights
    
    def _get_unfold_params(self, layer: nn.Module) -> Dict[str, Any]:
        """Get unfold parameters from a convolutional layer."""
        if isinstance(layer, (nn.Conv2d, nn.Conv1d)):
            return {
                'dilation': layer.dilation,
                'padding': layer.padding,
                'stride': layer.stride
            }
        return {}
    
    def preprocess_activations(
        self,
        activations: Dict[str, torch.Tensor],
        mode: Optional[str] = None,
        layer_modules: Optional[Dict[str, nn.Module]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess activations for metric computation based on CNN mode.
        
        Args:
            activations: Raw activations from hooks
            mode: Override preprocessing mode (uses self.cnn_mode if None)
            layer_modules: Optional dict of layer modules for unfold params
            
        Returns:
            Preprocessed activations
        """
        mode = mode or self.cnn_mode
        processed = {}
        
        # Get layer modules if not provided
        if layer_modules is None:
            layer_modules = dict(self._model.named_modules())
        
        for name, activation in activations.items():
            # Handle input activations (name ends with _input)
            is_input = name.endswith("_input")
            layer_name = name.replace("_input", "") if is_input else name
            module = layer_modules.get(layer_name)
            
            # For conv layers with spatial dimensions
            if activation.ndim == 4 and isinstance(module, nn.Conv2d):
                if mode == "unfold" or self.cnn_mode == "unfold":
                    # Unfold spatial dimensions
                    b, c, h, w = activation.shape
                    
                    # For inputs, we unfold based on the layer's kernel params
                    if is_input:
                        unfold_params = self._get_unfold_params(module)
                        unfolded = torch.nn.functional.unfold(
                            activation,
                            kernel_size=module.kernel_size,
                            **unfold_params
                        )
                        # Flatten: [b, features*kernel_size, num_patches] -> [b*num_patches, features]
                        unfolded = unfolded.transpose(1, 2).contiguous()
                        processed[name] = unfolded.view(-1, unfolded.size(2))
                    else:
                        # For outputs, just flatten spatial dims
                        processed[name] = activation.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)
                        
                elif mode == "patchwise" or self.cnn_mode == "patchwise":
                    # Keep patches separate: [b, c, h, w] -> [b, features, patches]
                    if is_input:
                        unfold_params = self._get_unfold_params(module)
                        unfolded = torch.nn.functional.unfold(
                            activation,
                            kernel_size=module.kernel_size,
                            **unfold_params
                        )
                        processed[name] = unfolded  # [b, features, patches]
                    else:
                        b, c, h, w = activation.shape
                        processed[name] = activation.reshape(b, c, h * w)
                        
                elif mode == "batch_patch_combined" or self.cnn_mode == "batch_patch_combined":
                    # Combine batch and patch dimensions
                    if is_input:
                        unfold_params = self._get_unfold_params(module)
                        unfolded = torch.nn.functional.unfold(
                            activation,
                            kernel_size=module.kernel_size,
                            **unfold_params
                        )
                        # [b, features, patches] -> [b*patches, features]
                        unfolded = unfolded.transpose(1, 2).contiguous()
                        processed[name] = unfolded.view(-1, unfolded.size(2))
                    else:
                        b, c, h, w = activation.shape
                        processed[name] = activation.permute(0, 2, 3, 1).reshape(-1, c)
                        
                elif mode == "flatten" or self.flatten_activations:
                    # Simple flattening
                    processed[name] = activation.reshape(activation.shape[0], -1)
                else:
                    processed[name] = activation
                    
            elif activation.ndim > 2 and (mode == "flatten" or self.flatten_activations):
                # Flatten non-conv activations if requested
                processed[name] = activation.reshape(activation.shape[0], -1)
            else:
                # Keep as is for 2D activations or when not flattening
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
        if hasattr(module, 'weight') and module.weight is not None:
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
    
    def apply_structured_dropout(
        self,
        dropout_masks: Dict[str, torch.Tensor],
        mode: str = "multiplicative",
        permanent: bool = False
    ) -> None:
        """
        Apply structured dropout to specified layers.
        
        Args:
            dropout_masks: Binary masks for each layer
            mode: How to apply masks ("multiplicative", "zeroing")
            permanent: Whether to permanently modify weights
        """
        for layer_name, mask in dropout_masks.items():
            module = dict(self._model.named_modules()).get(layer_name)
            
            if module is None or not hasattr(module, 'weight'):
                logger.warning(f"Cannot apply dropout to layer '{layer_name}'")
                continue
            
            if mode == "multiplicative":
                # Apply mask during forward pass
                if not permanent:
                    # Store original weight and apply mask temporarily
                    if not hasattr(self, '_original_weights'):
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
        if hasattr(self, '_original_weights'):
            for layer_name, original_weight in self._original_weights.items():
                module = dict(self._model.named_modules()).get(layer_name)
                if module is not None:
                    module.weight.data = original_weight
            delattr(self, '_original_weights') 