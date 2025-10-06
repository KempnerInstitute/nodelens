"""
Concrete model wrapper implementations.

This module provides the main model wrappers used for alignment analysis,
including the general-purpose ModelWrapper and specialized wrappers.
"""

import copy
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.registry import register_model
from .base import BaseModelWrapper

logger = logging.getLogger(__name__)


@register_model("model_wrapper")
class ModelWrapper(BaseModelWrapper):
    """
    General-purpose model wrapper for alignment analysis.
    
    This wrapper can handle any PyTorch model and provides all the
    functionality needed for computing alignment metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tracked_layers: Optional[List[str]] = None,
        preprocessing_mode: str = "auto",
        store_initial_weights: bool = False,
        **config: Any
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model: PyTorch model to wrap
            tracked_layers: Layers to track (None = auto-discover)
            preprocessing_mode: How to preprocess activations ("auto", "flatten", "unfold", "patchwise")
            store_initial_weights: Whether to store a copy of initial weights
            **config: Additional configuration
        """
        super().__init__(model, tracked_layers, **config)
        self.preprocessing_mode = preprocessing_mode
        
        # Store initial weights if requested (for delta alignment)
        if store_initial_weights:
            self._initial_weights = {}
            with torch.no_grad():
                for name, module in self._model.named_modules():
                    if name in self._tracked_layers and hasattr(module, 'weight'):
                        self._initial_weights[name] = module.weight.clone()
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the wrapped model."""
        return self._model(*args, **kwargs)
    
    def forward_with_activations(
        self,
        inputs: Any,
        preprocess: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns intermediate activations.
        
        Args:
            inputs: Input tensor
            preprocess: Whether to preprocess activations
            
        Returns:
            Tuple of (model outputs, activation dictionary)
        """
        self._activation_cache.clear()
        # Support both tensor and dict inputs (for HF models)
        if isinstance(inputs, dict):
            outputs = self.forward(**inputs)
        else:
            outputs = self.forward(inputs)
        
        # Get activations
        activations = self._activation_cache.copy()
        
        # Preprocess if requested
        if preprocess:
            mode = self._determine_preprocessing_mode(activations)
            activations = self.preprocess_activations(activations, mode)
        
        return outputs, activations
    
    def _determine_preprocessing_mode(self, activations: Dict[str, torch.Tensor]) -> str:
        """Automatically determine the best preprocessing mode."""
        if self.preprocessing_mode != "auto":
            return self.preprocessing_mode
        
        # Check if we have conv layers
        has_conv = False
        for name, act in activations.items():
            if act.ndim == 4:  # Conv output
                has_conv = True
                break
        
        # Default to unfold for conv layers, flatten for others
        return "unfold" if has_conv else "flatten"
    
    def get_initial_weights(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get stored initial weights for a layer."""
        if hasattr(self, '_initial_weights'):
            return self._initial_weights.get(layer_name)
        return None
    
    def compute_forward_with_dropout(
        self,
        inputs: torch.Tensor,
        dropout_indices: Dict[str, torch.Tensor],
        mode: str = "structured"
    ) -> torch.Tensor:
        """
        Forward pass with targeted dropout.
        
        Args:
            inputs: Input tensor
            dropout_indices: Indices to drop for each layer
            mode: Dropout mode ("structured", "random")
            
        Returns:
            Model outputs with dropout applied
        """
        # Create dropout masks from indices
        dropout_masks = {}
        for layer_name, indices in dropout_indices.items():
            module = dict(self._model.named_modules()).get(layer_name)
            if module is not None and hasattr(module, 'weight'):
                num_units = module.weight.shape[0]
                mask = torch.ones(num_units, device=module.weight.device)
                mask[indices] = 0
                dropout_masks[layer_name] = mask
        
        # Apply dropout
        self.apply_structured_dropout(dropout_masks, mode="multiplicative", permanent=False)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Restore weights
        self.restore_weights()
        
        return outputs


@register_model("alignment_network")
class AlignmentNetwork(ModelWrapper):
    """
    Specialized wrapper that mimics the original AlignmentNetwork functionality.
    
    This provides backward compatibility with the existing codebase while
    using the new architecture.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alignment_layers: Optional[Union[List[str], Dict[str, Any]]] = None,
        cnn_mode: str = "unfold",
        store_initial_weights: bool = True,
        **config: Any
    ):
        """
        Initialize AlignmentNetwork wrapper.
        
        Args:
            model: Base model to wrap
            alignment_layers: Layers to track for alignment
            cnn_mode: CNN processing mode
            store_initial_weights: Whether to store initial weights
            **config: Additional configuration
        """
        # Convert alignment_layers format if needed
        if isinstance(alignment_layers, dict):
            tracked_layers = list(alignment_layers.keys())
        else:
            tracked_layers = alignment_layers
        
        # Map cnn_mode to preprocessing_mode
        preprocessing_map = {
            "unfold": "unfold",
            "patchwise": "patchwise",
            "batch_patch_combined": "patchwise",
            "filter_patch_summary": "patchwise",
        }
        preprocessing_mode = preprocessing_map.get(cnn_mode, "auto")
        
        super().__init__(
            model=model,
            tracked_layers=tracked_layers,
            preprocessing_mode=preprocessing_mode,
            store_initial_weights=store_initial_weights,
            **config
        )
        
        self.cnn_mode = cnn_mode
        self.alignment_layers = alignment_layers
    
    def get_alignment_weights(self, flatten: bool = True) -> List[torch.Tensor]:
        """Get weights in the format expected by the original AlignmentNetwork."""
        weights_dict = self.get_layer_weights(flatten=flatten)
        
        # Return in the order of tracked_layers
        weights_list = []
        for layer_name in self._tracked_layers:
            if layer_name in weights_dict:
                weights_list.append(weights_dict[layer_name])
        
        return weights_list
    
    def get_layer_inputs(
        self,
        images: torch.Tensor,
        precomputed: bool = True
    ) -> List[torch.Tensor]:
        """Get layer inputs in the format expected by the original code."""
        _, activations = self.forward_with_activations(images, preprocess=False)
        
        # Extract inputs in order
        inputs_list = []
        for layer_name in self._tracked_layers:
            input_key = f"{layer_name}_input"
            if input_key in activations:
                inputs_list.append(activations[input_key])
        
        return inputs_list
    
    def _preprocess_inputs(
        self,
        raw_inputs: List[torch.Tensor],
        compress_convolutional: bool = True
    ) -> List[torch.Tensor]:
        """Preprocess inputs in the format expected by the original code."""
        processed = []
        
        for i, inp in enumerate(raw_inputs):
            if inp.ndim == 4 and compress_convolutional:
                if self.cnn_mode == "unfold":
                    # Unfold conv inputs
                    layer_name = self._tracked_layers[i]
                    module = dict(self._model.named_modules()).get(layer_name)
                    
                    if isinstance(module, nn.Conv2d):
                        unfolded = F.unfold(
                            inp,
                            kernel_size=module.kernel_size,
                            stride=module.stride,
                            padding=module.padding
                        )
                        # Reshape to [batch, features, patches]
                        processed.append(unfolded.permute(0, 2, 1))
                    else:
                        processed.append(inp.flatten(start_dim=1))
                        
                elif self.cnn_mode == "patchwise":
                    # Keep patch structure
                    b, c, h, w = inp.shape
                    processed.append(inp.reshape(b, c, h * w))
                    
                else:
                    # Flatten
                    processed.append(inp.flatten(start_dim=1))
            else:
                # Already 2D or not compressing
                processed.append(inp.flatten(start_dim=1) if inp.ndim > 2 else inp)
        
        return processed
    
    def forward_targeted_dropout(
        self,
        images: torch.Tensor,
        dropout_indices: List[torch.Tensor],
        target_layers: List[int]
    ) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass with targeted dropout (compatibility method).
        
        Args:
            images: Input images
            dropout_indices: List of indices to drop for each target layer
            target_layers: Indices of layers to apply dropout to
            
        Returns:
            Tuple of (outputs, None)
        """
        # Convert to new format
        dropout_dict = {}
        for layer_idx, indices in zip(target_layers, dropout_indices):
            if 0 <= layer_idx < len(self._tracked_layers):
                layer_name = self._tracked_layers[layer_idx]
                dropout_dict[layer_name] = indices
        
        outputs = self.compute_forward_with_dropout(images, dropout_dict)
        return outputs, None


class ActivationTracker:
    """
    Utility class for tracking activations across multiple forward passes.
    
    Useful for accumulating statistics over a dataset.
    """
    
    def __init__(self, model_wrapper: BaseModelWrapper):
        """
        Initialize the activation tracker.
        
        Args:
            model_wrapper: Model wrapper to track activations for
        """
        self.model_wrapper = model_wrapper
        self.accumulated_activations = {}
        self.activation_counts = {}
    
    def update(self, inputs: torch.Tensor) -> None:
        """Update accumulated activations with a new batch."""
        _, activations = self.model_wrapper.forward_with_activations(inputs)
        
        for layer_name, activation in activations.items():
            if layer_name not in self.accumulated_activations:
                self.accumulated_activations[layer_name] = []
                self.activation_counts[layer_name] = 0
            
            self.accumulated_activations[layer_name].append(activation.detach().cpu())
            self.activation_counts[layer_name] += activation.shape[0]
    
    def get_statistics(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute statistics over accumulated activations."""
        stats = {}
        
        for layer_name, act_list in self.accumulated_activations.items():
            if not act_list:
                continue
            
            # Concatenate all activations
            all_acts = torch.cat(act_list, dim=0)
            
            # Compute statistics
            stats[layer_name] = {
                "mean": all_acts.mean(dim=0),
                "std": all_acts.std(dim=0),
                "min": all_acts.min(dim=0)[0],
                "max": all_acts.max(dim=0)[0],
                "shape": list(all_acts.shape),
                "num_samples": self.activation_counts[layer_name]
            }
        
        return stats
    
    def clear(self) -> None:
        """Clear accumulated activations."""
        self.accumulated_activations.clear()
        self.activation_counts.clear() 