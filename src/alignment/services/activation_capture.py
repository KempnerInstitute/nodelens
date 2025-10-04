"""
Activation Capture Service for centralized activation and weight collection.

This service provides a clean API for capturing activations with automatic
lifecycle management and preprocessing.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for captured activations and weights."""
    inputs: Dict[str, torch.Tensor]
    outputs: Dict[str, torch.Tensor]
    weights: Dict[str, torch.Tensor]
    layer_names: List[str]


class ActivationCaptureService:
    """
    Centralized service for activation and weight collection.
    
    This service provides:
    - Automatic hook lifecycle management
    - Preprocessing options
    - Memory-efficient batch processing
    - Clean API for experiments
    
    Example:
        >>> service = ActivationCaptureService(model_wrapper)
        >>> data = service.capture(input_batch, layers=['conv1', 'fc1'])
        >>> print(data.inputs['conv1'].shape)
        >>> print(data.outputs['conv1'].shape)
    """
    
    def __init__(
        self,
        model_wrapper: Any,  # BaseModelWrapper or similar
        default_mode: str = 'flatten',
        conv_spatial: str = 'patchwise'
    ):
        """
        Initialize activation capture service.
        
        Args:
            model_wrapper: Model wrapper with HookManager support
            default_mode: Default preprocessing mode ('flatten', 'preserve_spatial')
            conv_spatial: How to handle Conv spatial dims ('patchwise', 'global', 'unfold')
        """
        self.model_wrapper = model_wrapper
        self.default_mode = default_mode
        self.conv_spatial = conv_spatial
    
    def capture(
        self,
        input_batch: torch.Tensor,
        layers: Optional[List[str]] = None,
        mode: Optional[str] = None,
        include_weights: bool = True,
        preprocess: bool = True
    ) -> ActivationData:
        """
        Capture activations and weights for specified layers.
        
        Args:
            input_batch: Input tensor [batch, ...]
            layers: Layers to capture (None = all tracked)
            mode: Preprocessing mode (None = use default)
            include_weights: Whether to capture weights
            preprocess: Whether to preprocess activations
            
        Returns:
            ActivationData with inputs, outputs, and weights
        """
        mode = mode or self.default_mode
        layers = layers or self.model_wrapper.tracked_layers
        
        # Capture activations using model wrapper
        try:
            output, activations = self.model_wrapper.forward_with_activations(
                input_batch, layers=layers
            )
        except Exception as e:
            logger.error(f"Failed to capture activations: {e}")
            raise
        
        # Separate inputs and outputs
        inputs = {}
        outputs = {}
        for layer in layers:
            input_key = f"{layer}_input"
            output_key = f"{layer}_output"
            
            if input_key in activations:
                inputs[layer] = activations[input_key]
            if output_key in activations:
                outputs[layer] = activations[output_key]
        
        # Preprocess if requested
        if preprocess:
            inputs = self._preprocess_activations(inputs, mode)
            outputs = self._preprocess_activations(outputs, mode)
        
        # Capture weights if requested
        weights = {}
        if include_weights:
            weights = self.model_wrapper.get_layer_weights(layers=layers)
        
        return ActivationData(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            layer_names=layers
        )
    
    def capture_batch_aggregated(
        self,
        dataloader,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        aggregation: str = 'concatenate'
    ) -> ActivationData:
        """
        Capture activations across multiple batches with aggregation.
        
        Args:
            dataloader: DataLoader providing batches
            layers: Layers to capture
            max_batches: Maximum number of batches to process
            aggregation: How to aggregate ('concatenate', 'mean', 'none')
            
        Returns:
            Aggregated ActivationData
        """
        layers = layers or self.model_wrapper.tracked_layers
        
        all_inputs = {layer: [] for layer in layers}
        all_outputs = {layer: [] for layer in layers}
        
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Move to model device
            inputs = inputs.to(next(self.model_wrapper.model.parameters()).device)
            
            # Capture
            data = self.capture(
                inputs, 
                layers=layers, 
                include_weights=(batch_idx == 0)  # Only get weights once
            )
            
            # Accumulate
            for layer in layers:
                if layer in data.inputs:
                    all_inputs[layer].append(data.inputs[layer])
                if layer in data.outputs:
                    all_outputs[layer].append(data.outputs[layer])
        
        # Aggregate
        if aggregation == 'concatenate':
            aggregated_inputs = {
                layer: torch.cat(tensors, dim=0) 
                for layer, tensors in all_inputs.items() if tensors
            }
            aggregated_outputs = {
                layer: torch.cat(tensors, dim=0)
                for layer, tensors in all_outputs.items() if tensors
            }
        elif aggregation == 'mean':
            aggregated_inputs = {
                layer: torch.stack(tensors, dim=0).mean(dim=0)
                for layer, tensors in all_inputs.items() if tensors
            }
            aggregated_outputs = {
                layer: torch.stack(tensors, dim=0).mean(dim=0)
                for layer, tensors in all_outputs.items() if tensors
            }
        else:  # 'none'
            aggregated_inputs = all_inputs
            aggregated_outputs = all_outputs
        
        # Get weights from first batch
        weights = data.weights if 'data' in locals() else {}
        
        return ActivationData(
            inputs=aggregated_inputs,
            outputs=aggregated_outputs,
            weights=weights,
            layer_names=layers
        )
    
    def _preprocess_activations(
        self,
        activations: Dict[str, torch.Tensor],
        mode: str
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess activations based on mode.
        
        Args:
            activations: Raw activations
            mode: Preprocessing mode
            
        Returns:
            Preprocessed activations
        """
        if mode == 'none':
            return activations
        
        processed = {}
        for name, tensor in activations.items():
            if mode == 'flatten':
                # Flatten to [batch, features]
                if tensor.ndim > 2:
                    processed[name] = tensor.reshape(tensor.shape[0], -1)
                else:
                    processed[name] = tensor
            
            elif mode == 'preserve_spatial':
                # Keep original shape
                processed[name] = tensor
            
            elif mode == 'patchwise':
                # For Conv: [B, C, H, W] -> [B, C, H*W]
                if tensor.ndim == 4:  # Conv2d
                    B, C, H, W = tensor.shape
                    processed[name] = tensor.reshape(B, C, H * W)
                elif tensor.ndim == 3:  # Conv1d
                    processed[name] = tensor
                else:
                    processed[name] = tensor
            
            else:
                logger.warning(f"Unknown preprocessing mode: {mode}, using 'flatten'")
                processed[name] = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 2 else tensor
        
        return processed
    
    def __enter__(self):
        """Support context manager usage."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if hasattr(self.model_wrapper, 'hook_manager'):
            self.model_wrapper.hook_manager.cleanup()
        return False


def create_capture_service(model_wrapper, **config) -> ActivationCaptureService:
    """
    Factory function for creating ActivationCaptureService.
    
    Args:
        model_wrapper: Model wrapper instance
        **config: Configuration options
        
    Returns:
        Configured ActivationCaptureService
    """
    return ActivationCaptureService(model_wrapper, **config)

