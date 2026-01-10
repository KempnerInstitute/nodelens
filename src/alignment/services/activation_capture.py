"""
Activation Capture Service for centralized activation and weight collection.

This service provides a clean API for capturing activations with automatic
lifecycle management and preprocessing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

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

    def __init__(self, model_wrapper: Any, default_mode: str = "flatten", conv_spatial: str = "patchwise"):  # BaseModelWrapper or similar
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
        preprocess: bool = True,
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
            # Prefer capturing RAW activations (we do preprocessing here for consistency)
            # Prefer passing explicit layers where supported
            output, activations = self.model_wrapper.forward_with_activations(
                input_batch, layers=layers, preprocess=False
            )
        except TypeError as e:
            # Backwards compatibility: wrappers may not accept 'layers' and/or 'preprocess'
            msg = str(e)
            try:
                if "unexpected keyword argument 'preprocess'" in msg:
                    # Retry without preprocess kwarg
                    output, activations = self.model_wrapper.forward_with_activations(input_batch, layers=layers)
                elif "unexpected keyword argument 'layers'" in msg:
                    logger.warning(
                        "Model wrapper.forward_with_activations does not accept 'layers'; "
                        "capturing activations for all tracked layers instead."
                    )
                    # Retry without layers (and without preprocess, which may also be unsupported)
                    try:
                        output, activations = self.model_wrapper.forward_with_activations(input_batch, preprocess=False)
                    except TypeError:
                        output, activations = self.model_wrapper.forward_with_activations(input_batch)
                else:
                    raise
            except Exception:
                logger.error(f"Failed to capture activations: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to capture activations: {e}")
            raise

        # Preprocess if requested (unified, layer-aware preprocessing)
        processed_activations = activations
        if preprocess and mode not in {"none", "preserve_spatial"}:
            if mode in {"unfold", "patchwise", "batch_patch_combined"}:
                try:
                    # Use the canonical preprocessing utilities that distinguish _input vs _output
                    from alignment.dataops.processing import preprocess_layer_activations

                    model = getattr(self.model_wrapper, "model", None) or getattr(self.model_wrapper, "_model", None)
                    if model is None:
                        raise AttributeError("Model wrapper has no .model or ._model attribute")

                    layer_modules = dict(model.named_modules())

                    # Only preprocess the activations we will actually consume
                    to_process = {}
                    for layer in layers:
                        in_key = f"{layer}_input"
                        out_key = f"{layer}_output"
                        if in_key in activations:
                            to_process[in_key] = activations[in_key]
                        if out_key in activations:
                            to_process[out_key] = activations[out_key]
                        # Legacy compatibility: some wrappers store outputs under the bare layer name
                        if out_key not in activations and layer in activations:
                            to_process[layer] = activations[layer]

                    processed_activations = preprocess_layer_activations(to_process, layer_modules, mode=mode)
                except Exception as e:
                    logger.warning(f"Layer-aware preprocessing failed ({e}); falling back to simple flatten")
                    processed_activations = activations
            elif mode == "flatten":
                # Simple flattening: [B, ...] -> [B, -1]
                processed_activations = {}
                for name, tensor in activations.items():
                    if isinstance(tensor, torch.Tensor) and tensor.ndim > 2:
                        processed_activations[name] = tensor.reshape(tensor.shape[0], -1)
                    else:
                        processed_activations[name] = tensor
            else:
                logger.warning(f"Unknown preprocessing mode '{mode}', using 'flatten'")
                processed_activations = {}
                for name, tensor in activations.items():
                    if isinstance(tensor, torch.Tensor) and tensor.ndim > 2:
                        processed_activations[name] = tensor.reshape(tensor.shape[0], -1)
                    else:
                        processed_activations[name] = tensor

        # Separate inputs and outputs (keys normalized to layer names)
        inputs: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        for layer in layers:
            input_key = f"{layer}_input"
            output_key = f"{layer}_output"

            if input_key in processed_activations:
                inputs[layer] = processed_activations[input_key]
            if output_key in processed_activations:
                outputs[layer] = processed_activations[output_key]
            elif layer in processed_activations:
                outputs[layer] = processed_activations[layer]

        # Capture weights if requested
        weights = {}
        if include_weights:
            try:
                weights = self.model_wrapper.get_layer_weights(layers=layers)
            except TypeError:
                # Older wrappers may not support 'layers' kwarg
                weights = self.model_wrapper.get_layer_weights()

        return ActivationData(inputs=inputs, outputs=outputs, weights=weights, layer_names=layers)

    def capture_batch_aggregated(
        self, dataloader, layers: Optional[List[str]] = None, max_batches: Optional[int] = None, aggregation: str = "concatenate"
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
            data = self.capture(inputs, layers=layers, include_weights=(batch_idx == 0))  # Only get weights once

            # Accumulate
            for layer in layers:
                if layer in data.inputs:
                    all_inputs[layer].append(data.inputs[layer])
                if layer in data.outputs:
                    all_outputs[layer].append(data.outputs[layer])

        # Aggregate
        if aggregation == "concatenate":
            aggregated_inputs = {layer: torch.cat(tensors, dim=0) for layer, tensors in all_inputs.items() if tensors}
            aggregated_outputs = {layer: torch.cat(tensors, dim=0) for layer, tensors in all_outputs.items() if tensors}
        elif aggregation == "mean":
            aggregated_inputs = {layer: torch.stack(tensors, dim=0).mean(dim=0) for layer, tensors in all_inputs.items() if tensors}
            aggregated_outputs = {layer: torch.stack(tensors, dim=0).mean(dim=0) for layer, tensors in all_outputs.items() if tensors}
        else:  # 'none'
            aggregated_inputs = all_inputs
            aggregated_outputs = all_outputs

        # Get weights from first batch
        weights = data.weights if "data" in locals() else {}

        return ActivationData(inputs=aggregated_inputs, outputs=aggregated_outputs, weights=weights, layer_names=layers)

    def _get_layer_module(self, layer_name: str) -> Optional[torch.nn.Module]:
        """
        Get the layer module by name, handling input/output suffixes.
        
        Args:
            layer_name: Layer name (possibly with _input or _output suffix)
            
        Returns:
            The layer module or None if not found
        """
        # Strip _input or _output suffix
        actual_name = layer_name
        if layer_name.endswith("_input"):
            actual_name = layer_name[:-6]
        elif layer_name.endswith("_output"):
            actual_name = layer_name[:-7]
        
        # Try to get from model's named_modules
        try:
            modules = dict(self.model_wrapper._model.named_modules())
            return modules.get(actual_name)
        except Exception:
            return None

    def _preprocess_activations(self, activations: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess activations based on mode.

        Args:
            activations: Raw activations
            mode: Preprocessing mode

        Returns:
            Preprocessed activations
        """
        if mode == "none":
            return activations

        processed = {}
        for name, tensor in activations.items():
            if mode == "flatten":
                # Flatten to [batch, features]
                if tensor.ndim > 2:
                    processed[name] = tensor.reshape(tensor.shape[0], -1)
                else:
                    processed[name] = tensor

            elif mode == "preserve_spatial":
                # Keep original shape
                processed[name] = tensor

            elif mode == "patchwise":
                # For Conv: [B, C, H, W] -> [B, C, H*W]
                if tensor.ndim == 4:  # Conv2d
                    B, C, H, W = tensor.shape
                    processed[name] = tensor.reshape(B, C, H * W)
                elif tensor.ndim == 3:  # Conv1d
                    processed[name] = tensor
                else:
                    processed[name] = tensor

            elif mode == "unfold":
                # For Conv: Use proper unfold with kernel parameters for RQ computation
                # This requires access to the layer module to get kernel size, stride, padding
                if tensor.ndim == 4:  # Conv2d
                    # Try to get the layer module for proper unfolding
                    layer_module = self._get_layer_module(name)
                    if layer_module is not None and isinstance(layer_module, torch.nn.Conv2d):
                        # Proper unfold using layer's kernel parameters
                        try:
                            unfolded = torch.nn.functional.unfold(
                                tensor,
                                kernel_size=layer_module.kernel_size,
                                dilation=layer_module.dilation,
                                padding=layer_module.padding,
                                stride=layer_module.stride,
                            )
                            # [B, C*kH*kW, num_patches] -> [B*num_patches, C*kH*kW]
                            B = tensor.shape[0]
                            unfolded = unfolded.transpose(1, 2).contiguous()
                            processed[name] = unfolded.view(-1, unfolded.size(2))
                        except Exception as e:
                            logger.debug(f"Proper unfold failed for {name}: {e}, using simple unfold")
                            B, C, H, W = tensor.shape
                            processed[name] = tensor.permute(0, 2, 3, 1).reshape(-1, C)
                    else:
                        # Fallback: simple unfold [B, C, H, W] -> [B*H*W, C]
                        B, C, H, W = tensor.shape
                        processed[name] = tensor.permute(0, 2, 3, 1).reshape(-1, C)
                elif tensor.ndim == 3:  # Conv1d: [B, C, L]
                    B, C, L = tensor.shape
                    processed[name] = tensor.permute(0, 2, 1).reshape(-1, C)
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
        if hasattr(self.model_wrapper, "hook_manager"):
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
