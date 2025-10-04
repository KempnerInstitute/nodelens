"""
Post-Training Quantization (PTQ) implementations.

Quantize trained models without retraining.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Literal
import logging

logger = logging.getLogger(__name__)


class INT8Quantizer:
    """
    INT8 post-training quantization.
    
    Converts FP32/FP16 weights and activations to INT8.
    """
    
    def __init__(
        self,
        per_channel: bool = True,
        symmetric: bool = True
    ):
        """
        Initialize INT8 quantizer.
        
        Args:
            per_channel: Per-channel vs per-tensor quantization
            symmetric: Symmetric vs asymmetric quantization
        """
        self.per_channel = per_channel
        self.symmetric = symmetric
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        dim: int = 0
    ) -> tuple:
        """
        Quantize a tensor to INT8.
        
        Args:
            tensor: Tensor to quantize
            dim: Channel dimension (for per-channel)
            
        Returns:
            (quantized_tensor, scale, zero_point)
        """
        if self.per_channel:
            # Per-channel quantization
            # Compute scale per channel
            dims_to_reduce = list(range(tensor.ndim))
            dims_to_reduce.remove(dim)
            
            if self.symmetric:
                # Symmetric: scale based on max absolute value
                max_val = tensor.abs().amax(dim=dims_to_reduce, keepdim=True)
                scale = max_val / 127.0
                zero_point = torch.zeros_like(scale, dtype=torch.int8)
            else:
                # Asymmetric: scale based on min/max
                min_val = tensor.amin(dim=dims_to_reduce, keepdim=True)
                max_val = tensor.amax(dim=dims_to_reduce, keepdim=True)
                scale = (max_val - min_val) / 255.0
                zero_point = (-min_val / scale).round().to(torch.int8)
        else:
            # Per-tensor quantization
            if self.symmetric:
                max_val = tensor.abs().max()
                scale = max_val / 127.0
                zero_point = torch.tensor(0, dtype=torch.int8)
            else:
                min_val = tensor.min()
                max_val = tensor.max()
                scale = (max_val - min_val) / 255.0
                zero_point = (-min_val / scale).round().to(torch.int8)
        
        # Quantize
        if self.symmetric:
            quantized = (tensor / (scale + 1e-8)).round().clamp(-127, 127).to(torch.int8)
        else:
            quantized = ((tensor / (scale + 1e-8)) + zero_point).round().clamp(0, 255).to(torch.int8)
        
        return quantized, scale, zero_point
    
    def quantize_linear_layer(
        self,
        layer: nn.Linear
    ) -> Dict:
        """
        Quantize a Linear layer.
        
        Args:
            layer: Linear layer to quantize
            
        Returns:
            Dict with quantized weight, scale, zero_point
        """
        weight = layer.weight.data
        
        # Quantize weights (per output channel)
        q_weight, scale, zero_point = self.quantize_tensor(weight, dim=0)
        
        result = {
            'weight': q_weight,
            'scale': scale,
            'zero_point': zero_point,
            'original_dtype': weight.dtype,
            'original_shape': weight.shape
        }
        
        if layer.bias is not None:
            result['bias'] = layer.bias.data
        
        return result


class INT4Quantizer:
    """
    INT4 quantization for aggressive compression.
    
    Commonly used for LLMs to reduce memory footprint.
    """
    
    def __init__(self, block_size: int = 128):
        """
        Initialize INT4 quantizer.
        
        Args:
            block_size: Block size for group quantization
        """
        self.block_size = block_size
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor
    ) -> tuple:
        """
        Quantize tensor to INT4 using block-wise quantization.
        
        Args:
            tensor: Tensor to quantize
            
        Returns:
            (quantized, scales, zero_points)
        """
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        # Pad to block size
        pad_size = (self.block_size - len(tensor_flat) % self.block_size) % self.block_size
        if pad_size > 0:
            tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_size))
        
        # Reshape into blocks
        blocks = tensor_flat.reshape(-1, self.block_size)
        num_blocks = blocks.shape[0]
        
        # Quantize each block
        quantized_blocks = []
        scales = []
        zero_points = []
        
        for block in blocks:
            # Symmetric INT4: range [-7, 7]
            max_val = block.abs().max()
            scale = max_val / 7.0
            
            q_block = (block / (scale + 1e-8)).round().clamp(-7, 7).to(torch.int8)
            
            quantized_blocks.append(q_block)
            scales.append(scale)
            zero_points.append(torch.tensor(0, dtype=torch.int8))
        
        quantized = torch.cat(quantized_blocks)
        
        # Remove padding
        if pad_size > 0:
            quantized = quantized[:-pad_size]
        
        # Reshape back
        quantized = quantized.reshape(original_shape)
        scales = torch.tensor(scales, dtype=tensor.dtype)
        
        return quantized, scales, torch.zeros_like(scales, dtype=torch.int8)


class MixedPrecisionQuantizer:
    """
    Mixed-precision quantization.
    
    Uses alignment metrics to decide precision per layer:
    - Important layers: Higher precision (FP16/INT8)
    - Less important: Lower precision (INT4)
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.5,
        high_precision: str = 'int8',
        low_precision: str = 'int4'
    ):
        """
        Initialize mixed-precision quantizer.
        
        Args:
            importance_threshold: Threshold for high vs low precision
            high_precision: Format for important layers
            low_precision: Format for less important layers
        """
        self.importance_threshold = importance_threshold
        self.high_precision = high_precision
        self.low_precision = low_precision
        
        self.int8_quantizer = INT8Quantizer()
        self.int4_quantizer = INT4Quantizer()
    
    def quantize_model_adaptive(
        self,
        model: nn.Module,
        layer_importance: Dict[str, float]
    ) -> Dict:
        """
        Quantize model with mixed precision based on importance.
        
        Args:
            model: Model to quantize
            layer_importance: Importance score per layer
            
        Returns:
            Quantization results per layer
        """
        results = {}
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            if name not in layer_importance:
                continue
            
            importance = layer_importance[name]
            
            # Choose precision based on importance
            if importance > self.importance_threshold:
                # High importance -> higher precision
                precision = self.high_precision
                result = self.int8_quantizer.quantize_linear_layer(module)
            else:
                # Low importance -> lower precision (more compression)
                precision = self.low_precision
                result = self.int4_quantizer.quantize_tensor(module.weight.data)
                result = {'weight': result[0], 'scale': result[1], 'zero_point': result[2]}
            
            result['precision'] = precision
            result['importance'] = importance
            results[name] = result
        
        return results


def quantize_model(
    model: nn.Module,
    precision: Literal['int8', 'int4', 'mixed'] = 'int8',
    layer_importance: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Quantize entire model.
    
    Args:
        model: Model to quantize
        precision: Quantization precision
        layer_importance: For mixed-precision (optional)
        
    Returns:
        Quantization results per layer
    """
    if precision == 'int8':
        quantizer = INT8Quantizer()
    elif precision == 'int4':
        quantizer = INT4Quantizer()
    elif precision == 'mixed':
        if layer_importance is None:
            raise ValueError("Mixed precision requires layer_importance")
        quantizer = MixedPrecisionQuantizer()
        return quantizer.quantize_model_adaptive(model, layer_importance)
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            results[name] = quantizer.quantize_linear_layer(module)
    
    return results


def quantize_layer(
    layer: nn.Module,
    precision: str = 'int8'
) -> Dict:
    """
    Quantize a single layer.
    
    Args:
        layer: Layer to quantize
        precision: Quantization precision
        
    Returns:
        Quantization result
    """
    if precision == 'int8':
        quantizer = INT8Quantizer()
    elif precision == 'int4':
        quantizer = INT4Quantizer()
    else:
        raise ValueError(f"Unknown precision: {precision}")
    
    if isinstance(layer, nn.Linear):
        return quantizer.quantize_linear_layer(layer)
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")

