"""
Quantization analysis utilities.

Analyze quantization effects on model performance and neuron importance.
"""

import logging
from typing import Callable, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def analyze_quantization_sensitivity(
    model: nn.Module,
    data_loader,
    eval_fn: Callable,
    precision_levels: List[str] = ['int8', 'int4']
) -> Dict[str, Dict]:
    """
    Analyze how sensitive each layer is to quantization.

    Args:
        model: Model to analyze
        data_loader: Validation data
        eval_fn: Evaluation function
        precision_levels: Precisions to test

    Returns:
        Sensitivity scores per layer per precision
    """
    from .ptq import quantize_layer

    baseline_score = eval_fn(model, data_loader)

    results = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        layer_results = {}

        for precision in precision_levels:
            # Save original weight
            original_weight = module.weight.data.clone()

            # Quantize and dequantize
            quant_result = quantize_layer(module, precision)

            # Dequantize for evaluation
            if precision == 'int8':
                dequant = quant_result['weight'].float() * quant_result['scale']
                if not quant_result.get('symmetric', True):
                    dequant += quant_result['zero_point'].float() * quant_result['scale']
            else:  # int4
                # Simplified dequantization
                dequant = quant_result['weight'].float() * quant_result['scale'].mean()

            # Apply quantized weight
            module.weight.data = dequant

            # Evaluate
            quantized_score = eval_fn(model, data_loader)

            # Compute sensitivity
            sensitivity = abs(baseline_score - quantized_score)

            layer_results[precision] = {
                'score': quantized_score,
                'sensitivity': sensitivity,
                'relative_error': sensitivity / baseline_score if baseline_score != 0 else 0
            }

            # Restore original
            module.weight.data = original_weight

        results[name] = layer_results

    return results


def compute_quantization_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor
) -> Dict[str, float]:
    """
    Compute quantization error metrics.

    Args:
        original_weight: Original FP weights
        quantized_weight: Quantized weights
        scale: Quantization scale

    Returns:
        Error metrics (MSE, MAE, SNR)
    """
    # Dequantize
    dequantized = quantized_weight.float() * scale

    # Compute errors
    mse = ((original_weight - dequantized) ** 2).mean().item()
    mae = (original_weight - dequantized).abs().mean().item()

    # Signal-to-noise ratio
    signal_power = (original_weight ** 2).mean().item()
    noise_power = mse
    snr = 10 * torch.log10(torch.tensor(signal_power / (noise_power + 1e-10)))

    return {
        'mse': mse,
        'mae': mae,
        'snr': snr.item()
    }


def find_optimal_bit_allocation(
    model: nn.Module,
    layer_scores: Dict[str, torch.Tensor],
    target_avg_bits: float = 6.0,
    min_bits: int = 4,
    max_bits: int = 8
) -> Dict[str, int]:
    """
    Find optimal bit allocation per layer based on importance.

    High importance layers get more bits, low importance get fewer bits,
    achieving target average bits overall.

    Args:
        model: Model
        layer_scores: Importance scores per layer
        target_avg_bits: Target average bits (e.g., 6.0)
        min_bits: Minimum bits per layer
        max_bits: Maximum bits per layer

    Returns:
        Bit allocation per layer
    """
    # Normalize scores to [0, 1]
    all_scores = []
    layer_names = []
    layer_sizes = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in layer_scores:
            score = layer_scores[name].mean().item()
            size = module.weight.numel()

            all_scores.append(score)
            layer_names.append(name)
            layer_sizes.append(size)

    if not all_scores:
        return {}

    scores_tensor = torch.tensor(all_scores)
    scores_norm = (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min() + 1e-8)

    # Allocate bits proportionally to importance
    # Important layers get more bits
    bit_range = max_bits - min_bits
    bits_initial = (min_bits + bit_range * scores_norm).round().int()

    # Adjust to meet target average
    total_size = sum(layer_sizes)
    current_avg = sum(bits_initial[i].item() * layer_sizes[i] for i in range(len(layer_sizes))) / total_size

    # Scale to hit target
    if current_avg != target_avg_bits:
        adjustment = target_avg_bits / current_avg
        bits_adjusted = (bits_initial.float() * adjustment).round().clamp(min_bits, max_bits).int()
    else:
        bits_adjusted = bits_initial

    # Create allocation dict
    allocation = {
        layer_names[i]: bits_adjusted[i].item()
        for i in range(len(layer_names))
    }

    # Verify average
    actual_avg = sum(allocation[name] * layer_sizes[i] for i, name in enumerate(layer_names)) / total_size
    logger.info(f"Bit allocation: target={target_avg_bits}, actual={actual_avg:.2f}")

    return allocation

