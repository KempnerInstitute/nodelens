"""
Quantization module for model compression.

Provides quantization strategies for neural networks, particularly LLMs:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Mixed-precision quantization
- Integration with pruning for combined compression

Supports:
- INT8, INT4, FP16, BF16
- Per-channel and per-tensor quantization
- Symmetric and asymmetric quantization
"""

from .analysis import (
    analyze_quantization_sensitivity,
    compute_quantization_error,
    find_optimal_bit_allocation,
)
from .ptq import (
    INT4Quantizer,
    INT8Quantizer,
    MixedPrecisionQuantizer,
    quantize_layer,
    quantize_model,
)

__all__ = [
    # PTQ
    "quantize_model",
    "quantize_layer",
    "INT8Quantizer",
    "INT4Quantizer",
    "MixedPrecisionQuantizer",
    # Analysis
    "analyze_quantization_sensitivity",
    "compute_quantization_error",
    "find_optimal_bit_allocation",
]
