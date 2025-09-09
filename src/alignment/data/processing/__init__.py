"""
Data processing module for alignment framework.

This module provides:
1. Batch processing utilities for efficient metric computation on large datasets
2. Layer-specific preprocessing for different neural network architectures
"""

from .batch import BatchMetricProcessor
from .layers import (
    LayerPreprocessor,
    LinearPreprocessor,
    CNNPreprocessor,
    AttentionPreprocessor,
    preprocess_layer_activations,
    get_preprocessor,
)

__all__ = [
    "BatchMetricProcessor",
    "LayerPreprocessor",
    "LinearPreprocessor", 
    "CNNPreprocessor",
    "AttentionPreprocessor",
    "preprocess_layer_activations",
    "get_preprocessor",
] 