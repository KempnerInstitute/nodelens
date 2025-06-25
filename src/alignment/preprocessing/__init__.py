"""
Preprocessing module for handling different layer types and data formats.

This module provides centralized preprocessing functionality for various
neural network layer types, ensuring consistent data formatting across
the alignment framework.
"""

from .layer_preprocessing import (
    LayerPreprocessor,
    CNNPreprocessor,
    preprocess_layer_activations,
    get_preprocessor,
)

__all__ = [
    'LayerPreprocessor',
    'CNNPreprocessor',
    'preprocess_layer_activations',
    'get_preprocessor',
] 