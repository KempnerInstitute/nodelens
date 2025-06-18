"""
Dataset implementations for alignment analysis.

This module provides a unified dataset interface that can handle
various dataset types without code duplication.
"""

from alignment.data.datasets.unified_dataset import (
    UnifiedDataset,
    DATASET_CONFIGS,
)

# Import for backward compatibility - these are now created dynamically
# but we import them to make them available at module level
from alignment.core.registry import DATASET_REGISTRY

# Get dynamically created dataset classes
MNISTDataset = DATASET_REGISTRY.get("mnist")
FashionMNISTDataset = DATASET_REGISTRY.get("fashion_mnist") 
CIFAR10Dataset = DATASET_REGISTRY.get("cifar10")
CIFAR100Dataset = DATASET_REGISTRY.get("cifar100")
ImageNetDataset = DATASET_REGISTRY.get("imagenet")
SVHNDataset = DATASET_REGISTRY.get("svhn")

__all__ = [
    'UnifiedDataset',
    'DATASET_CONFIGS',
    'MNISTDataset',
    'FashionMNISTDataset',
    'CIFAR10Dataset', 
    'CIFAR100Dataset',
    'ImageNetDataset',
    'SVHNDataset',
] 