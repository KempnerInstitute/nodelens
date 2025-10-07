"""
Dataset implementations for alignment analysis.

This module provides a unified dataset interface that can handle
various dataset types without code duplication.
"""

from typing import Optional, Tuple

import torch.utils.data

# Import for backward compatibility - these are now created dynamically
# but we import them to make them available at module level
from alignment.core.registry import DATASET_REGISTRY
from alignment.data.datasets.unified_dataset import (DATASET_CONFIGS,
                                                     UnifiedDataset)

# Try to import text datasets (optional - requires additional dependencies)
try:
    from alignment.data.datasets.text_datasets import (C4Dataset, TextDataset,
                                                       WikiTextDataset,
                                                       load_text_dataset)

    HAS_TEXT_DATASETS = True
except ImportError:
    HAS_TEXT_DATASETS = False
    TextDataset = None
    WikiTextDataset = None
    C4Dataset = None
    load_text_dataset = None

# Get dynamically created dataset classes
MNISTDataset = DATASET_REGISTRY.get("mnist")
FashionMNISTDataset = DATASET_REGISTRY.get("fashion_mnist")
CIFAR10Dataset = DATASET_REGISTRY.get("cifar10")
CIFAR100Dataset = DATASET_REGISTRY.get("cifar100")
ImageNetDataset = DATASET_REGISTRY.get("imagenet")
SVHNDataset = DATASET_REGISTRY.get("svhn")


def get_dataset(
    dataset_name: str, batch_size: int = 128, num_workers: int = 4, **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Get train and validation data loaders for a dataset.

    Args:
        dataset_name: Name of the dataset (mnist, cifar10, etc.)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        **kwargs: Additional arguments for dataset

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get dataset class from registry
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    if dataset_class is None:
        available = list(DATASET_REGISTRY._registry.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    # Create train and val datasets
    train_dataset = dataset_class(train=True, **kwargs)
    val_dataset = dataset_class(train=False, **kwargs)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


__all__ = [
    "UnifiedDataset",
    "DATASET_CONFIGS",
    "get_dataset",
    "MNISTDataset",
    "FashionMNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ImageNetDataset",
    "SVHNDataset",
]

# Add text datasets to exports if available
if HAS_TEXT_DATASETS:
    __all__.extend(
        [
            "TextDataset",
            "WikiTextDataset",
            "C4Dataset",
            "load_text_dataset",
        ]
    )
