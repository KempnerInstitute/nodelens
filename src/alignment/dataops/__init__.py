"""
Data module for the alignment metrics framework.

This module provides dataset wrappers and data loading utilities
for use with alignment analysis experiments.
"""

from alignment.dataops.base import BaseDataset, DatasetWrapper
from alignment.dataops.datasets import get_dataset
from alignment.dataops.loaders import (DataLoaderConfig, create_data_loader,
                                       create_distributed_loader)

# Import dataset implementations when they're created
try:
    from alignment.dataops.datasets.cifar import (CIFAR10Dataset,
                                                  CIFAR100Dataset)
    from alignment.dataops.datasets.imagenet import ImageNetDataset
    from alignment.dataops.datasets.mnist import MNISTDataset
except ImportError:
    pass  # Datasets will be implemented next

__all__ = [
    "BaseDataset",
    "DatasetWrapper",
    "create_data_loader",
    "create_distributed_loader",
    "DataLoaderConfig",
    "get_dataset",
    "MNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ImageNetDataset",
]
