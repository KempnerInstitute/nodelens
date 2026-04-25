"""
Data module for the alignment metrics framework.

This module provides dataset wrappers and data loading utilities
for use with alignment analysis experiments.
"""

from nodelens.dataops.base import BaseDataset, DatasetWrapper
from nodelens.dataops.datasets import get_dataset
from nodelens.dataops.loaders import DataLoaderConfig, create_data_loader, create_distributed_loader

# Import dataset implementations when they're created
try:
    from nodelens.dataops.datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
    from nodelens.dataops.datasets.imagenet import ImageNetDataset
    from nodelens.dataops.datasets.mnist import MNISTDataset
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
