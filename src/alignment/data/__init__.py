"""
Data module for the alignment metrics framework.

This module provides dataset wrappers and data loading utilities
for use with alignment analysis experiments.
"""

from alignment.data.base import BaseDataset, DatasetWrapper
from alignment.data.loaders import (
    create_data_loader,
    create_distributed_loader,
    DataLoaderConfig,
)
from alignment.data.datasets import get_dataset

# Import dataset implementations when they're created
try:
    from alignment.data.datasets.mnist import MNISTDataset
    from alignment.data.datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
    from alignment.data.datasets.imagenet import ImageNetDataset
except ImportError:
    pass  # Datasets will be implemented next

__all__ = [
    'BaseDataset',
    'DatasetWrapper',
    'create_data_loader',
    'create_distributed_loader',
    'DataLoaderConfig',
    'get_dataset',
    'MNISTDataset',
    'CIFAR10Dataset',
    'CIFAR100Dataset',
    'ImageNetDataset',
] 