"""
Dataset implementations for the alignment framework.

This module provides wrappers for common datasets used in
alignment analysis experiments.
"""

from alignment.data.datasets.mnist import MNISTDataset
from alignment.data.datasets.cifar import CIFAR10Dataset, CIFAR100Dataset
from alignment.data.datasets.imagenet import ImageNetDataset

__all__ = [
    'MNISTDataset',
    'CIFAR10Dataset', 
    'CIFAR100Dataset',
    'ImageNetDataset',
] 