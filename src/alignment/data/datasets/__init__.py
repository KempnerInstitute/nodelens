"""
Dataset implementations for alignment analysis.

NOTE: This module re-exports from alignment.dataops.datasets for backward compatibility.
The canonical location is alignment.dataops.datasets.
"""

# Re-export everything from the canonical location
from alignment.dataops.datasets import (
    DATASET_CONFIGS,
    CIFAR10Dataset,
    CIFAR100Dataset,
    FashionMNISTDataset,
    ImageNetDataset,
    MNISTDataset,
    SVHNDataset,
    UnifiedDataset,
    get_dataset,
)

# Try to import text datasets (optional)
try:
    from alignment.dataops.datasets import C4Dataset, TextDataset, WikiTextDataset, load_text_dataset

    HAS_TEXT_DATASETS = True
except ImportError:
    HAS_TEXT_DATASETS = False
    TextDataset = None
    WikiTextDataset = None
    C4Dataset = None
    load_text_dataset = None


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
