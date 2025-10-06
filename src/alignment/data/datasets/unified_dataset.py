"""
Unified dataset wrapper for alignment analysis.

This module provides a single, flexible dataset class that can handle
various dataset types without code duplication.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from alignment.core.registry import register_dataset
from alignment.data.base import BaseDataset

logger = logging.getLogger(__name__)


# Dataset configurations
DATASET_CONFIGS = {
        "mnist": {
            "dataset_class": datasets.MNIST,
            "mean": 0.1307,
            "std": 0.3081,
            "num_classes": 10,
            "input_shape": (1, 28, 28),
            "class_names": [str(i) for i in range(10)],
            "augmentation": {"rotation": 10, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
        },
        "fashion_mnist": {
            "dataset_class": datasets.FashionMNIST,
            "mean": 0.2860,
            "std": 0.3530,
            "num_classes": 10,
            "input_shape": (1, 28, 28),
            "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "augmentation": {"rotation": 10, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
        },
        "cifar10": {
            "dataset_class": datasets.CIFAR10,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616],
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "class_names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "augmentation": {
                "crop": 32,
                "padding": 4,
                "horizontal_flip": True,
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            },
        },
        "cifar100": {
            "dataset_class": datasets.CIFAR100,
            "mean": [0.5071, 0.4865, 0.4409],
            "std": [0.2673, 0.2564, 0.2762],
            "num_classes": 100,
            "input_shape": (3, 32, 32),
            "augmentation": {
                "crop": 32,
                "padding": 4,
                "horizontal_flip": True,
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
                "rotation": 15,
            },
        },
        "imagenet": {
            "dataset_class": datasets.ImageNet,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
            "input_shape": (3, 224, 224),
            "augmentation": {
                "random_resized_crop": 224,
                "horizontal_flip": True,
                "color_jitter": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
            },
            "val_transforms": {"resize": 256, "center_crop": 224},
        },
        "svhn": {
            "dataset_class": datasets.SVHN,
            "mean": [0.4377, 0.4438, 0.4728],
            "std": [0.1980, 0.2010, 0.1970],
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "class_names": [str(i) for i in range(10)],
            "split_arg": "split",  # SVHN uses 'split' instead of 'train'
            "augmentation": {
                "crop": 32,
                "padding": 4,
                "horizontal_flip": False,  # Numbers shouldn't be flipped
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            },
        },
    }


@register_dataset("unified")
class UnifiedDataset(BaseDataset):
    """
    Unified dataset wrapper that can handle multiple dataset types.

    This class provides a single interface for various datasets,
    eliminating code duplication while maintaining flexibility.
    """

    def __init__(
        self,
        dataset_type: str,
        data_path: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize: bool = True,
        augment: bool = False,
        custom_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize unified dataset.

        Args:
            dataset_type: Type of dataset ('mnist', 'cifar10', 'imagenet', etc.)
            data_path: Path to store/load data
            train: Whether to load training set
            transform: Additional transforms
            target_transform: Transform for targets
            download: Whether to download if not found
            normalize: Whether to normalize
            augment: Whether to apply augmentation
            custom_config: Custom configuration to override defaults
            **kwargs: Additional dataset-specific arguments
        """
        # Get dataset configuration
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset type: {dataset_type}. " f"Available: {list(DATASET_CONFIGS.keys())}")

        self.dataset_config = DATASET_CONFIGS[dataset_type].copy()

        # Apply custom config if provided
        if custom_config:
            self.dataset_config.update(custom_config)

        # Initialize base class
        super().__init__(
            name=dataset_type.upper(),
            data_path=data_path or "./data",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            normalize=normalize,
            augment=augment,
            **kwargs,
        )

        self.dataset_type = dataset_type
        self._initialize_dataset(**kwargs)

    def _initialize_dataset(self, **kwargs):
        """Initialize the underlying dataset."""
        dataset_class = self.dataset_config["dataset_class"]

        # Prepare dataset arguments
        dataset_args = {
            "root": self._data_path,
            "transform": self.get_transform(),
            "target_transform": self.target_transform,
            "download": self.download,
        }

        # Handle dataset-specific arguments
        if self.dataset_type == "svhn":
            # SVHN uses 'split' instead of 'train'
            dataset_args["split"] = "train" if self.train else "test"
        elif self.dataset_type == "imagenet":
            # ImageNet uses 'split' parameter
            dataset_args["split"] = "train" if self.train else "val"
        else:
            # Most datasets use 'train' parameter
            dataset_args["train"] = self.train

        # Add any additional kwargs
        dataset_args.update(kwargs)

        # Create dataset
        try:
            self._dataset = dataset_class(**dataset_args)
        except Exception as e:
            logger.error(f"Failed to initialize {self.dataset_type} dataset: {e}")
            raise

    @property
    def mean(self) -> Union[float, List[float]]:
        """Dataset mean for normalization."""
        return self.dataset_config["mean"]

    @property
    def std(self) -> Union[float, List[float]]:
        """Dataset standard deviation for normalization."""
        return self.dataset_config["std"]

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        return self.dataset_config["num_classes"]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Shape of a single input sample."""
        return self.dataset_config["input_shape"]

    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        if "class_names" in self.dataset_config:
            return self.dataset_config["class_names"]
        else:
            # Generate generic class names
            return [f"class_{i}" for i in range(self.num_classes)]

    def _get_basic_transforms(self) -> List[Callable]:
        """Get basic transforms based on dataset type."""
        transforms_list = []

        # Add dataset-specific basic transforms
        if self.dataset_type == "imagenet" and not self.train:
            # ImageNet validation needs resize and center crop
            val_config = self.dataset_config.get("val_transforms", {})
            if "resize" in val_config:
                transforms_list.append(transforms.Resize(val_config["resize"]))
            if "center_crop" in val_config:
                transforms_list.append(transforms.CenterCrop(val_config["center_crop"]))

        # Always convert to tensor
        transforms_list.append(transforms.ToTensor())

        return transforms_list

    def _get_augmentation_transforms(self) -> List[Callable]:
        """Get augmentation transforms based on dataset configuration."""
        if "augmentation" not in self.dataset_config:
            return []

        aug_config = self.dataset_config["augmentation"]
        transforms_list = []

        # Random crop with padding
        if "crop" in aug_config and "padding" in aug_config:
            transforms_list.append(transforms.RandomCrop(aug_config["crop"], padding=aug_config["padding"]))

        # Random resized crop (for ImageNet)
        if "random_resized_crop" in aug_config:
            transforms_list.append(transforms.RandomResizedCrop(aug_config["random_resized_crop"]))

        # Random horizontal flip
        if aug_config.get("horizontal_flip", False):
            transforms_list.append(transforms.RandomHorizontalFlip())

        # Random rotation
        if "rotation" in aug_config:
            transforms_list.append(transforms.RandomRotation(aug_config["rotation"]))

        # Random affine
        if "translate" in aug_config or "scale" in aug_config:
            transforms_list.append(transforms.RandomAffine(degrees=0, translate=aug_config.get("translate"), scale=aug_config.get("scale")))

        # Color jitter
        if "color_jitter" in aug_config:
            jitter_params = aug_config["color_jitter"]
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=jitter_params.get("brightness", 0),
                    contrast=jitter_params.get("contrast", 0),
                    saturation=jitter_params.get("saturation", 0),
                    hue=jitter_params.get("hue", 0),
                )
            )

        return transforms_list

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get a sample from the dataset."""
        return self._dataset[idx]

    def get_targets(self) -> torch.Tensor:
        """Get all targets as a tensor."""
        if hasattr(self._dataset, "targets"):
            return torch.tensor(self._dataset.targets)
        elif hasattr(self._dataset, "labels"):
            return torch.tensor(self._dataset.labels)
        else:
            # Fallback: iterate through dataset
            targets = []
            for _, target in self._dataset:
                targets.append(target)
            return torch.tensor(targets)

    def add_dataset_type(self, name: str, dataset_class: type, config: Dict[str, Any]) -> None:
        """
        Add a new dataset type to the registry.

        Args:
            name: Name for the dataset type
            dataset_class: Dataset class (e.g., from torchvision)
            config: Configuration dictionary with required fields
        """
        required_fields = ["mean", "std", "num_classes", "input_shape"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Config must include '{field}'")

        config["dataset_class"] = dataset_class
        DATASET_CONFIGS[name] = config
        logger.info(f"Added new dataset type: {name}")


# Register specific dataset types for backward compatibility
def create_dataset_class(dataset_type):
    """Create a dataset class for a specific type."""

    @register_dataset(dataset_type)
    class SpecificDataset(UnifiedDataset):
        """Dataset wrapper for specific dataset type."""

        def __init__(self, **kwargs):
            super().__init__(dataset_type=dataset_type, **kwargs)

    # Set proper class name
    SpecificDataset.__name__ = f"{dataset_type.upper()}Dataset"
    SpecificDataset.__qualname__ = f"{dataset_type.upper()}Dataset"
    return SpecificDataset


# Create and register all dataset types
for dataset_type in DATASET_CONFIGS.keys():
    create_dataset_class(dataset_type)
