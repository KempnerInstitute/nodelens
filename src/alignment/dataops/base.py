"""

Base classes for dataset wrappers.

This module provides base functionality for all dataset implementations,
ensuring consistency across different datasets.
"""

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from alignment.core.base import BaseDataset as CoreBaseDataset

logger = logging.getLogger(__name__)


class BaseDataset(CoreBaseDataset):
    """
    Extended base dataset class with additional utilities.

    This class provides common functionality for all dataset wrappers,
    including transforms, normalization, and data augmentation.
    """

    def __init__(
        self,
        name: str,
        data_path: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize: bool = True,
        augment: bool = False,
        **config: Any,
    ):
        """
        Initialize the dataset.

        Args:
            name: Dataset name
            data_path: Path to dataset files
            train: Whether this is training data
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            download: Whether to download dataset if not found
            normalize: Whether to apply normalization
            augment: Whether to apply data augmentation
            **config: Additional configuration
        """
        super().__init__(name, data_path, **config)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.normalize = normalize
        self.augment = augment

        # Will be set by subclasses
        self._dataset = None
        self._mean = None
        self._std = None

    @property
    @abstractmethod
    def mean(self) -> Union[float, List[float]]:
        """Dataset mean for normalization."""
        pass

    @property
    @abstractmethod
    def std(self) -> Union[float, List[float]]:
        """Dataset standard deviation for normalization."""
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """List of class names."""
        pass

    def get_transform(self) -> Optional[Callable]:
        """
        Get the complete transform pipeline.

        Returns:
            Transform callable or None
        """
        from torchvision import transforms

        transform_list = []

        # Add data augmentation if training
        if self.train and self.augment:
            transform_list.extend(self._get_augmentation_transforms())

        # Add basic transforms
        transform_list.extend(self._get_basic_transforms())

        # Add normalization
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        # Combine with user transform
        if transform_list:
            base_transform = transforms.Compose(transform_list)
            if self.transform is not None:
                # Chain transforms
                return transforms.Compose([base_transform, self.transform])
            return base_transform

        return self.transform

    @abstractmethod
    def _get_basic_transforms(self) -> List[Callable]:
        """Get basic transforms (e.g., ToTensor)."""
        pass

    @abstractmethod
    def _get_augmentation_transforms(self) -> List[Callable]:
        """Get augmentation transforms for training."""
        pass

    def get_train_loader(
        self, batch_size: int, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True, drop_last: bool = True, **kwargs: Any
    ) -> DataLoader:
        """Get training data loader."""
        if not self.train:
            logger.warning(f"Getting train loader from test dataset {self.name}")

        return DataLoader(
            self._dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs
        )

    def get_test_loader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 4, pin_memory: bool = True, drop_last: bool = False, **kwargs: Any
    ) -> DataLoader:
        """Get test data loader."""
        if self.train:
            logger.warning(f"Getting test loader from train dataset {self.name}")

        return DataLoader(
            self._dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs
        )

    def get_val_loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4, **kwargs: Any) -> DataLoader:
        """Get validation loader (same as test for most datasets)."""
        return self.get_test_loader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def get_subset_loader(self, indices: Union[List[int], torch.Tensor], batch_size: int, shuffle: bool = False, **kwargs: Any) -> DataLoader:
        """
        Get a data loader for a subset of the dataset.

        Args:
            indices: Indices of samples to include
            batch_size: Batch size
            shuffle: Whether to shuffle
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader for the subset
        """
        from torch.utils.data import Subset

        subset = Subset(self._dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_class_balanced_sampler(self) -> Sampler:
        """
        Get a sampler that balances classes.

        Returns:
            Sampler for class-balanced training
        """
        from torch.utils.data import WeightedRandomSampler

        # Get class counts
        if hasattr(self._dataset, "targets"):
            targets = torch.tensor(self._dataset.targets)
        elif hasattr(self._dataset, "labels"):
            targets = torch.tensor(self._dataset.labels)
        else:
            raise AttributeError(f"Dataset {self.name} doesn't have targets/labels")

        class_counts = torch.bincount(targets)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[targets]

        return WeightedRandomSampler(weights=sample_weights, num_samples=len(targets), replacement=True)

    def compute_statistics(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Args:
            num_samples: Number of samples to use (None = all)

        Returns:
            Dictionary with statistics
        """
        loader = self.get_test_loader(batch_size=100, shuffle=True)

        # Initialize accumulators
        mean_acc = 0
        std_acc = 0
        min_val = float("inf")
        max_val = float("-inf")
        total_samples = 0

        for i, (data, _) in enumerate(loader):
            if num_samples and total_samples >= num_samples:
                break

            # Update statistics
            batch_samples = data.shape[0]
            mean_acc += data.mean().item() * batch_samples
            std_acc += data.std().item() * batch_samples
            min_val = min(min_val, data.min().item())
            max_val = max(max_val, data.max().item())
            total_samples += batch_samples

        return {
            "mean": mean_acc / total_samples,
            "std": std_acc / total_samples,
            "min": min_val,
            "max": max_val,
            "num_samples": total_samples,
            "shape": self.input_shape,
            "num_classes": self.num_classes,
        }


class DatasetWrapper:
    """
    Wrapper that provides a unified interface for different dataset types.

    This is useful for handling custom datasets or datasets from different
    sources with a consistent API.
    """

    def __init__(
        self,
        dataset: Dataset,
        name: str,
        num_classes: int,
        input_shape: Tuple[int, ...],
        mean: Optional[Union[float, List[float]]] = None,
        std: Optional[Union[float, List[float]]] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the dataset wrapper.

        Args:
            dataset: PyTorch dataset to wrap
            name: Dataset name
            num_classes: Number of classes
            input_shape: Shape of input samples
            mean: Mean for normalization
            std: Standard deviation for normalization
            class_names: Optional list of class names
        """
        self.dataset = dataset
        self.name = name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.mean = mean or 0.0
        self.std = std or 1.0
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    def get_loader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4, **kwargs: Any) -> DataLoader:
        """Get a data loader for the wrapped dataset."""
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """Get a sample from the dataset."""
        return self.dataset[idx]
