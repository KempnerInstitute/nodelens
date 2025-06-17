"""
MNIST dataset wrapper for alignment analysis.
"""

from typing import Optional, List, Callable, Tuple
import torch
from torchvision import datasets, transforms
from pathlib import Path

from alignment.data.base import BaseDataset
from alignment.core.registry import register_dataset


@register_dataset("mnist")
class MNISTDataset(BaseDataset):
    """
    MNIST dataset wrapper with alignment-specific features.
    
    Provides the MNIST handwritten digits dataset with
    proper normalization and optional augmentation.
    """
    
    # Dataset statistics
    MEAN = 0.1307
    STD = 0.3081
    NUM_CLASSES = 10
    INPUT_SHAPE = (1, 28, 28)
    CLASS_NAMES = [str(i) for i in range(10)]
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize: bool = True,
        augment: bool = False,
        **config
    ):
        """
        Initialize MNIST dataset.
        
        Args:
            data_path: Path to store/load data
            train: Whether to load training set
            transform: Additional transforms
            target_transform: Transform for targets
            download: Whether to download if not found
            normalize: Whether to normalize
            augment: Whether to apply augmentation
            **config: Additional configuration
        """
        super().__init__(
            name="MNIST",
            data_path=data_path or "./data",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            normalize=normalize,
            augment=augment,
            **config
        )
        
        # Initialize the dataset
        self._dataset = datasets.MNIST(
            root=self.data_path,
            train=self.train,
            transform=self.get_transform(),
            target_transform=self.target_transform,
            download=self.download
        )
    
    @property
    def mean(self) -> float:
        """Dataset mean."""
        return self.MEAN
    
    @property
    def std(self) -> float:
        """Dataset standard deviation."""
        return self.STD
    
    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return self.NUM_CLASSES
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Input tensor shape."""
        return self.INPUT_SHAPE
    
    @property
    def class_names(self) -> List[str]:
        """List of class names."""
        return self.CLASS_NAMES
    
    def _get_basic_transforms(self) -> List[Callable]:
        """Get basic transforms."""
        return [transforms.ToTensor()]
    
    def _get_augmentation_transforms(self) -> List[Callable]:
        """Get augmentation transforms for training."""
        return [
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ]
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        return self._dataset[idx]
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sample weights for balanced training.
        
        Returns:
            Tensor of sample weights
        """
        # Count samples per class
        class_counts = torch.zeros(self.NUM_CLASSES)
        for _, label in self._dataset:
            class_counts[label] += 1
        
        # Compute weights
        class_weights = 1.0 / class_counts
        sample_weights = torch.zeros(len(self._dataset))
        
        for idx, (_, label) in enumerate(self._dataset):
            sample_weights[idx] = class_weights[label]
        
        return sample_weights
    
    def get_targets(self) -> torch.Tensor:
        """Get all targets as a tensor."""
        return torch.tensor(self._dataset.targets) 