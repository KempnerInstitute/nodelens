"""
CIFAR dataset wrappers for alignment analysis.
"""

from typing import Optional, List, Callable, Tuple
import torch
from torchvision import datasets, transforms
from pathlib import Path

from alignment_refactor.data.base import BaseDataset
from alignment_refactor.core.registry import register_dataset


@register_dataset("cifar10")
class CIFAR10Dataset(BaseDataset):
    """
    CIFAR-10 dataset wrapper with alignment-specific features.
    
    Provides the CIFAR-10 dataset (32x32 color images in 10 classes)
    with proper normalization and optional augmentation.
    """
    
    # Dataset statistics
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
    NUM_CLASSES = 10
    INPUT_SHAPE = (3, 32, 32)
    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
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
        """Initialize CIFAR-10 dataset."""
        super().__init__(
            name="CIFAR10",
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
        self._dataset = datasets.CIFAR10(
            root=self.data_path,
            train=self.train,
            transform=self.get_transform(),
            target_transform=self.target_transform,
            download=self.download
        )
    
    @property
    def mean(self) -> List[float]:
        """Dataset mean."""
        return self.MEAN
    
    @property
    def std(self) -> List[float]:
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
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ]
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        return self._dataset[idx]
    
    def get_targets(self) -> torch.Tensor:
        """Get all targets as a tensor."""
        return torch.tensor(self._dataset.targets)


@register_dataset("cifar100")
class CIFAR100Dataset(BaseDataset):
    """
    CIFAR-100 dataset wrapper with alignment-specific features.
    
    Provides the CIFAR-100 dataset (32x32 color images in 100 classes)
    with proper normalization and optional augmentation.
    """
    
    # Dataset statistics
    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]
    NUM_CLASSES = 100
    INPUT_SHAPE = (3, 32, 32)
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        normalize: bool = True,
        augment: bool = False,
        use_coarse_labels: bool = False,
        **config
    ):
        """
        Initialize CIFAR-100 dataset.
        
        Args:
            data_path: Path to store/load data
            train: Whether to load training set
            transform: Additional transforms
            target_transform: Transform for targets
            download: Whether to download if not found
            normalize: Whether to normalize
            augment: Whether to apply augmentation
            use_coarse_labels: Use 20 coarse labels instead of 100 fine
            **config: Additional configuration
        """
        self.use_coarse_labels = use_coarse_labels
        
        super().__init__(
            name="CIFAR100",
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
        self._dataset = datasets.CIFAR100(
            root=self.data_path,
            train=self.train,
            transform=self.get_transform(),
            target_transform=self.target_transform,
            download=self.download
        )
        
        # Get class names
        self._setup_class_names()
    
    def _setup_class_names(self):
        """Setup class names based on label type."""
        if self.use_coarse_labels:
            self.NUM_CLASSES = 20
            self.CLASS_NAMES = [
                'aquatic_mammals', 'fish', 'flowers', 'food_containers',
                'fruit_and_vegetables', 'household_electrical_devices',
                'household_furniture', 'insects', 'large_carnivores',
                'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
                'large_omnivores_and_herbivores', 'medium_mammals',
                'non-insect_invertebrates', 'people', 'reptiles',
                'small_mammals', 'trees', 'vehicles_1', 'vehicles_2'
            ]
        else:
            self.CLASS_NAMES = [f"class_{i}" for i in range(100)]
    
    @property
    def mean(self) -> List[float]:
        """Dataset mean."""
        return self.MEAN
    
    @property
    def std(self) -> List[float]:
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
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),
        ]
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        img, label = self._dataset[idx]
        
        # Use coarse labels if requested
        if self.use_coarse_labels:
            label = self._dataset.targets[idx] // 5  # Convert fine to coarse
        
        return img, label
    
    def get_targets(self) -> torch.Tensor:
        """Get all targets as a tensor."""
        targets = torch.tensor(self._dataset.targets)
        if self.use_coarse_labels:
            targets = targets // 5  # Convert fine to coarse
        return targets 