"""
ImageNet dataset wrapper for alignment analysis.
"""

from typing import Optional, List, Callable, Tuple
import torch
from torchvision import datasets, transforms
from pathlib import Path
import logging

from alignment.data.base import BaseDataset
from alignment.core.registry import register_dataset

logger = logging.getLogger(__name__)


@register_dataset("imagenet")
class ImageNetDataset(BaseDataset):
    """
    ImageNet dataset wrapper with alignment-specific features.
    
    Provides the ImageNet dataset with proper normalization
    and optional augmentation. Supports both full ImageNet
    and ImageNet-1K variants.
    """
    
    # Dataset statistics (ImageNet normalization)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NUM_CLASSES = 1000
    INPUT_SHAPE = (3, 224, 224)
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,  # ImageNet typically not downloadable
        normalize: bool = True,
        augment: bool = False,
        image_size: int = 224,
        crop_size: int = 224,
        **config
    ):
        """
        Initialize ImageNet dataset.
        
        Args:
            data_path: Path to ImageNet data
            train: Whether to load training set
            transform: Additional transforms
            target_transform: Transform for targets
            download: Whether to download (not supported for ImageNet)
            normalize: Whether to normalize
            augment: Whether to apply augmentation
            image_size: Size for resizing images
            crop_size: Size for random/center crop
            **config: Additional configuration
        """
        self.image_size = image_size
        self.crop_size = crop_size
        self.INPUT_SHAPE = (3, crop_size, crop_size)
        
        super().__init__(
            name="ImageNet",
            data_path=data_path or "/path/to/imagenet",
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
            normalize=normalize,
            augment=augment,
            **config
        )
        
        # Check if data exists
        data_dir = Path(self.data_path) / ("train" if self.train else "val")
        if not data_dir.exists():
            raise ValueError(
                f"ImageNet data not found at {data_dir}. "
                "Please download ImageNet and set the correct data_path."
            )
        
        # Initialize the dataset
        try:
            self._dataset = datasets.ImageFolder(
                root=str(data_dir),
                transform=self.get_transform(),
                target_transform=self.target_transform
            )
            logger.info(f"Loaded ImageNet {'train' if train else 'val'} set with {len(self._dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load ImageNet dataset: {e}")
            raise
        
        # Setup class names
        self._setup_class_names()
    
    def _setup_class_names(self):
        """Setup ImageNet class names."""
        # This would ideally load from a class mapping file
        # For now, use indices as names
        self.CLASS_NAMES = [f"class_{i}" for i in range(self.NUM_CLASSES)]
        
        # If the dataset has class_to_idx mapping, use it
        if hasattr(self._dataset, 'class_to_idx'):
            idx_to_class = {v: k for k, v in self._dataset.class_to_idx.items()}
            self.CLASS_NAMES = [idx_to_class.get(i, f"class_{i}") 
                                for i in range(len(idx_to_class))]
            self.NUM_CLASSES = len(self.CLASS_NAMES)
    
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
        if self.train:
            return [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
        else:
            # For validation, use center crop
            return [
                transforms.Resize(int(self.image_size * 256 / 224)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
            ]
    
    def _get_augmentation_transforms(self) -> List[Callable]:
        """Get augmentation transforms for training."""
        return [
            transforms.RandomResizedCrop(
                self.crop_size,
                scale=(0.08, 1.0),
                ratio=(3/4, 4/3)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
        ]
    
    def __len__(self) -> int:
        """Dataset length."""
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample."""
        return self._dataset[idx]
    
    def get_targets(self) -> torch.Tensor:
        """Get all targets as a tensor."""
        # ImageFolder doesn't have a targets attribute, so we build it
        targets = [self._dataset.samples[i][1] for i in range(len(self._dataset))]
        return torch.tensor(targets) 