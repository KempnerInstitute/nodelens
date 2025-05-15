# --------------------------------------------
# datasets.py
# --------------------------------------------

import sys
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod

import torch
import torchvision
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms

from alignment.models.base import AlignmentNetwork
from alignment.config import ExperimentConfig

REQUIRED_PROPERTIES = ["dataset_constructor", "loss_function"]

def default_loader_parameters(
    distributed,
    batch_size=1024,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
):
    """
    contains the default dataloader parameters with the option of updating them
    using key word arguments

    # adjusting 'shuffle' when using distributed data parallel.
    """
    default_parameters = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if distributed else shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return default_parameters

class DataSet(ABC):
    """
    Abstract base class for any dataset wrapper.
    Responsible for creating train/test datasets & loaders, 
    storing transforms, etc.
    """

    def __init__(
        self,
        device=None,
        distributed=False,
        dataset_parameters={},
        transform_parameters={},
        loader_parameters={},
    ):
        self.set_properties()
        self.check_properties()

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed

        self.extra_transform = transform_parameters.pop("extra_transform", None)

        self.transform_parameters = transform_parameters
        self.make_transform(**transform_parameters)

        self.dataloader_parameters = default_loader_parameters(distributed, **loader_parameters)
        self.dataset_parameters = dataset_parameters
        self.load_dataset(**dataset_parameters)

    def check_properties(self):
        if not all([hasattr(self, prop) for prop in REQUIRED_PROPERTIES]):
            not_found = [prop for prop in REQUIRED_PROPERTIES if not hasattr(self, prop)]
            raise ValueError(f"The following required properties were not set: {not_found}")

    @abstractmethod
    def set_properties(self):
        """
        Must set:
          self.dataset_constructor
          self.loss_function
        and optionally self.dist_params for normalization, etc.
        """
        pass

    @abstractmethod
    def dataset_kwargs(self, train=True, **kwargs):
        """
        Returns the dict of kwargs for constructing the train or test dataset.
        """
        pass

    def load_dataset(self, **kwargs):
        """Instantiate self.train_dataset and self.test_dataset, then build DataLoaders."""
        self.train_dataset = self.dataset_constructor(**self.dataset_kwargs(train=True, **kwargs))
        self.test_dataset = self.dataset_constructor(**self.dataset_kwargs(train=False, **kwargs))
        self.train_sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        self.test_sampler = DistributedSampler(self.test_dataset) if self.distributed else None
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, **self.dataloader_parameters)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, sampler=self.test_sampler, **self.dataloader_parameters)

    def unwrap_batch(self, batch, device=None):
        """
        Simple method for unwrapping a batch (inputs, targets) 
        and placing them on the correct device.
        """
        device = self.device if device is None else device
        if self.extra_transform:
            if isinstance(self.extra_transform, list):
                for et in self.extra_transform:
                    batch = et(batch)
            else:
                warn("extra_transform is not a list, this is deprecated!", DeprecationWarning, stacklevel=2)
                batch = self.extra_transform(batch)
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        return inputs, targets

    def make_transform(self, center_crop=None, resize=None, flatten=False, out_channels=None):
        """
        Create a transforms.Compose for the dataset.
        Default: converts to float32, normalizes, etc.

        # For standard dataset usage, we typically do:
        #   transforms.ToImage(),
        #   transforms.ToDtype(torch.float32, scale=True),
        #   transforms.Normalize(...),
        #   etc.
        """
        use_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
        if center_crop:
            use_transforms.append(transforms.CenterCrop(center_crop))
        use_transforms.append(transforms.Normalize((self.dist_params["mean"]), (self.dist_params["std"])))
        if resize:
            use_transforms.append(transforms.Resize(resize, antialias=True))
        if out_channels:
            use_transforms.append(transforms.Grayscale(num_output_channels=out_channels))
        if flatten:
            use_transforms.append(transforms.Lambda(torch.flatten))
        self.transform = transforms.Compose(use_transforms)

    def measure_loss(self, outputs, targets, reduction=None):
        """Compute the loss via self.loss_function, optionally changing reduction."""
        if reduction is None:
            return self.loss_function(outputs, targets)
        standard_reduction = self.loss_function.reduction
        self.loss_function.reduction = reduction
        loss = self.loss_function(outputs, targets)
        self.loss_function.reduction = standard_reduction
        return loss

    def measure_accuracy(self, outputs, targets, k=1, percentage=True):
        """
        Return top-k accuracy on classification problems.
        By default, top-1 accuracy in percentage form.
        """
        # Use a simple, direct implementation that's guaranteed to work correctly
        # First check for invalid values
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("WARNING: NaN or Inf values in measure_accuracy inputs")
            return torch.tensor(0.0, device=outputs.device)
            
        # Get the predicted classes (most likely class for each sample)
        _, predicted = outputs.max(1)
        
        # Calculate accuracy
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        
        # Convert to percentage if requested
        accuracy = (correct / total) * (100.0 if percentage else 1.0)        

            
        return torch.tensor(accuracy, device=outputs.device)


class MNIST(DataSet):
    def set_properties(self):
        """defines the required properties for MNIST"""
        self.dataset_constructor = torchvision.datasets.MNIST
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.1307], std=[0.3081])

    def dataset_kwargs(self, train=True, download=False, root=None):
        kwargs = dict(
            train=train,
            download=download,
            root=root,
            transform=self.transform,
        )
        return kwargs

class CIFAR10(DataSet):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR10
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def dataset_kwargs(self, train=True, download=False, root=None):
        kwargs = dict(
            train=train,
            download=download,
            root=root,
            transform=self.transform,
        )
        return kwargs

class CIFAR100(CIFAR10):
    def set_properties(self):
        self.dataset_constructor = torchvision.datasets.CIFAR100
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

class ImageNet2012(DataSet):
    def set_properties(self):
        """
        defines the required properties for ImageNet 2012
        (ILSVRC2012) with 1000 classes.
        """
        self.dataset_constructor = torchvision.datasets.ImageNet
        self.loss_function = nn.CrossEntropyLoss()
        self.dist_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.center_crop = 224

    def dataset_kwargs(self, train=True, root=None):
        kwargs = dict(
            split="train" if train else "val",
            root=root,
            transform=self.transform,
        )
        return kwargs

DATASET_REGISTRY = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "ImageNet": ImageNet2012,
}

def get_dataset(
    dataset_name,
    build=False,
    dataset_parameters={},
    transform_parameters={},
    loader_parameters={},
    **kwargs,
):
    """
    lookup dataset constructor from dataset registry by name.
    if build=True, return an instantiated dataset object,
    otherwise return the constructor.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset ({dataset_name}) is not in DATASET_REGISTRY")
    dataset = DATASET_REGISTRY[dataset_name]
    if build:
        if not isinstance(transform_parameters, dict):
            raise TypeError("transform_parameters must be a dictionary")
        return dataset(
            dataset_parameters=dataset_parameters,
            transform_parameters=transform_parameters,
            loader_parameters=loader_parameters,
            **kwargs,
        )
    return dataset

if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1]
    except IndexError:
        raise ValueError(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = ExperimentConfig.load(yaml_path)
    dataset = get_dataset(cfg.dataset.name, build=True, dataset_parameters=dict(download=True, root=Path(cfg.dataset.path)))
