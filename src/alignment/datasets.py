# --------------------------------------------
# datasets.py
# --------------------------------------------

import sys
import logging
from pathlib import Path
from warnings import warn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torchvision
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from alignment.models.base import AlignmentNetwork
from alignment.config import DatasetConfig, ExperimentConfig

logger = logging.getLogger(__name__)

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
        
        # Set the number of classes
        self._set_num_classes()

    def _set_num_classes(self):
        """Set the number of classes based on the dataset type."""
        dataset_name = self.__class__.__name__.upper()
        if dataset_name == 'MNIST' or dataset_name == 'CIFAR10':
            self.num_classes = 10
        elif dataset_name == 'CIFAR100':
            self.num_classes = 100
        elif dataset_name == 'IMAGENET2012':
            self.num_classes = 1000
        else:
            # Try to infer from the test dataset if possible
            try:
                if hasattr(self.test_dataset, 'classes'):
                    self.num_classes = len(self.test_dataset.classes)
                else:
                    # Default to a common value as fallback
                    logger.warning(f"Could not determine number of classes for {dataset_name}, defaulting to 10")
                    self.num_classes = 10
            except Exception as e:
                logger.warning(f"Error determining number of classes: {str(e)}, defaulting to 10")
                self.num_classes = 10

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
            logger.warning("WARNING: NaN or Inf values in measure_accuracy inputs")
            return torch.tensor(0.0, device=outputs.device)
            
        # Get the predicted classes (most likely class for each sample)
        _, predicted = outputs.max(1)
        
        # Calculate accuracy
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        
        # Convert to percentage if requested
        accuracy = (correct / total) * (100.0 if percentage else 1.0)        
            
        return torch.tensor(accuracy, device=outputs.device)

    def get_loader(self, batch_size=None, num_batches=None):
        """
        Get a data loader for the dataset.
        
        Args:
            batch_size: Batch size to use. Defaults to self.batch_size.
            num_batches: Number of batches to include. If None, includes all batches.
            
        Returns:
            DataLoader for the dataset.
        """
        if batch_size is None:
            batch_size = self.dataloader_parameters['batch_size']
            
        loader = self.train_loader if self.train_loader else self.test_loader
        
        if num_batches is not None:
            # Calculate total number of examples to use
            total_examples = batch_size * num_batches
            
            # Create a subset of the dataset
            subset = torch.utils.data.Subset(loader.dataset, range(min(total_examples, len(loader.dataset))))
            
            # Create a new loader with the subset
            return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        
        return loader

    def evaluate(self, model, device='cpu', num_batches=None, show_progress: bool = True):
        """
        Evaluate a model on the dataset.
        
        Args:
            model: Model to evaluate.
            device: Device to use for evaluation.
            num_batches: Number of batches to evaluate on. If None, evaluates on all batches.
            show_progress: Whether to show batch-level progress and final evaluation log.
        """
        model.eval()
        
        if show_progress: # Only log weight states if showing progress
            logger.info(f"Evaluating model {type(model).__name__} on device {device}")
            if hasattr(model, 'alignment_layers'):
                for i, layer in enumerate(model.alignment_layers):
                    if hasattr(layer, 'weight') and layer.weight is not None:
                        weight_shape = layer.weight.shape
                        # Ensure weight_shape has at least 2 dimensions for numel calculation
                        if len(weight_shape) >= 2:
                            total_weights = layer.weight.numel() # Use numel for total elements
                            num_zeros = (layer.weight == 0).sum().item()
                        
                            num_rows = weight_shape[0]
                            zero_rows = 0
                            for row_idx in range(num_rows):
                                if torch.all(layer.weight[row_idx] == 0):
                                    zero_rows += 1
                            
                            logger.info(f"Layer {i}: Shape {weight_shape}, "
                                    f"zeros: {num_zeros}/{total_weights} ({num_zeros/total_weights:.2%}), "
                                    f"pruned neurons: {zero_rows}/{num_rows} ({zero_rows/num_rows:.2%})")
                        else:
                            logger.info(f"Layer {i}: Shape {weight_shape} - Not a standard 2D+ weight matrix for this logging.")

        loader = self.get_loader(num_batches=num_batches)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Use tqdm for progress bar if show_progress is True
        batch_iterator = tqdm(loader, desc="Evaluating Batches", leave=False) if show_progress else loader

        with torch.no_grad():
            for inputs, targets in batch_iterator:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    logger.warning(f"NaN or Inf detected in model outputs")
                loss = self.measure_loss(outputs, targets)
                _, predicted = outputs.max(1)
                batch_total = targets.size(0)
                batch_correct = (predicted == targets).sum().item()
                total += batch_total
                correct += batch_correct
                total_loss += loss.item()
                
                # Batch logging only if show_progress is True (and not too frequent)
                if show_progress and isinstance(batch_iterator, tqdm):
                    # Update tqdm postfix instead of random logging
                    batch_iterator.set_postfix({
                        'loss': f"{total_loss / (batch_iterator.n + 1) if (batch_iterator.n + 1) > 0 else 0:.4f}", 
                        'acc': f"{100.0 * correct / total if total > 0 else 0:.2f}%"
                    })
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(loader) if len(loader) > 0 else float('inf')
        
        if show_progress: # Only log final summary if progress was shown
            logger.info(f"Evaluation complete: Accuracy = {accuracy:.2f}%, Loss = {avg_loss:.4f}")
        
        return accuracy, avg_loss


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
    Lookup dataset constructor from dataset registry by name.
    
    Args:
        dataset_name: Name of the dataset to retrieve
        build: If True, instantiate the dataset; otherwise return the constructor
        dataset_parameters: Parameters to pass to the dataset constructor
        transform_parameters: Parameters for transforms
        loader_parameters: Parameters for data loaders
        **kwargs: Additional arguments for dataset
        
    Returns:
        Dataset constructor or instance
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


def load_dataset(
    dataset_config: Union[DatasetConfig, Dict[str, Any]],
    batch_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    transform_params: Optional[Dict[str, Any]] = None
) -> DataSet:
    """
    Load a dataset based on configuration.
    
    Args:
        dataset_config: Dataset configuration object or dictionary
        batch_size: Optional batch size (overrides config)
        device: Optional device to place tensors on
        transform_params: Optional transform parameters (overrides config)
        
    Returns:
        Dataset object with loaders
    """
    # Handle dict or object config
    if isinstance(dataset_config, dict):
        dataset_name = dataset_config.get('dataset_name')
        dataset_path = dataset_config.get('data_path')
    else:
        dataset_name = dataset_config.dataset_name
        dataset_path = dataset_config.data_path
        
    # Check for required parameters
    if not dataset_name:
        raise ValueError("Dataset name must be provided in configuration")
        
    # Get transform parameters from model if necessary
    if transform_params is None:
        if hasattr(dataset_config, 'transform_params') and dataset_config.transform_params:
            transform_params = dataset_config.transform_params
        else:
            # Try to load from models registry if it exists
            try:
                from alignment.models.models import get_transform_parameters
                model_name = None
                if hasattr(dataset_config, 'model_name'):
                    model_name = dataset_config.model_name
                elif hasattr(dataset_config, 'model') and hasattr(dataset_config.model, 'model_name'):
                    model_name = dataset_config.model.model_name
                else:
                    model_name = 'mlp'  # Default
                
                transform_params = get_transform_parameters(model_name, dataset_name)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load transform parameters: {str(e)}")
                transform_params = {}
    
    # Set up loader parameters
    loader_params = {}
    if batch_size is not None:
        loader_params['batch_size'] = batch_size
    elif hasattr(dataset_config, 'batch_size'):
        loader_params['batch_size'] = dataset_config.batch_size
    
    # Load the dataset
    dataset = get_dataset(
        dataset_name=dataset_name,
        build=True,
        dataset_parameters={'root': dataset_path, 'download': True},
        transform_parameters=transform_params,
        loader_parameters=loader_params,
        device=device
    )
    
    # Log at debug level instead of info to reduce console output
    logger.debug(f"Loaded {dataset_name} dataset with batch size {loader_params.get('batch_size', 'default')}")
    
    return dataset


if __name__ == "__main__":
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise ValueError(f"Usage: {sys.argv[0]} [CONFIG_PATH]")

    cfg = ExperimentConfig.load(yaml_path)
    dataset = load_dataset(cfg.dataset)
    
    # Print dataset info
    print(f"Dataset: {cfg.dataset.dataset_name}")
    print(f"Train samples: {len(dataset.train_dataset)}")
    print(f"Test samples: {len(dataset.test_dataset)}")
    print(f"Batch size: {dataset.dataloader_parameters['batch_size']}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Sample a batch
    sample_batch = next(iter(dataset.train_loader))
    inputs, targets = dataset.unwrap_batch(sample_batch)
    print(f"Sample batch shape: {inputs.shape}")
    print(f"Sample targets shape: {targets.shape}")
    print(f"Sample target values: {targets[:5]}")
