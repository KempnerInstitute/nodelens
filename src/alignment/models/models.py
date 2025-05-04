"""
Model implementations for neural network alignment experiments.

This module contains concrete implementations of models used in alignment
experiments, including MLPs and CNNs with various architectures.
"""

import logging
from typing import List, Optional, Dict, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

from alignment.models.base import AlignmentNetwork
from alignment.models.registry import register_model

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable number of hidden layers.
    
    Attributes:
        input_dim: Dimension of input features
        output_dim: Dimension of output (number of classes)
        num_hidden: List of hidden layer dimensions
        layers: Sequential container of linear layers and activations
    """
    
    def __init__(
        self, 
        input_dim: int = 784, 
        output_dim: int = 10, 
        num_hidden: List[int] = [128, 64],
        dropout_rate: float = 0.0
    ):
        """
        Initialize MLP with configurable architecture.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (number of classes)
            num_hidden: List of hidden layer dimensions
            dropout_rate: Dropout probability between layers
        """
        super().__init__()
        
        # Ensure num_hidden is a list
        if not isinstance(num_hidden, list):
            num_hidden = [num_hidden]
            
        # Build sequential layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for h_dim in num_hidden:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        return self.layers(x)


class CNN2P2(nn.Module):
    """
    Convolutional Neural Network with 2 convolutional layers and 2 pooling layers.
    
    Attributes:
        conv1: First convolutional layer
        conv2: Second convolutional layer
        fc1: First fully connected layer
        fc2: Last fully connected layer (output layer)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_hidden: List[int] = [128, 64],
        output_dim: int = 10,
        dropout_rate: float = 0.0
    ):
        """
        Initialize CNN2P2 network.
        
        Args:
            in_channels: Number of input channels
            num_hidden: List with hidden layer dimensions [conv_out_dim, fc_hidden_dim]
            output_dim: Number of output classes
            dropout_rate: Dropout probability between fully connected layers
        """
        super().__init__()
        
        # Default hidden dimensions if not provided
        if not isinstance(num_hidden, list) or len(num_hidden) < 2:
            num_hidden = [128, 64]
        
        conv_out_dim = num_hidden[0]
        fc_hidden_dim = num_hidden[1]
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # Calculate size after convolutions
        # For MNIST/CIFAR: 28x28 -> 24x24 -> 12x12 -> 8x8 -> 4x4
        # Output is 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(1024, fc_hidden_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # If input is flattened, reshape it
        if x.dim() == 2:
            batch_size = x.size(0)
            # Assuming square images for simplicity
            side_length = int(torch.sqrt(torch.tensor(x.size(1) / self.conv1.in_channels)))
            x = x.view(batch_size, self.conv1.in_channels, side_length, side_length)
            
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


# Register model constructors
@register_model("mlp")
def create_mlp(dropout_rate=0.0, alignment_layers=None, **kwargs) -> AlignmentNetwork:
    """
    Create an MLP model wrapped in AlignmentNetwork.
    
    Args:
        dropout_rate: Dropout probability between layers
        alignment_layers: Dictionary mapping layer names to input layers for alignment
        **kwargs: Arguments for MLP constructor
        
    Returns:
        AlignmentNetwork wrapping an MLP
    """
    base_model = MLP(dropout_rate=dropout_rate, **kwargs)
    
    # Default alignment layer configuration for MLP
    alignment_layer_names = alignment_layers
    if alignment_layer_names is None:
        alignment_layer_names = {
            name: i for i, name in enumerate([
                module_name for module_name, module in base_model.named_modules()
                if isinstance(module, nn.Linear) and not module_name.endswith("layers.0")
            ])
        }
    
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=alignment_layer_names)


@register_model("cnn2p2")
def create_cnn2p2(dropout_rate=0.0, alignment_layers=None, **kwargs) -> AlignmentNetwork:
    """
    Create a CNN2P2 model wrapped in AlignmentNetwork.
    
    Args:
        dropout_rate: Dropout probability between layers
        alignment_layers: Dictionary mapping layer names to input layers for alignment
        **kwargs: Arguments for CNN2P2 constructor
        
    Returns:
        AlignmentNetwork wrapping a CNN2P2
    """
    base_model = CNN2P2(dropout_rate=dropout_rate, **kwargs)
    
    # Default alignment layer configuration for CNN2P2
    alignment_layer_names = alignment_layers
    if alignment_layer_names is None:
        alignment_layer_names = {
            "fc1": 0,
            "fc2": 1
        }
    
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=alignment_layer_names)


@register_model("alexnet")
def create_alexnet(dropout_rate=0.0, alignment_layers=None, **kwargs) -> AlignmentNetwork:
    """
    Create an AlexNet model wrapped in AlignmentNetwork.
    
    Args:
        dropout_rate: Dropout probability between layers
        alignment_layers: Dictionary mapping layer names to input layers for alignment
        **kwargs: Arguments for AlexNet constructor
        
    Returns:
        AlignmentNetwork wrapping an AlexNet
    """
    base_model = alexnet(**kwargs)
    
    # Set dropout rates if not using default
    if dropout_rate != 0.0:
        for module in base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    # Default alignment layer configuration for AlexNet
    alignment_layer_names = alignment_layers
    if alignment_layer_names is None:
        alignment_layer_names = {
            "classifier.1": 0,  # First ReLU in classifier
            "classifier.4": 1,  # Second ReLU in classifier
            "classifier.6": 2   # Final layer
        }
    
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=alignment_layer_names)


# Dictionary for dataset-specific model parameters
DATASET_PARAMETERS = {
    "mlp": {
        "MNIST": dict(input_dim=784, output_dim=10),
        "CIFAR10": dict(input_dim=3072, output_dim=10),
        "CIFAR100": dict(input_dim=3072, output_dim=100),
    },
    "cnn2p2": {
        "MNIST": dict(in_channels=1, output_dim=10),
        "CIFAR10": dict(in_channels=3, num_hidden=[4096, 128], output_dim=10),
        "CIFAR100": dict(in_channels=3, num_hidden=[4096, 128], output_dim=100),
        "ImageNet": dict(in_channels=3, output_dim=1000),
    },
    "alexnet": {
        "MNIST": dict(num_classes=10),
        "CIFAR10": dict(num_classes=10),
        "CIFAR100": dict(num_classes=100),
        "ImageNet": dict(num_classes=1000),
    },
}


def get_model_dataset_parameters(model_name: str, dataset: str) -> Dict[str, Any]:
    """
    Get dataset-specific parameters for a model.
    
    Args:
        model_name: Name of the model
        dataset: Name of the dataset
        
    Returns:
        Dictionary of parameters specific to the model-dataset combination
        
    Raises:
        ValueError: If model_name or dataset is not supported
    """
    if model_name not in DATASET_PARAMETERS:
        available_models = ', '.join(sorted(DATASET_PARAMETERS.keys()))
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        
    if dataset not in DATASET_PARAMETERS[model_name]:
        available_datasets = ', '.join(sorted(DATASET_PARAMETERS[model_name].keys()))
        raise ValueError(f"Dataset '{dataset}' not supported for model '{model_name}'. Available datasets: {available_datasets}")
        
    return DATASET_PARAMETERS[model_name][dataset]


# Transform configuration for different model-dataset combinations
def gray_to_rgb(batch):
    """
    Convert grayscale images to RGB by repeating the channel.
    
    Args:
        batch: Input batch [batch_size, 1, height, width]
        
    Returns:
        RGB batch [batch_size, 3, height, width]
    """
    batch[0] = batch[0].expand(-1, 3, -1, -1)
    return batch


TRANSFORM_PARAMETERS = {
    "mlp": {
        "MNIST": dict(flatten=True, resize=None),
        "CIFAR10": dict(flatten=True, resize=None),
        "CIFAR100": dict(flatten=True, resize=None),
        "ImageNet": dict(flatten=True),
    },
    "cnn2p2": {
        "MNIST": dict(flatten=False, resize=None),
        "CIFAR10": dict(flatten=False, resize=None),
        "CIFAR100": dict(flatten=False, resize=None),
        "ImageNet": dict(flatten=False),
    },
    "alexnet": {
        "MNIST": dict(flatten=False, resize=(256, 256), extra_transform=[gray_to_rgb,]),
        "CIFAR10": dict(flatten=False, resize=(256, 256)),
        "CIFAR100": dict(flatten=False, resize=(256, 256)),
        "ImageNet": dict(center_crop=224, flatten=False, resize=(256, 256)),
    },
}


def get_transform_parameters(model_name: str, dataset: str) -> Dict[str, Any]:
    """
    Get transform parameters for a specific model-dataset combination.
    
    Args:
        model_name: Name of the model
        dataset: Name of the dataset
        
    Returns:
        Dictionary of transform parameters
        
    Raises:
        ValueError: If model_name or dataset is not supported
    """
    model_name = model_name.lower()
    if model_name not in TRANSFORM_PARAMETERS:
        available_models = ', '.join(sorted(TRANSFORM_PARAMETERS.keys()))
        raise ValueError(f"Transform parameters for model '{model_name}' not found. Available models: {available_models}")
        
    if dataset not in TRANSFORM_PARAMETERS[model_name]:
        available_datasets = ', '.join(sorted(TRANSFORM_PARAMETERS[model_name].keys()))
        raise ValueError(f"Transform parameters for dataset '{dataset}' not found for model '{model_name}'. Available datasets: {available_datasets}")
        
    return TRANSFORM_PARAMETERS[model_name][dataset]