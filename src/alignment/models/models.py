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
        hidden_dims: List of hidden layer dimensions
        layers: Sequential container of linear layers and activations
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [100, 100, 50],
        output_dim: int = 10,
        dropout_rate: float = 0.5,
        activation_type: str = "relu"
    ):
        """
        Initialize MLP with configurable architecture.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (number of classes)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability between layers
            activation_type: Activation function type
        """
        super().__init__()
        
        if activation_type.lower() == "relu":
            activation_fn = nn.ReLU()
        elif activation_type.lower() == "tanh":
            activation_fn = nn.Tanh()
        elif activation_type.lower() == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation_type.lower() == "identity":
            activation_fn = nn.Identity()
        else:
            logger.warning(f"Unknown activation_type '{activation_type}'. Defaulting to ReLU.")
            activation_fn = nn.ReLU()

        layers = []
        current_dim = input_dim
        
        # Input layer (can be considered part of the first hidden block conceptually from v2)
        # Or as a separate input projection if no dropout/activation is desired right after it.
        # For simplicity matching v2 structure: (Linear -> Activation) then (Dropout -> Linear -> Activation)
        
        # First hidden layer (or input projection + first hidden layer based on interpretation)
        if hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dims[0]))
            layers.append(activation_fn)
            current_dim = hidden_dims[0]

            # Subsequent hidden layers
            for i in range(len(hidden_dims) - 1):
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.Linear(current_dim, hidden_dims[i+1]))
                layers.append(activation_fn)
                current_dim = hidden_dims[i+1]
        
        # Output layer
        if dropout_rate > 0.0 and hidden_dims: # Only add dropout if there were hidden layers to apply it after
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, channels, height, width]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Ensure input is flattened
        if x.dim() > 2:
            # If input is [batch_size, channels, height, width], flatten it
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
            # Ensure the flattened dimension matches the expected input_dim
            if x.size(1) != self.network[0].in_features:
                # Log a warning about dimension mismatch
                logger.warning(
                    f"Input dimension mismatch. Expected {self.network[0].in_features}, "
                    f"got {x.size(1)}. Adjusting to expected dimension."
                )
                # Calculate expected dimensions based on common image sizes
                if self.network[0].in_features == 784:  # MNIST (28x28)
                    x = x.view(batch_size, -1)[:, :784]
                elif self.network[0].in_features == 3072:  # CIFAR (3x32x32)
                    x = x.view(batch_size, -1)[:, :3072]
        
        return self.network(x)


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
        output_dim: int = 10,
        conv_channels: List[int] = [32, 64],
        kernel_sizes: List[int] = [5, 5],
        strides: List[int] = [1, 1],
        paddings: List[int] = [2, 2],
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        hidden_fc_dim: int = 128,
        dropout_rate: float = 0.5,
        example_input_hw: Tuple[int, int] = (28, 28)
    ):
        """
        Initialize CNN2P2 network.
        
        Args:
            in_channels: Number of input channels
            output_dim: Number of output classes
            conv_channels: List of convolutional layer output dimensions
            kernel_sizes: List of convolutional layer kernel sizes
            strides: List of convolutional layer strides
            paddings: List of convolutional layer paddings
            pool_kernel_size: Pooling layer kernel size
            pool_stride: Pooling layer stride
            hidden_fc_dim: Size of the one hidden FC layer
            dropout_rate: Dropout probability between fully connected layers
            example_input_hw: Example input image height and width
        """
        super().__init__()

        if not (len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings) == 2):
            raise ValueError("conv_channels, kernel_sizes, strides, paddings must all be lists of 2 elements.")

        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        h, w = example_input_hw

        # Conv Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(current_channels, conv_channels[0], kernel_sizes[0], strides[0], paddings[0]),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        )
        current_channels = conv_channels[0]
        h = (h - kernel_sizes[0] + 2 * paddings[0]) // strides[0] + 1 # after conv1
        h = (h - pool_kernel_size) // pool_stride + 1 # after pool1
        w = (w - kernel_sizes[0] + 2 * paddings[0]) // strides[0] + 1
        w = (w - pool_kernel_size) // pool_stride + 1

        # Conv Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(current_channels, conv_channels[1], kernel_sizes[1], strides[1], paddings[1]),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        )
        current_channels = conv_channels[1]
        h = (h - kernel_sizes[1] + 2 * paddings[1]) // strides[1] + 1 # after conv2
        h = (h - pool_kernel_size) // pool_stride + 1 # after pool2
        w = (w - kernel_sizes[1] + 2 * paddings[1]) // strides[1] + 1
        w = (w - pool_kernel_size) // pool_stride + 1
        
        flattened_dim = current_channels * h * w

        self.fc_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout_rate), # As per v2: Dropout before first FC linear
            nn.Linear(flattened_dim, hidden_fc_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), # As per v2: Dropout before second FC linear
            nn.Linear(hidden_fc_dim, output_dim)
        )

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
            side_length = int(torch.sqrt(torch.tensor(x.size(1) / self.conv1[0].in_channels)))
            x = x.view(batch_size, self.conv1[0].in_channels, side_length, side_length)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc_layers(x)
        return x


# Register model constructors
@register_model("mlp")
def create_mlp(config_model: Dict, # Expecting a dict derived from ModelConfig fields
               alignment_layers: Optional[Dict[str, Any]] = None) -> AlignmentNetwork:
    
    # Parameters from your YAML mapped to the new MLP class
    mlp_params = {
        "input_dim": config_model.get("input_dim", 784),
        "hidden_dims": config_model.get("hidden_dims", [100, 100, 50]),
        "output_dim": config_model.get("output_dim", 10),
        "dropout_rate": config_model.get("dropout_rate", config_model.get("dropout", 0.5)), # aLlows "dropout" or "dropout_rate"
        "activation_type": config_model.get("activation", "relu")
    }
    base_model = MLP(**mlp_params)
    
    # Default alignment layer naming for the new MLP structure
    # Linear layers are now direct children of base_model.network (a Sequential module)
    # Their names will be like "network.0", "network.2" (if dropout is present), etc.
    # Or, if we name them: layerInput, layerHidden.0.linear, layerOutput.linear
    # For AlignmentNetwork, we need the names of modules that HAVE a .weight attribute.
    default_alignment_layer_names = {}
    idx_for_alignment = 0
    for i, layer in enumerate(base_model.network):
        if isinstance(layer, nn.Linear):
            # Using a generic name format for layers within the sequential block
            default_alignment_layer_names[f"network.{i}"] = idx_for_alignment 
            idx_for_alignment += 1

    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names
    
    # Extract cnn_mode from config_model, default if not present
    cnn_mode = config_model.get("cnn_mode", "unfold")
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode)


@register_model("cnn2p2")
def create_cnn2p2(config_model: Dict, # Expecting a dict derived from ModelConfig fields
                  alignment_layers: Optional[Dict[str, Any]] = None) -> AlignmentNetwork:
    # Parameters from your YAML mapped to the new CNN2P2 class
    # Need to add these to ModelConfig and YAML if not present
    cnn_params = {
        "in_channels": config_model.get("in_channels", 1),
        "output_dim": config_model.get("output_dim", 10),
        "conv_channels": config_model.get("conv_channels", [32, 64]),
        "kernel_sizes": config_model.get("kernel_sizes", [5, 5]),
        "strides": config_model.get("strides", [1, 1]),
        "paddings": config_model.get("paddings", [0, 0]), # Default to 0 if not specified, adjust based on common use
        "pool_kernel_size": config_model.get("pool_kernel_size", 2),
        "pool_stride": config_model.get("pool_stride", 2),
        "hidden_fc_dim": config_model.get("hidden_fc_dim", 128), # Your v2 num_hidden[1]
        "dropout_rate": config_model.get("dropout_rate", config_model.get("dropout", 0.5)),
        "example_input_hw": tuple(config_model.get("example_input_hw", [28,28])) # e.g. (28,28) for MNIST
    }
    # Determine input_dim for fc layer based on conv output and example_input_hw
    # This calculation is now inside CNN2P2 __init__

    base_model = CNN2P2(**cnn_params)
    
    # Default alignment layers for the new CNN2P2 structure
    # Names will be self.conv1.0 (Conv2d), self.conv2.0 (Conv2d)
    # And for FC layers within self.fc_layers (Sequential): self.fc_layers.1 (Linear), self.fc_layers.4 (Linear)
    default_alignment_layer_names = {
        "conv1.0": 0,       # First Conv2d inside self.conv1 Sequential
        "conv2.0": 1,       # First Conv2d inside self.conv2 Sequential
        "fc_layers.1": 2,   # First Linear layer in self.fc_layers
        "fc_layers.4": 3    # Second Linear layer in self.fc_layers (after Dropout, ReLU, Dropout)
    }
    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names

    cnn_mode = config_model.get("cnn_mode", "unfold")
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode)


@register_model("alexnet")
def create_alexnet(config_model: Dict,
                   alignment_layers: Optional[Dict[str, Any]] = None) -> AlignmentNetwork:
    dropout_rate = config_model.get("dropout_rate", config_model.get("dropout", 0.5))
    num_classes = config_model.get("output_dim", 1000) # AlexNet torchvision default is 1000

    base_model = alexnet(weights=None, progress=False, num_classes=num_classes) # Use num_classes
    
    # Modify dropout rates in AlexNet if specified
    # AlexNet has dropout layers named classifier.2 and classifier.5
    # The original AlexNet paper used 0.5 dropout.
    # torchvision.models.alexnet() already adds nn.Dropout(p=0.5) at these positions.
    # We can adjust them if dropout_rate is different from 0.5 and > 0.
    if dropout_rate > 0 and dropout_rate != 0.5:
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
                logger.info(f"Set dropout for {name} in AlexNet to {dropout_rate}")
    
    # Default alignment layers for AlexNet (key linear and conv layers)
    # Names from base_model.named_modules():
    # features.0, features.3, features.6, features.8, features.10 (Conv2d)
    # classifier.1, classifier.4, classifier.6 (Linear)
    default_alignment_layer_names = {
        "features.0": 0, # Conv1
        "features.3": 1, # Conv2
        "features.6": 2, # Conv3
        "features.8": 3, # Conv4
        "features.10": 4, # Conv5
        "classifier.1": 5, # FC1
        "classifier.4": 6, # FC2
        "classifier.6": 7  # Output FC
    }
    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names
    
    cnn_mode = config_model.get("cnn_mode", "unfold") # AlexNet is a CNN
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode)


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