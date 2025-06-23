"""
Standard model architectures for alignment experiments.

This module provides implementations of common architectures used in alignment
studies, matching the functionality from the old alignment codebase.
"""

import logging
from typing import List, Optional, Dict, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden layers.
    
    This implementation matches the old alignment.models.models.MLP functionality.
    
    Args:
        input_dim: Dimension of input features (default: 784 for MNIST)
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes
        dropout_rate: Dropout probability between layers
        activation_type: Type of activation function ('relu', 'tanh', 'sigmoid', 'identity')
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [100, 100, 50],
        output_dim: int = 10,
        dropout_rate: float = 0.5,
        activation_type: str = "relu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Select activation function
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "identity": nn.Identity()
        }
        
        if activation_type.lower() not in activation_map:
            logger.warning(f"Unknown activation_type '{activation_type}'. Defaulting to ReLU.")
            activation_fn = nn.ReLU()
        else:
            activation_fn = activation_map[activation_type.lower()]
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        # First hidden layer
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
        if dropout_rate > 0.0 and hidden_dims:
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
        # Flatten input if needed
        if x.dim() > 2:
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            
            # Check dimension compatibility
            if x.size(1) != self.input_dim:
                logger.warning(
                    f"Input dimension mismatch. Expected {self.input_dim}, "
                    f"got {x.size(1)}. Attempting to adjust."
                )
                # Common adjustments for standard datasets
                if self.input_dim == 784:  # MNIST
                    x = x[:, :784]
                elif self.input_dim == 3072:  # CIFAR
                    x = x[:, :3072]
        
        return self.network(x)


class CNN2P2(nn.Module):
    """
    Convolutional Neural Network with 2 convolutional layers and 2 pooling layers.
    
    This implementation matches the old alignment.models.models.CNN2P2 functionality.
    
    Architecture:
        Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> FC1 -> ReLU -> FC2
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        output_dim: Number of output classes
        conv_channels: Output channels for each conv layer
        kernel_sizes: Kernel sizes for conv layers
        strides: Strides for conv layers
        paddings: Paddings for conv layers
        pool_kernel_size: Kernel size for pooling layers
        pool_stride: Stride for pooling layers
        hidden_fc_dim: Dimension of hidden FC layer
        dropout_rate: Dropout probability
        example_input_hw: Example input height and width for calculating FC input size
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
        super().__init__()
        
        if not (len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings) == 2):
            raise ValueError("conv_channels, kernel_sizes, strides, paddings must all be lists of 2 elements.")
        
        self.in_channels = in_channels
        current_channels = in_channels
        h, w = example_input_hw
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(current_channels, conv_channels[0], kernel_sizes[0], strides[0], paddings[0]),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        )
        current_channels = conv_channels[0]
        # Calculate size after conv1
        h = (h - kernel_sizes[0] + 2 * paddings[0]) // strides[0] + 1
        h = (h - pool_kernel_size) // pool_stride + 1
        w = (w - kernel_sizes[0] + 2 * paddings[0]) // strides[0] + 1
        w = (w - pool_kernel_size) // pool_stride + 1
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(current_channels, conv_channels[1], kernel_sizes[1], strides[1], paddings[1]),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, stride=pool_stride)
        )
        current_channels = conv_channels[1]
        # Calculate size after conv2
        h = (h - kernel_sizes[1] + 2 * paddings[1]) // strides[1] + 1
        h = (h - pool_kernel_size) // pool_stride + 1
        w = (w - kernel_sizes[1] + 2 * paddings[1]) // strides[1] + 1
        w = (w - pool_kernel_size) // pool_stride + 1
        
        flattened_dim = current_channels * h * w
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(flattened_dim, hidden_fc_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_fc_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Handle flattened input
        if x.dim() == 2:
            batch_size = x.size(0)
            # Infer image dimensions
            total_pixels = x.size(1) // self.in_channels
            side_length = int(torch.sqrt(torch.tensor(total_pixels, dtype=torch.float)))
            x = x.view(batch_size, self.in_channels, side_length, side_length)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc_layers(x)
        return x


class SimpleConvNet(nn.Module):
    """
    A simple convolutional network for quick experiments.
    
    This is a minimal CNN with configurable depth, useful for testing.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        hidden_channels: List of hidden channel sizes for conv layers
        fc_hidden: Hidden dimension for FC layer
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_channels: List[int] = [32, 64, 128],
        fc_hidden: int = 256,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        # Build convolutional layers
        conv_layers = []
        current_channels = in_channels
        
        for hidden_ch in hidden_channels:
            conv_layers.extend([
                nn.Conv2d(current_channels, hidden_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ])
            current_channels = hidden_ch
        
        self.features = nn.Sequential(*conv_layers)
        
        # Calculate feature size (assuming input is 32x32 like CIFAR)
        # Each MaxPool2d reduces size by factor of 2
        num_pools = len(hidden_channels)
        final_size = 32 // (2 ** num_pools)
        flattened_size = current_channels * final_size * final_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetBlock(nn.Module):
    """
    Basic ResNet block for building custom ResNet architectures.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution
        downsample: Optional downsampling layer
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# Dataset-specific parameter configurations (matching old codebase)
DATASET_PARAMETERS = {
    "mlp": {
        "mnist": {"input_dim": 784, "output_dim": 10},
        "cifar10": {"input_dim": 3072, "output_dim": 10},
        "cifar100": {"input_dim": 3072, "output_dim": 100},
    },
    "cnn2p2": {
        "mnist": {"in_channels": 1, "output_dim": 10, "example_input_hw": (28, 28)},
        "cifar10": {"in_channels": 3, "output_dim": 10, "example_input_hw": (32, 32)},
        "cifar100": {"in_channels": 3, "output_dim": 100, "example_input_hw": (32, 32)},
        "imagenet": {"in_channels": 3, "output_dim": 1000, "example_input_hw": (224, 224)},
    },
}


def create_model(
    model_name: str,
    dataset_name: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Create a model with dataset-specific parameters.
    
    This function provides compatibility with the old codebase's model creation.
    
    Args:
        model_name: Name of the model ('mlp', 'cnn2p2', 'simple_conv')
        dataset_name: Name of the dataset for automatic parameter configuration
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Instantiated model
        
    Example:
        >>> model = create_model('mlp', 'mnist', hidden_dims=[300, 200])
        >>> model = create_model('cnn2p2', 'cifar10', dropout_rate=0.3)
    """
    model_name = model_name.lower()
    
    # Get dataset-specific parameters if provided
    if dataset_name and model_name in DATASET_PARAMETERS:
        dataset_name = dataset_name.lower()
        if dataset_name in DATASET_PARAMETERS[model_name]:
            default_params = DATASET_PARAMETERS[model_name][dataset_name].copy()
            default_params.update(kwargs)
            kwargs = default_params
    
    # Create model
    if model_name == "mlp":
        return MLP(**kwargs)
    elif model_name == "cnn2p2":
        return CNN2P2(**kwargs)
    elif model_name == "simple_conv" or model_name == "simpleconvnet":
        return SimpleConvNet(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Available models: mlp, cnn2p2, simple_conv")


# For backward compatibility
def create_mlp(**kwargs) -> MLP:
    """Create an MLP model. Provided for compatibility with old codebase."""
    return MLP(**kwargs)


def create_cnn2p2(**kwargs) -> CNN2P2:
    """Create a CNN2P2 model. Provided for compatibility with old codebase."""
    return CNN2P2(**kwargs) 