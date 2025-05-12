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
def create_mlp(
    # Common params from ModelConfig
    output_dim: int,
    dropout_rate: float,
    # Specific params from MLPParamsConfig (via ModelConfig.mlp_params)
    input_dim: int,
    hidden_dims: List[int],
    activation: str,
    # Params for AlignmentNetwork wrapper
    alignment_layers: Optional[Dict[str, Any]] = None,
    cnn_mode_for_wrapper: str = "unfold", # cnn_mode from ModelConfig to be used by AlignmentNetwork
    extra_params: Optional[Dict[str, Any]] = None # from ModelConfig.extra_model_params
) -> AlignmentNetwork:
    
    mlp_constructor_params = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim, # Pass common output_dim to MLP class
        "dropout_rate": dropout_rate, # Pass common dropout_rate to MLP class
        "activation_type": activation
    }
    # Include any extra_params if provided and MLP class accepts them (or handle selectively)
    # For now, assuming MLP class takes these specific args.
    # If extra_params are for MLP class, they need to be merged into mlp_constructor_params.
    # If they are for something else, they are ignored here for MLP construction.
    if extra_params:
        # Example: if MLP had a `use_bias` param not in MLPParamsConfig but in extra_model_params
        # mlp_constructor_params.update({k:v for k,v in extra_params.items() if k in MLP.__init__.__code__.co_varnames})
        pass # No generic merging for now, rely on explicit params

    base_model = MLP(**mlp_constructor_params)
    
    default_alignment_layer_names = {}
    idx_for_alignment = 0
    for i, layer in enumerate(base_model.network):
        if isinstance(layer, nn.Linear):
            default_alignment_layer_names[f"network.{i}"] = idx_for_alignment 
            idx_for_alignment += 1

    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names
    
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode_for_wrapper)


@register_model("cnn2p2")
def create_cnn2p2(
    # Common params from ModelConfig
    output_dim: int,
    dropout_rate: float, # General dropout, CNN2P2 specific dropout is internal or via this
    # Specific params from CNN2P2ParamsConfig (via ModelConfig.cnn2p2_params)
    in_channels: int,
    conv_channels: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
    pool_kernel_size: int,
    pool_stride: int,
    hidden_fc_dim: int,
    example_input_hw: List[int],
    # Params for AlignmentNetwork wrapper
    alignment_layers: Optional[Dict[str, Any]] = None,
    cnn_mode_for_wrapper: str = "unfold", # cnn_mode from ModelConfig to be used by AlignmentNetwork
    extra_params: Optional[Dict[str, Any]] = None # from ModelConfig.extra_model_params
) -> AlignmentNetwork:
    
    cnn_constructor_params = {
        "in_channels": in_channels,
        "output_dim": output_dim, # Pass common output_dim
        "conv_channels": conv_channels,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "paddings": paddings,
        "pool_kernel_size": pool_kernel_size,
        "pool_stride": pool_stride,
        "hidden_fc_dim": hidden_fc_dim,
        "dropout_rate": dropout_rate, # Pass common dropout_rate, CNN2P2 class uses it for its FC dropout
        "example_input_hw": tuple(example_input_hw) # Ensure it's a tuple for CNN2P2 class
    }
    # Handle extra_params if CNN2P2 class supports them
    if extra_params: 
        pass # No generic merging for now, rely on explicit params

    base_model = CNN2P2(**cnn_constructor_params)
    
    default_alignment_layer_names = {
        "conv1.0": 0,       
        "conv2.0": 1,       
        "fc_layers.2": 2,   # Updated based on typical CNN2P2 sequential: Flatten, Dropout, Linear, ReLU, Dropout, Linear
                            # Index 2 for Linear after Dropout(1)
        "fc_layers.5": 3    # Index 5 for Linear after Dropout(4), ReLU(3)
    }
    # To be more robust, inspect base_model.fc_layers for Linear instances:
    # linear_fc_indices = [i for i, m in enumerate(base_model.fc_layers) if isinstance(m, nn.Linear)]
    # if len(linear_fc_indices) >= 1: default_alignment_layer_names[f"fc_layers.{linear_fc_indices[0]}"] = 2
    # if len(linear_fc_indices) >= 2: default_alignment_layer_names[f"fc_layers.{linear_fc_indices[1]}"] = 3
    
    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names

    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode_for_wrapper)


@register_model("alexnet") # This is for a custom/non-pretrained AlexNet if needed
def create_alexnet(
    # Common params from ModelConfig
    output_dim: int,
    dropout_rate: float, # General dropout for potential override
    # Params for AlignmentNetwork wrapper
    alignment_layers: Optional[Dict[str, Any]] = None,
    cnn_mode_for_wrapper: str = "unfold", 
    extra_params: Optional[Dict[str, Any]] = None
) -> AlignmentNetwork:
    # This constructor is for a basic AlexNet, likely non-pretrained torchvision one or custom.
    # If user wants pretrained, they should use model_name: "torchvision_alexnet" which is handled by registry.py directly.
    # Here, we assume extra_params might contain specific AlexNet constructor args if this isn't just torchvision.alexnet()
    
    # Determine if torchvision.models.alexnet specific params are in extra_params
    use_torchvision_default = True
    tv_alexnet_kwargs = {}
    if extra_params:
        # Example: if extra_params was {"torchvision_progress": False}
        if "progress" in extra_params: tv_alexnet_kwargs["progress"] = extra_params["progress"]
        # if any other specific args are passed for a custom AlexNet, this logic would change

    base_model = alexnet(weights=None, num_classes=output_dim, **tv_alexnet_kwargs) 
    
    if dropout_rate > 0 and dropout_rate != 0.5: # Default torchvision AlexNet has p=0.5
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
                logger.info(f"Set dropout for {name} in custom AlexNet to {dropout_rate}")
    
    default_alignment_layer_names = {
        "features.0": 0, "features.3": 1, "features.6": 2, "features.8": 3, "features.10": 4,
        "classifier.1": 5, "classifier.4": 6, "classifier.6": 7
    }
    final_alignment_layer_names = alignment_layers if alignment_layers is not None else default_alignment_layer_names
    
    return AlignmentNetwork(base_model=base_model, alignment_layer_names=final_alignment_layer_names, cnn_mode=cnn_mode_for_wrapper)


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