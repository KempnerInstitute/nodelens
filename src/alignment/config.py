"""
Configuration classes for alignment experiments.

This module defines dataclasses for experiment configuration, with support for
validation, loading from YAML, and type checking.
"""

import os
from os import PathLike
from dataclasses import dataclass, field
from typing import cast, List, Dict, Union, Type, TypeVar, Optional, Any
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

@dataclass
class BaseConfig:
    """Base class for all configuration classes."""
    
    @classmethod
    def load(cls, path: Union[str, PathLike]):
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Load the YAML file with OmegaConf
            conf = OmegaConf.load(str(path))
            
            # Create an instance of the class
            obj = cls()
            
            # Convert OmegaConf to regular dict
            config_dict = OmegaConf.to_container(conf)
            
            # Update the object with top-level values from the config
            for key, value in config_dict.items():
                if hasattr(obj, key):
                    current_val = getattr(obj, key)
                    
                    # For nested configurations, create new instances
                    if hasattr(current_val, '__class__') and issubclass(current_val.__class__, BaseConfig) and isinstance(value, dict):
                        # Create a new instance of the nested config
                        nested_config = current_val.__class__()
                        
                        # Update its fields
                        for k, v in value.items():
                            if hasattr(nested_config, k):
                                setattr(nested_config, k, v)
                        
                        # Set the nested config on the parent
                        setattr(obj, key, nested_config)
                    else:
                        # Direct assignment for simple values
                        setattr(obj, key, value)
            
            # Validate top-level config (nested validation happens inside)
            if hasattr(obj, "validate"):
                obj.validate()
                
            return obj
        except Exception as e:
            raise ValueError(f"Error loading config from {path}: {str(e)}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Create an instance of the class
            obj = cls()
            
            # Update the object with values from the dictionary
            for key, value in config_dict.items():
                if hasattr(obj, key):
                    current_val = getattr(obj, key)
                    
                    # For nested configurations, create new instances
                    if hasattr(current_val, '__class__') and issubclass(current_val.__class__, BaseConfig) and isinstance(value, dict):
                        # Create a new instance of the nested config
                        nested_config = current_val.__class__()
                        
                        # Update its fields
                        for k, v in value.items():
                            if hasattr(nested_config, k):
                                setattr(nested_config, k, v)
                        
                        # Set the nested config on the parent
                        setattr(obj, key, nested_config)
                    else:
                        # Direct assignment for simple values
                        setattr(obj, key, value)
            
            # Validate the configuration
            if hasattr(obj, "validate"):
                obj.validate()
                
            return obj
        except Exception as e:
            raise ValueError(f"Error creating config from dictionary: {str(e)}")
            
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for k, v in vars(self).items():
            if isinstance(v, BaseConfig):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key, returning default if not found.
        
        Args:
            key: Key to look up
            default: Default value to return if key not found
            
        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)


@dataclass
class ModelConfig(BaseConfig):
    """Neural network model configuration."""
    
    model_name: str = "MLP"
    # Common params
    dropout_rate: float = 0.0 # Used by MLP, CNN2P2, AlexNet (for overriding default)
    output_dim: int = 10      # Used by MLP, CNN2P2, AlexNet (as num_classes)
    alignment_layers: Optional[Dict[str, Any]] = None # Using Any for value type, can be int or other spec

    # MLP specific
    input_dim: Optional[int] = None # e.g., 784 for MNIST with MLP
    hidden_dims: Optional[List[int]] = field(default_factory=lambda: [100, 100, 50])
    activation: Optional[str] = "relu" # For MLP: relu, tanh, sigmoid, identity

    # CNN2P2 specific
    in_channels: Optional[int] = None # e.g., 1 for MNIST, 3 for CIFAR
    conv_channels: Optional[List[int]] = field(default_factory=lambda: [32, 64])
    kernel_sizes: Optional[List[int]] = field(default_factory=lambda: [5, 5])
    strides: Optional[List[int]] = field(default_factory=lambda: [1, 1])
    paddings: Optional[List[int]] = field(default_factory=lambda: [0, 0]) # Default to 0, adjust based on kernel/stride
    pool_kernel_size: Optional[int] = 2
    pool_stride: Optional[int] = 2
    hidden_fc_dim: Optional[int] = 128
    example_input_hw: Optional[List[int]] = field(default_factory=lambda: [28,28]) # For dynamic FC input calculation

    # AlexNet specific (uses output_dim as num_classes, dropout_rate to override internal dropout)
    # No extra specific params here needed beyond common ones, unless more customization is desired.

    # Field for other model-specific kwargs not explicitly defined
    extra_model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Backward compatibility for "dropout" if present in config
        if "dropout" in self.extra_model_params and self.dropout_rate == 0.0:
            self.dropout_rate = self.extra_model_params.pop("dropout")
        # Ensure example_input_hw is a tuple for CNN2P2
        if self.example_input_hw is not None and not isinstance(self.example_input_hw, tuple):
            self.example_input_hw = tuple(self.example_input_hw)

    def validate(self) -> bool:
        """Validate model configuration."""
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(f"Dropout rate must be between 0 and 1, got {self.dropout_rate}")
            
        # Basic model name check (can be expanded)
        known_models = ["MLP", "CNN2P2", "AlexNet"]
        if self.model_name not in known_models:
            logger.warning(f"Unusual model name: {self.model_name}. Ensure a corresponding create_{self.model_name.lower()} function exists.")
        
        if self.model_name == "MLP":
            if self.input_dim is None:
                logger.warning("MLP input_dim not set, model might fail if data isn't auto-flattened to a known size.")
            if not self.hidden_dims:
                logger.warning("MLP hidden_dims not set, will be a linear model if empty.")
            valid_activations = ["relu", "tanh", "sigmoid", "identity"]
            if self.activation.lower() not in valid_activations:
                raise ValueError(f"Invalid MLP activation: {self.activation}. Valid: {valid_activations}")

        if self.model_name == "CNN2P2":
            if self.in_channels is None:
                raise ValueError("CNN2P2 in_channels must be set.")
            for param_name, param_list in [
                ("conv_channels", self.conv_channels),
                ("kernel_sizes", self.kernel_sizes),
                ("strides", self.strides),
                ("paddings", self.paddings)
            ]:
                if not (isinstance(param_list, list) and len(param_list) == 2):
                    raise ValueError(f"CNN2P2 {param_name} must be a list of 2 elements, got {param_list}")
            if not (isinstance(self.example_input_hw, (list, tuple)) and len(self.example_input_hw) == 2):
                 raise ValueError(f"CNN2P2 example_input_hw must be a list/tuple of 2 elements, got {self.example_input_hw}")

        return True


@dataclass
class DatasetConfig(BaseConfig):
    """Dataset configuration."""
    
    dataset_name: str = "MNIST"
    data_path: Optional[str] = None
    batch_size: int = 128
    
    def validate(self) -> bool:
        """Validate dataset configuration."""
        if self.data_path is not None and not os.path.exists(self.data_path):
            import warnings
            warnings.warn(f"Dataset path does not exist: {self.data_path}")
            
        if self.dataset_name not in ["MNIST", "CIFAR10", "CIFAR100", "ImageNet"]:
            # This is just a warning, not an error, as custom datasets might be used
            import warnings
            warnings.warn(f"Unusual dataset name: {self.dataset_name}")
            
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
            
        return True


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""
    
    epochs: int = 10
    replicates: int = 1
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    def validate(self) -> bool:
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {self.epochs}")
            
        if self.replicates <= 0:
            raise ValueError(f"Replicates must be positive, got {self.replicates}")
            
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
            
        if self.weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {self.weight_decay}")
            
        if self.optimizer not in ["Adam", "SGD", "RMSprop", "AdamW"]:
            # This is just a warning, not an error, as custom optimizers might be used
            import warnings
            warnings.warn(f"Unusual optimizer name: {self.optimizer}")
            
        return True


@dataclass
class AlignmentConfig(BaseConfig):
    """Alignment measurement configuration."""
    
    metric: str = "RQ"
    run_progressive: bool = True
    run_eigenvector: bool = True
    dropout_min: float = 0.0
    dropout_max: float = 0.95
    dropout_steps: int = 40
    scale_by_norm: bool = False
    
    def validate(self) -> bool:
        """Validate alignment configuration."""
        valid_metrics = ["RQ", "NullSpace", "MI", "weight_similarity", "redundancy", "delta_alignment"]
        
        if self.metric not in valid_metrics:
            # This is just a warning, not an error, as custom methods might be added
            import warnings
            warnings.warn(f"Unusual alignment metric: {self.metric}")
            
        if self.dropout_min < 0.0 or self.dropout_min > 1.0:
            raise ValueError(f"Dropout min must be between 0 and 1, got {self.dropout_min}")
            
        if self.dropout_max < 0.0 or self.dropout_max > 1.0:
            raise ValueError(f"Dropout max must be between 0 and 1, got {self.dropout_max}")
            
        if self.dropout_min >= self.dropout_max:
            raise ValueError(f"Dropout min must be less than dropout max, got {self.dropout_min} >= {self.dropout_max}")
            
        if self.dropout_steps <= 0:
            raise ValueError(f"Dropout steps must be positive, got {self.dropout_steps}")
            
        return True


@dataclass
class CheckpointingConfig(BaseConfig):
    """Checkpointing configuration."""
    
    save_checkpoints: bool = False
    checkpoint_frequency: int = 1
    use_wandb: bool = False
    load_checkpoint: bool = False
    
    def validate(self) -> bool:
        """Validate checkpointing configuration."""
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"Checkpoint frequency must be positive, got {self.checkpoint_frequency}")
            
        return True


@dataclass
class ExtraConfig(BaseConfig):
    """Extra configuration parameters."""
    
    dropout_mode: str = "scaled"
    dropout_pruning_mode: str = "layer_wise"
    exclude_classification_layer: bool = True
    cnn_mode: str = "unfold"
    training_method: str = "auto"  # Options: "auto", "sequential", "tensorized", "fully_tensorized"
    
    def validate(self) -> bool:
        """Validate extra configuration."""
        valid_pruning_modes = [
            # Original modes
            "global", "per_layer_combined", "per_layer_independent",
            # New names for backward compatibility
            "global_joint", "layer_wise", "layer_isolated",
            # New progressive pruning mode
            "cascading_layer"
        ]
        
        if self.dropout_pruning_mode not in valid_pruning_modes:
            raise ValueError(f"Invalid dropout pruning mode: {self.dropout_pruning_mode}")
            
        if self.dropout_mode not in ["scaled", "unscaled"]:
            raise ValueError(f"Invalid dropout mode: {self.dropout_mode}")
            
        if self.cnn_mode not in ["unfold", "patchwise", "batch_patch_combined"]:
            raise ValueError(f"Invalid CNN mode: {self.cnn_mode}")
            
        valid_training_methods = ["auto", "sequential", "tensorized", "fully_tensorized"]
        if self.training_method not in valid_training_methods:
            raise ValueError(f"Invalid training method: {self.training_method}. Valid options are: {valid_training_methods}")
            
        return True


@dataclass
class ExperimentConfig(BaseConfig):
    """Main experiment configuration."""
    
    # Experiment metadata
    experiment_type: str = "alignment"
    results_path: str = "results"
    use_timestamp: bool = True
    
    # Core execution options
    device: Optional[str] = None
    no_save: bool = False
    just_plot: bool = False
    save_networks: bool = False
    show_all: bool = False
    timestamp: Optional[str] = None
    debug_mode: bool = False  # Enable debug output
    
    # Nested configurations
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    extra: ExtraConfig = field(default_factory=ExtraConfig)
    
    def validate(self) -> bool:
        """
        Validate all configuration components.
        
        Returns:
            True if all components are valid
            
        Raises:
            ValueError: If any component is invalid
        """
        # Validate device
        if self.device is not None:
            import torch
            try:
                torch.device(self.device)
            except Exception as e:
                raise ValueError(f"Invalid device: {self.device}, {str(e)}")
                
        # Validate results path
        if self.results_path is not None and self.results_path != "results":
            results_parent = os.path.dirname(self.results_path)
            if results_parent and not os.path.exists(results_parent):
                try:
                    os.makedirs(results_parent, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Cannot create results directory: {str(e)}")
                    
        # Ensure nested configs are proper instances
        if not isinstance(self.dataset, DatasetConfig):
            self.dataset = DatasetConfig()
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig()
        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig()
        if not isinstance(self.alignment, AlignmentConfig):
            self.alignment = AlignmentConfig()
        if not isinstance(self.checkpointing, CheckpointingConfig):
            self.checkpointing = CheckpointingConfig()
        if not isinstance(self.extra, ExtraConfig):
            self.extra = ExtraConfig()
            
        # Validate nested configs (manually to handle exceptions)
        try:
            self.dataset.validate()
            self.model.validate()
            self.training.validate()
            self.alignment.validate()
            self.checkpointing.validate()
            self.extra.validate()
        except Exception as e:
            raise ValueError(f"Nested config validation error: {str(e)}")
        
        return True

# For backward compatibility
@dataclass
class ExperimentArgs:
    """Experiment arguments for CLI."""
    
    dataclass_fields = []

    exp_name: str
    name: str
    device: str = "cpu"
    
    def validate(self) -> bool:
        """Validate experiment arguments."""
        import torch
        try:
            torch.device(self.device)
        except Exception as e:
            raise ValueError(f"Invalid device: {self.device}, {str(e)}")
            
        return True


# For backward compatibility
@dataclass
class ExtraArgs:
    """Extra arguments for various experiments."""
    
    aggregate_alignment: bool = False
    num_drops: int = 9
    dropout_pruning_mode: str = "global"
    dropout_mode: str = "scaled"
    exclude_classification_layer: bool = False
    
    def validate(self) -> bool:
        """Validate extra arguments."""
        if self.num_drops <= 0:
            raise ValueError(f"Number of drops must be positive, got {self.num_drops}")
            
        valid_pruning_modes = [
            # Original modes
            "global", "per_layer_combined", "per_layer_independent",
            # New names for backward compatibility
            "global_joint", "layer_wise", "layer_isolated",
            # New progressive pruning mode
            "cascading_layer"
        ]
        
        if self.dropout_pruning_mode not in valid_pruning_modes:
            raise ValueError(f"Invalid dropout pruning mode: {self.dropout_pruning_mode}")
            
        if self.dropout_mode not in ["scaled", "unscaled"]:
            raise ValueError(f"Invalid dropout mode: {self.dropout_mode}")
            
        return True

# For backward compatibility
Config = ExperimentConfig

def load_config(path: Union[str, PathLike]) -> ExperimentConfig:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    return ExperimentConfig.load(path)