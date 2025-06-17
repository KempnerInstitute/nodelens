"""
Configuration classes for alignment experiments.

This module defines dataclasses for experiment configuration, with support for
validation, loading from YAML, and type checking.
"""

import os
from os import PathLike
from dataclasses import dataclass, field
from typing import cast, List, Dict, Union, Type, TypeVar, Optional, Any
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.errors import OmegaConfBaseException
import logging
import warnings
import torch
from alignment_refac1.metrics import ALIGNMENT_METRICS_REGISTRY

config_logger = logging.getLogger(__name__)
T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base class for all configuration classes."""
    
    @classmethod
    def load(cls: Type[T], path: Union[str, PathLike]) -> T:
        """
        Load configuration from YAML file using OmegaConf structured capabilities.
        """
        try:
            yaml_conf = OmegaConf.load(str(path))
            # Create a schema from the dataclass itself. This will include defaults.
            schema = OmegaConf.structured(cls) 
            # Merge the schema with the loaded YAML. YAML values override schema defaults.
            # OmegaConf handles type validation and conversion based on schema type hints.
            merged_conf = OmegaConf.merge(schema, yaml_conf)
                    
            # Resolve any variable interpolations (e.g., ${oc.env:SOME_VAR})
            OmegaConf.resolve(merged_conf)
            
            # Convert the merged OmegaConf object to an instance of the dataclass.
            # This should recursively instantiate nested dataclasses correctly.
            obj: T = OmegaConf.to_object(merged_conf)
            
            if not isinstance(obj, cls):
                # This is unexpected if OmegaConf.to_object works correctly with a structured schema.
                # It might indicate a very complex type hint that OmegaConf didn't fully resolve
                # to the exact cls type, or an issue in older OmegaConf versions.
                config_logger.critical(
                    f"OmegaConf.to_object did not return an instance of {cls.__name__}. "
                    f"Got {type(obj)}. Configuration loading might be incomplete or incorrect."
                )
                # Depending on strictness, one might raise an error here.
                # For now, we proceed and rely on subsequent validation.

            # Call custom validation method if defined on the class.
            # __post_init__ of the dataclass (and nested ones) should have run during OmegaConf.to_object.
            if hasattr(obj, "validate") and callable(getattr(obj, "validate")):
                obj.validate()
                
            return obj
        except OmegaConfBaseException as e:
            raise ValueError(f"Error processing config from {path} with OmegaConf: {str(e)}") from e
        except Exception as e: # Catch other errors (e.g., from our custom validate)
            raise ValueError(f"Error loading or validating config from {path}: {str(e)}") from e
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create a configuration from a dictionary using OmegaConf structured capabilities.
        """
        try:
            # Create an OmegaConf object from the dictionary
            # This allows OmegaConf to then merge it with a typed schema.
            dict_conf = OmegaConf.create(config_dict)

            # Create a schema from the dataclass itself.
            schema = OmegaConf.structured(cls)

            # Merge schema (defaults) with the dict_conf.
            merged_conf = OmegaConf.merge(schema, dict_conf)
            
            OmegaConf.resolve(merged_conf)

            obj: T = OmegaConf.to_object(merged_conf)

            if not isinstance(obj, cls):
                config_logger.critical(
                    f"OmegaConf.to_object did not return an instance of {cls.__name__} from dict. "
                    f"Got {type(obj)}. Configuration loading might be incomplete or incorrect."
                )

            if hasattr(obj, "validate") and callable(getattr(obj, "validate")):
                obj.validate()
                
            return obj
        except OmegaConfBaseException as e:
            raise ValueError(f"Error creating config from dictionary with OmegaConf: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Error creating or validating config from dictionary: {str(e)}") from e
            
    def validate(self) -> bool:
        # This base validate can be overridden by subclasses.
        # For simple dataclasses without cross-field validation, it might not be needed
        # if OmegaConf handles type validation during structured loading.
        return True

    def to_dict(self) -> Dict[str, Any]:
        try:
            # Convert self (which is a dataclass instance) to an OmegaConf config,
            # then to a primitive container (dict, list, etc.).
            # resolve=True ensures any interpolations are resolved.
            return OmegaConf.to_container(OmegaConf.create(self), resolve=True)
        except Exception as e:
            config_logger.warning(f"Error converting config to dict using OmegaConf: {e}. Falling back to vars().")
        result = {}
        for k, v in vars(self).items():
            if isinstance(v, BaseConfig):
                    result[k] = v.to_dict() # Recursive call
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# --- NEW: Nested Model Parameter Dataclasses ---
@dataclass
class MLPParamsConfig(BaseConfig):
    input_dim: Optional[int] = 784      # e.g., 784 for MNIST
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"          # Options: "relu", "tanh", "sigmoid", "identity"

    def validate(self) -> bool:
        if self.input_dim is None or self.input_dim <= 0:
            # Allow None if data is auto-flattened and dim inferred, but explicit is better
            config_logger.warning("MLPParamsConfig input_dim not set or invalid, model might fail.")
        if not self.hidden_dims:
            config_logger.warning("MLPParamsConfig hidden_dims not set, will be a linear model if empty.")
        valid_activations = ["relu", "tanh", "sigmoid", "identity"]
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"Invalid MLP activation: {self.activation}. Valid: {valid_activations}")
        return True

@dataclass
class CNN2P2ParamsConfig(BaseConfig):
    in_channels: int = 1                 # e.g., 1 for MNIST, 3 for CIFAR
    conv_channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_sizes: List[int] = field(default_factory=lambda: [5, 5])
    strides: List[int] = field(default_factory=lambda: [1, 1])
    paddings: List[int] = field(default_factory=lambda: [0, 0]) 
    pool_kernel_size: int = 2
    pool_stride: int = 2
    hidden_fc_dim: int = 128
    example_input_hw: List[int] = field(default_factory=lambda: [28,28])

    def validate(self) -> bool:
        for param_name, param_list in [
            ("conv_channels", self.conv_channels),
            ("kernel_sizes", self.kernel_sizes),
            ("strides", self.strides),
            ("paddings", self.paddings)
        ]:
            if not (isinstance(param_list, list) and len(param_list) == 2):
                raise ValueError(f"CNN2P2ParamsConfig {param_name} must be a list of 2 elements, got {param_list}")
        if not (isinstance(self.example_input_hw, (list, tuple)) and len(self.example_input_hw) == 2):
                raise ValueError(f"CNN2P2ParamsConfig example_input_hw must be a list/tuple of 2 elements, got {self.example_input_hw}")
        return True

@dataclass
class ExternalModelParamsConfig(BaseConfig):
    source: Optional[str] = None  # e.g., "torchvision", "huggingface_transformers"
    name_or_path: Optional[str] = None # e.g., "resnet18", "bert-base-uncased"
    pretrained: bool = True
    freeze_feature_extractor: bool = False

    def validate(self) -> bool:
        if not self.source:
            raise ValueError("ExternalModelParamsConfig: 'source' is required (e.g., 'torchvision').")
        if not self.name_or_path:
            raise ValueError("ExternalModelParamsConfig: 'name_or_path' is required.")
        valid_sources = ["torchvision", "huggingface_transformers"]
        if self.source not in valid_sources:
            raise ValueError(f"Invalid external_model_source: '{self.source}'. Valid are: {valid_sources}")
        return True
# --- End NEW: Nested Model Parameter Dataclasses ---

@dataclass
class ModelConfig(BaseConfig):
    """Neural network model configuration."""
    
    model_name: str = "MLP" # Main model type identifier
    
    # --- Common Parameters (apply to most models or the AlignmentNetwork wrapper) ---
    output_dim: int = 10      # Number of output classes, typically for the final layer.
    dropout_rate: float = 0.0 # A general dropout rate. Internal models might use this for their nn.Dropout layers.
                              # For external models, this is usually not directly applied unless model supports it.
    cnn_mode: Optional[str] = "unfold" # For AlignmentNetwork wrapper: "unfold", "patchwise", "batch_patch_combined".
                                     # If None, AlignmentNetwork defaults to "unfold".
    alignment_layers: Optional[Dict[str, Any]] = None # Specifies layers for AlignmentNetwork analysis.
    
    # --- Model-Specific Parameter Blocks ---
    # Only the block corresponding to model_name (or inferred for external) should be populated in YAML.
    mlp_params: Optional[MLPParamsConfig] = None
    cnn2p2_params: Optional[CNN2P2ParamsConfig] = None
    external_params: Optional[ExternalModelParamsConfig] = None
    
    # --- Deprecated/Legacy fields (will be removed or ignored if new structure is used) ---
    # These are now part of the nested configs above or external_params.
    input_dim: Optional[int] = field(default=None, metadata={"deprecated": True})
    hidden_dims: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    activation: Optional[str] = field(default=None, metadata={"deprecated": True})
    in_channels: Optional[int] = field(default=None, metadata={"deprecated": True})
    conv_channels: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    kernel_sizes: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    strides: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    paddings: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    pool_kernel_size: Optional[int] = field(default=None, metadata={"deprecated": True})
    pool_stride: Optional[int] = field(default=None, metadata={"deprecated": True})
    hidden_fc_dim: Optional[int] = field(default=None, metadata={"deprecated": True})
    example_input_hw: Optional[List[int]] = field(default=None, metadata={"deprecated": True})
    external_model_source: Optional[str] = field(default=None, metadata={"deprecated": True})
    external_model_name_or_path: Optional[str] = field(default=None, metadata={"deprecated": True})
    pretrained: Optional[bool] = field(default=None, metadata={"deprecated": True}) # Note: external_params.pretrained is bool=True
    freeze_feature_extractor: Optional[bool] = field(default=None, metadata={"deprecated": True})
    # --- End Deprecated --- 

    extra_model_params: Dict[str, Any] = field(default_factory=dict) # For ad-hoc params to internal constructors.

    def __post_init__(self):
        # Logic to populate specific param blocks if top-level ones are still used (for backward compatibility)
        # Or, ideally, enforce that only one specific param block is used.
        # For now, we primarily expect the YAML to use the nested structure.
        
        # Infer external_params if model_name suggests it and external_params is not explicitly set
        if self.model_name and not self.external_params:
            if self.model_name.lower().startswith("torchvision_") or self.model_name.lower().startswith("hf_"):
                source = "torchvision" if self.model_name.lower().startswith("torchvision_") else "huggingface_transformers"
                name_or_path = self.model_name.split("_", 1)[1] if "_" in self.model_name else ""
                if name_or_path:
                    self.external_params = ExternalModelParamsConfig(source=source, name_or_path=name_or_path)
                    config_logger.info(f"Inferred external_params for {self.model_name}: source={source}, name_or_path={name_or_path}")
                else:
                    config_logger.warning(f"Model name '{self.model_name}' suggests an external model, but could not infer name_or_path. Please set external_params explicitly.")
            elif self.model_name.lower() == "external" and (self.external_model_source and self.external_model_name_or_path):
                 # Handle legacy top-level external fields if model_name="external"
                 self.external_params = ExternalModelParamsConfig(
                     source=self.external_model_source,
                     name_or_path=self.external_model_name_or_path,
                     pretrained=self.pretrained if self.pretrained is not None else True,
                     freeze_feature_extractor=self.freeze_feature_extractor if self.freeze_feature_extractor is not None else False
                 )
                 config_logger.info(f"Populated external_params from legacy top-level fields for model_name='external'.")

    def validate(self) -> bool:
        """Validate model configuration."""
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError(f"Dropout rate must be between 0 and 1, got {self.dropout_rate}")
            
        model_name_lower = self.model_name.lower()
        known_internal_models = ["mlp", "cnn2p2", "alexnet"] # AlexNet is now external via torchvision
        is_external = model_name_lower.startswith("torchvision_") or model_name_lower.startswith("hf_") or model_name_lower == "external" or (self.external_params is not None)

        if not is_external and model_name_lower not in known_internal_models:
            # This warning is if it's not a known internal name AND not clearly an external model config
            config_logger.warning(f"Unusual model_name: {self.model_name}. Ensure it's a registered internal model or configure external_params correctly.")

        # Validate specific param blocks based on model_name
        if model_name_lower == "mlp":
            if not self.mlp_params: raise ValueError("model_name is MLP, but mlp_params block is missing.")
            self.mlp_params.validate()
        elif model_name_lower == "cnn2p2":
            if not self.cnn2p2_params: raise ValueError("model_name is CNN2P2, but cnn2p2_params block is missing.")
            self.cnn2p2_params.validate()
        elif is_external:
            if not self.external_params: raise ValueError(f"model_name '{self.model_name}' indicates an external model, but external_params block is missing or could not be inferred.")
            self.external_params.validate()
        
        # Validate cnn_mode if set (this is for AlignmentNetwork wrapper)
        if self.cnn_mode is not None:
            valid_cnn_modes_wrapper = ["unfold", "patchwise", "batch_patch_combined"]
            # Note: "filter_patch_summary", "filter_specific_covariance_rq" are more for metric computation interpretation
            # rather than AlignmentNetwork preprocessing mode directly.
            if self.cnn_mode not in valid_cnn_modes_wrapper:
                config_logger.warning(f"ModelConfig.cnn_mode (for AlignmentNetwork wrapper) '{self.cnn_mode}' provided. Ensure it is valid. Known wrapper modes: {valid_cnn_modes_wrapper}")

        return True


@dataclass
class DatasetConfig(BaseConfig):
    """Dataset configuration."""
    
    dataset_name: str = "MNIST"
    data_path: Optional[str] = None
    batch_size: int = 128
    num_workers: int = 4  # ADDED: num_workers field, default to 4 as in YAML
    
    def validate(self) -> bool:
        """Validate dataset configuration."""
        if self.data_path is not None and not os.path.exists(self.data_path):
            warnings.warn(f"Dataset path does not exist: {self.data_path}")
            
        if self.dataset_name not in ["MNIST", "CIFAR10", "CIFAR100", "ImageNet"]:
            # This is just a warning, not an error, as custom datasets might be used
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
    loss: str = "cross_entropy" # From your YAML
    momentum: float = 0.9 # From your YAML
    
    # MOVED from ExtraConfig
    training_method: str = "auto"  # Options: "auto", "sequential", "fully_tensorized"
    train_before_dropout: bool = True  # ADDED: Controls initial training before dropout experiments
    
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
            warnings.warn(f"Unusual optimizer name: {self.optimizer}")
            
        valid_training_methods = ["auto", "sequential", "fully_tensorized"]
        if self.training_method not in valid_training_methods:
            raise ValueError(f"Invalid training method: {self.training_method}. Valid options: {valid_training_methods}")
            
        return True


@dataclass
class PruningConfig(BaseConfig):
    """Configuration specific to pruning experiments."""
    dropout_min: float = 0.0
    dropout_max: float = 0.9
    dropout_steps: int = 40
    dropout_mode: str = "scaled"
    dropout_pruning_mode: str = "global_joint"
    exclude_classification_layer: bool = True
    use_multi_strategy_dropout: bool = True
    num_batches_for_scores: Optional[int] = 5  # MODIFIED: Allow None for all batches, default to 5

    def validate(self) -> bool:
        if not (0.0 <= self.dropout_min <= 1.0):
            raise ValueError(f"Pruning dropout_min must be between 0 and 1, got {self.dropout_min}")
        if not (0.0 <= self.dropout_max <= 1.0):
            raise ValueError(f"Pruning dropout_max must be between 0 and 1, got {self.dropout_max}")
        if self.dropout_min > self.dropout_max:
            raise ValueError(f"Pruning dropout_min must be less than or equal to dropout_max, got min={self.dropout_min}, max={self.dropout_max}")
        if self.dropout_steps <= 0 and not (self.dropout_min == self.dropout_max):
            if self.dropout_min == self.dropout_max and self.dropout_steps ==0: pass
            else: raise ValueError(f"Pruning dropout_steps must be positive, got {self.dropout_steps}")
        
        current_valid_pruning_modes = ["global_joint", "layer_wise", "layer_isolated", "cascading_layer"]
        if self.dropout_pruning_mode not in current_valid_pruning_modes:
            raise ValueError(f"Invalid dropout_pruning_mode: '{self.dropout_pruning_mode}'. Valid are: {current_valid_pruning_modes}")

        if self.dropout_mode not in ["scaled", "unscaled"]:
            raise ValueError(f"Invalid dropout_mode: '{self.dropout_mode}'. Valid are: ['scaled', 'unscaled']")
        if self.num_batches_for_scores is not None and self.num_batches_for_scores <= 0:
            raise ValueError(f"PruningConfig num_batches_for_scores must be positive if specified, got {self.num_batches_for_scores}")
        return True


@dataclass
class MetricTrackerConfig(BaseConfig):
    name: str = "RQ"  
    num_batches: Optional[int] = 5 # MODIFIED: Allow None for all batches, default to 5

    def validate(self) -> bool:
        if not self.name:
            raise ValueError("MetricTrackerConfig 'name' cannot be empty.")
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError(f"MetricTrackerConfig 'num_batches' must be positive if specified, got {self.num_batches}")
        return True


@dataclass
class CallbackSettings(BaseConfig):
    alignment_metrics: List[MetricTrackerConfig] = field(default_factory=list)

    def __post_init__(self):
        # Ensure elements of alignment_metrics are MetricTrackerConfig instances
        if self.alignment_metrics and isinstance(self.alignment_metrics, list):
            self.alignment_metrics = [
                MetricTrackerConfig(**item) if isinstance(item, dict) else item
                for item in self.alignment_metrics
            ]
            for item in self.alignment_metrics:
                if not isinstance(item, MetricTrackerConfig):
                    raise ValueError(f"Invalid item in alignment_metrics: {item}. Expected MetricTrackerConfig or dict.")

    def validate(self) -> bool:
        for item_config in self.alignment_metrics:
            item_config.validate()
        return True


@dataclass
class AlignmentConfig(BaseConfig):
    """Configuration for alignment metric calculation."""
    metric: List[str] = field(default_factory=lambda: ["RQ"])
    scale_by_norm: bool = False
    cnn_mode: str = "unfold"
    cnn_rq_aggregation_op: str = "mean"
    run_progressive: bool = True 
    run_eigenvector: bool = False
    callbacks: Optional[CallbackSettings] = None
    force_cpu_for_large_metric_ops: bool = True

    def __post_init__(self):
        if self.callbacks is not None and isinstance(self.callbacks, dict):
            config_logger.debug(f"AlignmentConfig.__post_init__: self.callbacks is a dict, attempting to cast to CallbackSettings.")
            try:
                # Ensure that MetricTrackerConfig items are correctly instantiated if they are dicts
                processed_metrics_for_callback = []
                if 'alignment_metrics' in self.callbacks and isinstance(self.callbacks['alignment_metrics'], list):
                    for item_conf_dict in self.callbacks['alignment_metrics']:
                        if isinstance(item_conf_dict, dict):
                            processed_metrics_for_callback.append(MetricTrackerConfig(**item_conf_dict))
                        elif isinstance(item_conf_dict, MetricTrackerConfig):
                            processed_metrics_for_callback.append(item_conf_dict)
                        else:
                            raise ValueError(f"Invalid item type in callback alignment_metrics: {type(item_conf_dict)}")
                    # Create a new dict for CallbackSettings constructor to avoid modifying the original
                    callback_args = {k: v for k, v in self.callbacks.items() if k != 'alignment_metrics'}
                    callback_args['alignment_metrics'] = processed_metrics_for_callback
                    self.callbacks = CallbackSettings(**callback_args)
                else:
                     # If alignment_metrics is not a list or not present, try direct instantiation
                     self.callbacks = CallbackSettings(**self.callbacks)

            except Exception as e:
                config_logger.error(f"Failed to cast AlignmentConfig.callbacks to CallbackSettings in __post_init__: {e}. Setting to None.")
                self.callbacks = None
        elif self.callbacks is not None and not isinstance(self.callbacks, CallbackSettings):
            config_logger.warning(f"AlignmentConfig.__post_init__: self.callbacks is not a dict but also not CallbackSettings. Type: {type(self.callbacks)}. Leaving as is.")

    def validate(self) -> bool:
        """Validate alignment settings."""
        if not self.metric: # Check if the list is empty
            raise ValueError("alignment_settings.metric list cannot be empty. Please specify at least one metric.")
        
        valid_metrics_registry = ALIGNMENT_METRICS_REGISTRY.keys()
        for metric_name in self.metric: # Iterate through the list of metrics
            if not isinstance(metric_name, str):
                raise ValueError(f"Invalid type for metric name in alignment_settings.metric: {metric_name}. Expected str.")
            if metric_name.lower() not in valid_metrics_registry:
                warnings.warn(f"Specified metric '{metric_name}' in alignment_settings.metric not in ALIGNMENT_METRICS_REGISTRY. Available: {list(valid_metrics_registry)}")

        valid_cnn_modes = ["unfold", "patchwise", "batch_patch_combined", "filter_patch_summary", "filter_specific_covariance_rq"]
        if self.cnn_mode not in valid_cnn_modes:
            raise ValueError(f"Invalid CNN mode: '{self.cnn_mode}'. Supported: {valid_cnn_modes}")

        valid_rq_ops = ["mean", "max", "var", "sum"]
        if self.cnn_rq_aggregation_op not in valid_rq_ops:
            raise ValueError(f"Invalid cnn_rq_aggregation_op: '{self.cnn_rq_aggregation_op}'. Supported: {valid_rq_ops}")
        
        if self.callbacks is not None:
            if not isinstance(self.callbacks, CallbackSettings):
                # This check might be redundant if __post_init__ handles it, but good for safety
                raise ValueError(f"AlignmentConfig.callbacks is not a CallbackSettings instance after __post_init__. Type: {type(self.callbacks)}")
            self.callbacks.validate()
        return True


@dataclass
class WandbConfig(BaseConfig):
    """Configuration for Weights & Biases logging."""
    use_wandb: bool = False # Default to False, set to true in YAML if needed
    wandb_project: Optional[str] = "neural_alignment"
    wandb_entity: Optional[str] = None # User should fill this or it defaults to None (W&B default entity)
    # log_frequency: int = 1 # These were in your YAML under extra, could move here
    # log_images: bool = False
    # detailed_logging: bool = False 

    def validate(self) -> bool:
        if self.use_wandb and not self.wandb_project:
            # logger.warning("wandb_project not set, using default 'neural_alignment'.") # or raise error
            # self.wandb_project = "neural_alignment" # Already has a default
            pass 
        # wandb_entity can be None to use default W&B entity.
        return True


@dataclass
class CheckpointingConfig(BaseConfig):
    """Checkpointing configuration."""
    save_checkpoints: bool = False
    checkpoint_frequency: int = 1
    # use_wandb: bool = False # REMOVED - Moved to WandbConfig
    load_checkpoint: bool = False
    # save_model: true - this is save_networks at top level now
    # save_interval: 1 - this is checkpoint_frequency
    
    def validate(self) -> bool:
        if self.save_checkpoints and self.checkpoint_frequency <= 0:
            raise ValueError(f"Checkpoint frequency must be positive if save_checkpoints is true, got {self.checkpoint_frequency}")
        return True


@dataclass
class ExtraConfig(BaseConfig):
    """Extra configuration parameters - now mostly for logging or less critical options."""
    # Parameters like log_frequency, log_images, detailed_logging from your YAML's old extra can go here.
    log_frequency: int = 1
    log_images: bool = True 
    detailed_logging: bool = True
    dummy_extra_param: Optional[str] = None # Placeholder if it becomes empty

    def validate(self) -> bool:
        if self.log_frequency <= 0:
            # Allow 0 if it means "log never" or handle appropriately, for now positive
            # Or make it Optional[int] if it can be absent.
            # Based on typical usage, 1 might be "every epoch/step", so >0 is reasonable.
            pass # Or raise ValueError if it must be strictly positive
        return True


@dataclass
class ExperimentConfig(BaseConfig):
    """Main experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = "default_experiment"
    experiment_type: str = "AUTO"  # Changed default to AUTO
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
    seed: Optional[int] = None # Added seed for reproducibility
    
    # --- NEW: DDP Configuration ---
    use_ddp: bool = False # Whether to use Distributed Data Parallel
    ddp_backend: Optional[str] = "nccl" # DDP backend, e.g., "nccl", "gloo"
    # ddp_rank, ddp_world_size, ddp_local_rank will be set at runtime if use_ddp is true
    ddp_rank: int = 0 
    ddp_world_size: int = 1
    ddp_local_rank: int = 0 # Typically set from ENV or launcher
    # --- End NEW ---

    # Nested configurations
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    alignment_settings: AlignmentConfig = field(default_factory=AlignmentConfig) 
    pruning_settings: PruningConfig = field(default_factory=PruningConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    extra: ExtraConfig = field(default_factory=ExtraConfig)
    
    def __post_init__(self):
        # Auto-determine experiment_type if set to "AUTO"
        if self.experiment_type == "AUTO":
            if self.pruning_settings:
                pruning_mode = self.pruning_settings.dropout_pruning_mode
                if pruning_mode == "layer_isolated":
                    self.experiment_type = "layer_isolated_pruning"
                    config_logger.info(f"Experiment type AUTO resolved to 'layer_isolated_pruning' based on dropout_pruning_mode.")
                elif pruning_mode in ["global_joint", "layer_wise"]:
                    self.experiment_type = "progressive_dropout"
                    config_logger.info(f"Experiment type AUTO resolved to 'progressive_dropout' based on dropout_pruning_mode: {pruning_mode}.")
                elif pruning_mode == "cascading_layer":
                    self.experiment_type = "cascading_layer_pruning"
                    config_logger.info(f"Experiment type AUTO resolved to 'cascading_layer_pruning' based on dropout_pruning_mode.")
                else:
                    config_logger.warning(f"Experiment type AUTO could not be resolved from dropout_pruning_mode '{pruning_mode}'. Defaulting to 'progressive_dropout'.")
                    self.experiment_type = "progressive_dropout"
            else:
                # Default if pruning_settings is not available for some reason
                config_logger.warning("Experiment type AUTO specified, but pruning_settings are missing. Defaulting to 'progressive_dropout'.")
                self.experiment_type = "progressive_dropout"

        if self.wandb is not None and isinstance(self.wandb, dict):
             config_logger.debug(f"ExperimentConfig.__post_init__: self.wandb is a dict, attempting to cast to WandbConfig.")
             try:
                 self.wandb = WandbConfig(**self.wandb)
             except TypeError as e:
                 config_logger.error(f"Failed to cast self.wandb to WandbConfig: {e}. Using default WandbConfig.")
                 self.wandb = WandbConfig()
        elif self.wandb is not None and not isinstance(self.wandb, WandbConfig):
            config_logger.warning(f"ExperimentConfig.__post_init__: self.wandb is not a dict but also not WandbConfig. Type: {type(self.wandb)}. Leaving as is.")

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
        if not isinstance(self.alignment_settings, AlignmentConfig):
            self.alignment_settings = AlignmentConfig()
        if not isinstance(self.pruning_settings, PruningConfig):
            self.pruning_settings = PruningConfig()
        if not isinstance(self.checkpointing, CheckpointingConfig):
            self.checkpointing = CheckpointingConfig()
        if not isinstance(self.extra, ExtraConfig):
            self.extra = ExtraConfig()
            
        # Validate nested configs (manually to handle exceptions)
        try:
            self.dataset.validate()
            self.model.validate()
            self.training.validate()
            self.alignment_settings.validate()
            self.pruning_settings.validate()
            self.checkpointing.validate()
            if self.wandb: self.wandb.validate()
            self.extra.validate()
        except Exception as e:
            raise ValueError(f"Nested config validation error: {str(e)}")
        
        # Validate DDP config
        if self.use_ddp and self.ddp_backend not in ["nccl", "gloo", "mpi"]:
            raise ValueError(f"Invalid ddp_backend: {self.ddp_backend}. Supported: nccl, gloo, mpi")

        # Validate pruning settings against experiment type
        if self.experiment_type == "progressive_dropout" and \
           self.pruning_settings and \
           self.pruning_settings.dropout_pruning_mode == "layer_isolated":
            raise ValueError(
                "Invalid configuration: experiment_type 'progressive_dropout' cannot be used with "
                "dropout_pruning_mode 'layer_isolated'. For layer-isolated pruning, "
                "set experiment_type to 'layer_isolated_pruning' or AUTO."
            )
        # Add validation for cascading_layer_pruning experiment type
        if self.experiment_type == "cascading_layer_pruning" and \
           self.pruning_settings and \
           self.pruning_settings.dropout_pruning_mode != "cascading_layer":
            # If they explicitly set experiment_type to cascading, mode should match or be auto-inferred
            config_logger.warning(
                f"Experiment_type is 'cascading_layer_pruning', but dropout_pruning_mode is '{self.pruning_settings.dropout_pruning_mode}'. "
                f"Expected 'cascading_layer'. The experiment will run as cascading."
            )
            # Optionally, could force self.pruning_settings.dropout_pruning_mode = "cascading_layer" here
        
        config_logger.debug("ExperimentConfig.validate_config() completed.")

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