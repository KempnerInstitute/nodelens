"""
Composable configuration components for experiments.

This module provides reusable configuration building blocks that can be
composed to create experiment configurations with less duplication.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Training-related configuration."""
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    gradient_clip_val: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_before_dropout": self.train_before_dropout,
            "training_epochs": self.training_epochs,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "scheduler_config": self.scheduler_config,
            "batch_size": self.batch_size,
            "gradient_clip_val": self.gradient_clip_val,
            "early_stopping_patience": self.early_stopping_patience
        }


@dataclass
class PruningConfig:
    """Pruning/dropout configuration."""
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"
    num_random_trials: int = 3
    pruning_metric: str = "rayleigh_quotient"
    pruning_strategy: str = "low"  # "low", "high", "random"
    exclude_classification_layer: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dropout_rates": self.dropout_rates,
            "dropout_mode": self.dropout_mode,
            "num_random_trials": self.num_random_trials,
            "pruning_metric": self.pruning_metric,
            "pruning_strategy": self.pruning_strategy,
            "exclude_classification_layer": self.exclude_classification_layer
        }


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_batches: Optional[int] = None
    eval_frequency: int = 1
    compute_alignment_during_eval: bool = True
    save_predictions: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "eval_batches": self.eval_batches,
            "eval_frequency": self.eval_frequency,
            "compute_alignment_during_eval": self.compute_alignment_during_eval,
            "save_predictions": self.save_predictions
        }


@dataclass
class CNNConfig:
    """CNN-specific configuration."""
    cnn_mode: str = "unfold"  # "unfold", "patchwise", "batch_patch_combined"
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cnn_mode": self.cnn_mode,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding
        }


@dataclass
class MultiNetworkConfig:
    """Multi-network training configuration."""
    num_networks: int = 1
    parallel_batch_size: Optional[int] = None
    use_tensorized_training: bool = True
    aggregate_metrics: bool = True
    save_individual_networks: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_networks": self.num_networks,
            "parallel_batch_size": self.parallel_batch_size,
            "use_tensorized_training": self.use_tensorized_training,
            "aggregate_metrics": self.aggregate_metrics,
            "save_individual_networks": self.save_individual_networks
        }


# Factory functions for common configurations

def create_standard_training_config(
    epochs: int = 10,
    learning_rate: float = 0.001,
    optimizer: str = "adam",
    **kwargs
) -> TrainingConfig:
    """Create a standard training configuration."""
    return TrainingConfig(
        training_epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        **kwargs
    )


def create_standard_pruning_config(
    dropout_rates: Optional[List[float]] = None,
    metric: str = "rayleigh_quotient",
    strategy: str = "low",
    **kwargs
) -> PruningConfig:
    """Create a standard pruning configuration."""
    return PruningConfig(
        dropout_rates=dropout_rates or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        pruning_metric=metric,
        pruning_strategy=strategy,
        **kwargs
    )


def create_quick_test_config(
    epochs: int = 2,
    dropout_rates: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Create a configuration for quick testing."""
    return {
        "training": create_standard_training_config(epochs=epochs),
        "pruning": create_standard_pruning_config(
            dropout_rates=dropout_rates or [0.0, 0.5],
            num_random_trials=1
        ),
        "evaluation": EvaluationConfig(eval_batches=5)
    }


# Backward compatibility helper

def flatten_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested config dictionary for backward compatibility.
    
    Converts:
        {"training": {"epochs": 10}, "pruning": {"dropout_rates": [0.1]}}
    To:
        {"epochs": 10, "dropout_rates": [0.1]}
    """
    flat = {}
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            flat.update(value)
        elif hasattr(value, 'to_dict'):
            flat.update(value.to_dict())
        else:
            flat[key] = value
    
    return flat


def unflatten_config_dict(
    flat_dict: Dict[str, Any],
    training_keys: Optional[List[str]] = None,
    pruning_keys: Optional[List[str]] = None,
    evaluation_keys: Optional[List[str]] = None,
    cnn_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Unflatten a config dictionary into components.
    
    Converts flat dictionary into nested structure with components.
    """
    # Default key mappings
    if training_keys is None:
        training_keys = [
            'train_before_dropout', 'training_epochs', 'learning_rate',
            'optimizer', 'scheduler', 'scheduler_config', 'batch_size',
            'gradient_clip_val', 'early_stopping_patience'
        ]
    
    if pruning_keys is None:
        pruning_keys = [
            'dropout_rates', 'dropout_mode', 'num_random_trials',
            'pruning_metric', 'pruning_strategy', 'exclude_classification_layer'
        ]
    
    if evaluation_keys is None:
        evaluation_keys = [
            'eval_batches', 'eval_frequency', 'compute_alignment_during_eval',
            'save_predictions'
        ]
    
    if cnn_keys is None:
        cnn_keys = ['cnn_mode', 'kernel_size', 'stride', 'padding']
    
    # Extract component configs
    components = {}
    remaining = flat_dict.copy()
    
    # Extract training config
    training_dict = {}
    for key in training_keys:
        if key in remaining:
            training_dict[key] = remaining.pop(key)
    if training_dict:
        components['training'] = TrainingConfig(**training_dict)
    
    # Extract pruning config
    pruning_dict = {}
    for key in pruning_keys:
        if key in remaining:
            pruning_dict[key] = remaining.pop(key)
    if pruning_dict:
        components['pruning'] = PruningConfig(**pruning_dict)
    
    # Extract evaluation config
    eval_dict = {}
    for key in evaluation_keys:
        if key in remaining:
            eval_dict[key] = remaining.pop(key)
    if eval_dict:
        components['evaluation'] = EvaluationConfig(**eval_dict)
    
    # Extract CNN config
    cnn_dict = {}
    for key in cnn_keys:
        if key in remaining:
            cnn_dict[key] = remaining.pop(key)
    if cnn_dict:
        components['cnn'] = CNNConfig(**cnn_dict)
    
    # Add remaining keys
    components.update(remaining)
    
    return components


def create_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration components from a dictionary.
    
    This is a convenience function that takes a flat or nested dictionary
    and returns properly structured configuration components.
    """
    # If already structured with components, return as-is
    if any(key in config_dict for key in ['training', 'pruning', 'evaluation', 'cnn']):
        return config_dict
    
    # Otherwise, unflatten into components
    return unflatten_config_dict(config_dict)


def create_backward_compatible_config(
    base_config: Any,
    training: Optional[TrainingConfig] = None,
    pruning: Optional[PruningConfig] = None,
    evaluation: Optional[EvaluationConfig] = None,
    **kwargs
) -> Any:
    """
    Create a backward-compatible configuration by merging components into a base config.
    
    This allows using the new component system with existing experiment classes.
    """
    # Start with base config
    if hasattr(base_config, '__dict__'):
        config_dict = vars(base_config).copy()
    else:
        config_dict = base_config.copy() if isinstance(base_config, dict) else {}
    
    # Merge in component configs
    if training:
        config_dict.update(training.to_dict())
    if pruning:
        config_dict.update(pruning.to_dict())
    if evaluation:
        config_dict.update(evaluation.to_dict())
    
    # Add any additional kwargs
    config_dict.update(kwargs)
    
    # If base_config was a class instance, create new instance with merged values
    if hasattr(base_config, '__class__') and hasattr(base_config.__class__, '__init__'):
        return base_config.__class__(**config_dict)
    
    return config_dict 