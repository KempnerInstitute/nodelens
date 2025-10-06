"""
Configuration validation utilities.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def validate_config(config_dict: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    if 'name' not in config_dict:
        errors.append("Missing required field: 'name'")

    # Model validation (dynamic from registry if available)
    if 'model_name' in config_dict:
        try:
            from alignment.core.registry import MODEL_REGISTRY
            valid_models = MODEL_REGISTRY.list()
        except Exception:
            valid_models = ['mlp', 'cnn2p2', 'resnet18', 'resnet50']
        if config_dict['model_name'] not in valid_models:
            errors.append(f"Invalid model_name: {config_dict['model_name']}. Must be one of {valid_models}")

    # Dataset validation (dynamic)
    if 'dataset_name' in config_dict:
        try:
            from alignment.core.registry import DATASET_REGISTRY
            valid_datasets = DATASET_REGISTRY.list()
        except Exception:
            valid_datasets = ['mnist', 'cifar10', 'cifar100', 'imagenet']
        if config_dict['dataset_name'] not in valid_datasets:
            errors.append(f"Invalid dataset_name: {config_dict['dataset_name']}. Must be one of {valid_datasets}")

    # Device validation
    if 'device' in config_dict:
        dev = str(config_dict['device'])
        if not (dev == 'cpu' or dev.startswith('cuda')):
            errors.append(f"Invalid device: {dev}. Must be 'cpu' or 'cuda[:N]'")

    # Numeric validations
    numeric_fields = {
        'batch_size': (1, 10000),
        'num_workers': (0, 32),
        'training_epochs': (0, 1000),
        'learning_rate': (1e-10, 10.0),
        'seed': (0, 2**32-1)
    }

    for field, (min_val, max_val) in numeric_fields.items():
        if field in config_dict:
            value = config_dict[field]
            if not isinstance(value, (int, float)):
                errors.append(f"{field} must be numeric, got {type(value).__name__}")
            elif value < min_val or value > max_val:
                errors.append(f"{field} must be between {min_val} and {max_val}, got {value}")

    # List validations
    if 'metrics' in config_dict:
        if not isinstance(config_dict['metrics'], list):
            errors.append("'metrics' must be a list")
        else:
            try:
                from alignment.core.registry import METRIC_REGISTRY
                valid_metrics = METRIC_REGISTRY.list()
            except Exception:
                valid_metrics = []
            for metric in config_dict['metrics']:
                if valid_metrics and metric not in valid_metrics:
                    errors.append(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    # Dropout fractions validation
    if 'dropout_fractions' in config_dict:
        if not isinstance(config_dict['dropout_fractions'], list):
            errors.append("'dropout_fractions' must be a list")
        else:
            for frac in config_dict['dropout_fractions']:
                if not isinstance(frac, (int, float)) or frac < 0 or frac > 1:
                    errors.append(f"Invalid dropout fraction: {frac}. Must be between 0 and 1")

    return errors


def validate_experiment_config(config_dict: Dict[str, Any], experiment_type: str) -> List[str]:
    """
    Validate configuration for specific experiment type.

    Args:
        config_dict: Configuration dictionary
        experiment_type: Type of experiment

    Returns:
        List of validation errors
    """
    errors = validate_config(config_dict)

    # Experiment-specific validations
    if experiment_type in ['progressive_dropout', 'global_dropout']:
        if 'dropout_fractions' not in config_dict:
            errors.append("Progressive dropout requires 'dropout_fractions'")

    elif experiment_type == 'eigenvector':
        if 'num_components' in config_dict:
            if config_dict['num_components'] < 1:
                errors.append("'num_components' must be at least 1")

    elif experiment_type == 'layer_isolated':
        if 'pruning_percentages' not in config_dict:
            errors.append("Layer isolated pruning requires 'pruning_percentages'")

    return errors


def check_compatibility(config_dict: Dict[str, Any]) -> List[str]:
    """
    Check for compatibility issues in configuration.

    Args:
        config_dict: Configuration dictionary

    Returns:
        List of warnings
    """
    warnings = []

    # Model-dataset compatibility
    if config_dict.get('model_name') == 'cnn2p2' and config_dict.get('dataset_name') == 'mnist':
        model_config = config_dict.get('model_config', {})
        if model_config.get('in_channels') != 1:
            warnings.append("CNN2P2 with MNIST should have in_channels=1")

    # Batch size warnings
    if config_dict.get('batch_size', 128) > 512 and config_dict.get('device', '').startswith('cpu'):
        warnings.append("Large batch size on CPU may be slow")

    # Memory warnings
    if config_dict.get('force_cpu_for_large_metric_ops', True) is False:
        if config_dict.get('model_name') in ['resnet50'] and config_dict.get('batch_size', 128) > 64:
            warnings.append("Large model with large batch size may cause GPU memory issues")

    return warnings
