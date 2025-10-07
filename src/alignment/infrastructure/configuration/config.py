"""
Configuration utilities for the alignment framework.
"""

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load based on extension
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to a file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save based on extension
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    logger.info(f"Saved config to {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


@dataclass
class Config:
    """
    Base configuration class with utilities.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create from dictionary."""
        return cls(**config_dict)

    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        save_config(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from file."""
        config_dict = load_config(path)
        return cls.from_dict(config_dict)

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")


@dataclass
class ExperimentConfig(Config):
    """
    Example experiment configuration.
    """

    # Experiment settings
    name: str = "experiment"
    seed: int = 42
    device: str = "cuda"

    # Model settings
    model_name: str = "resnet18"
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Dataset settings
    dataset_name: str = "cifar10"
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 128
    num_workers: int = 4

    # Training settings
    num_epochs: int = 100
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Logging settings
    log_interval: int = 100
    checkpoint_interval: int = 10

    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


def parse_args_to_config(args: Any, config_class: type = Config) -> Config:
    """
    Parse command line arguments to configuration.

    Args:
        args: Parsed arguments (e.g., from argparse)
        config_class: Configuration class to use

    Returns:
        Configuration instance
    """
    # Convert args to dict
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    else:
        args_dict = args

    # Filter out None values and non-config keys
    config_dict = {}
    for key, value in args_dict.items():
        if value is not None and not key.startswith("_"):
            config_dict[key] = value

    # Create config
    try:
        config = config_class(**config_dict)
    except TypeError:
        # If some args don't match, create default and update
        config = config_class()
        config.update(config_dict)

    return config
