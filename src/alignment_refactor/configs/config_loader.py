"""
Configuration loading and saving utilities.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
import logging

from ..experiments.base import ExperimentConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ExperimentConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load raw config
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Handle environment variable substitution
    config_dict = _substitute_env_vars(config_dict)
    
    # Create ExperimentConfig
    try:
        config = ExperimentConfig.from_dict(config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def save_config(config: ExperimentConfig, save_path: Union[str, Path], 
                format: str = 'yaml') -> None:
    """
    Save configuration to a file.
    
    Args:
        config: ExperimentConfig object
        save_path: Path to save configuration
        format: Output format ('yaml' or 'json')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    if format == 'yaml':
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif format == 'json':
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved configuration to {save_path}")


def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in config.
    
    Environment variables should be specified as ${VAR_NAME} or ${VAR_NAME:default}.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Config dict with environment variables substituted
    """
    import re
    
    def substitute_value(value):
        if isinstance(value, str):
            # Pattern for ${VAR} or ${VAR:default}
            pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default if default is not None else match.group(0))
            
            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    
    return substitute_value(config_dict)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        Merged configuration
    """
    import copy
    
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    cli_args: Optional[List[str]] = None
) -> ExperimentConfig:
    """
    Load configuration with optional overrides.
    
    Args:
        config_path: Path to base configuration
        overrides: Dictionary of overrides
        cli_args: Command-line arguments in format ["key=value", ...]
        
    Returns:
        ExperimentConfig with overrides applied
    """
    # Load base config
    config = load_config(config_path)
    config_dict = config.to_dict()
    
    # Apply dictionary overrides
    if overrides:
        config_dict = merge_configs(config_dict, overrides)
    
    # Apply CLI overrides
    if cli_args:
        for arg in cli_args:
            if '=' in arg:
                key, value = arg.split('=', 1)
                # Convert value to appropriate type
                try:
                    value = eval(value)
                except:
                    pass  # Keep as string
                
                # Handle nested keys (e.g., "model.hidden_dims=[300,200]")
                keys = key.split('.')
                target = config_dict
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value
    
    return ExperimentConfig.from_dict(config_dict) 