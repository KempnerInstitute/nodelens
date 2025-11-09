"""
Configuration loading and saving utilities.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

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
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    # Handle environment variable substitution
    config_dict = _substitute_env_vars(config_dict)

    # Map nested config to flat ExperimentConfig structure
    config_dict = _map_nested_to_flat_config(config_dict)

    # Create ExperimentConfig
    try:
        config = ExperimentConfig.from_dict(config_dict)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")



def save_config(config: ExperimentConfig, save_path: Union[str, Path], format: str = "yaml") -> None:
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

    if format == "yaml":
        with open(save_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif format == "json":
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved configuration to {save_path}")


def _map_nested_to_flat_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map nested YAML config structure to flat ExperimentConfig structure.

    Args:
        nested_config: Nested configuration from YAML

    Returns:
        Flat configuration suitable for ExperimentConfig
    """
    flat_config = {}

    # Map experiment metadata
    if "experiment_name" in nested_config:
        flat_config["name"] = nested_config["experiment_name"]
    elif "name" in nested_config:
        flat_config["name"] = nested_config["name"]
    else:
        flat_config["name"] = "default_experiment"

    flat_config["description"] = nested_config.get("description", "")
    flat_config["tags"] = nested_config.get("tags", [])

    # Map other top-level fields
    flat_config["pretrained"] = nested_config.get("pretrained", False)
    if "tracked_layers" in nested_config:
        flat_config["tracked_layers"] = nested_config["tracked_layers"]

    # Map dataset configuration
    if "dataset" in nested_config:
        dataset = nested_config["dataset"]
        # Normalize dataset name
        dataset_name = dataset.get("name", dataset.get("dataset_name", "MNIST"))
        flat_config["dataset_name"] = dataset_name.lower()  # Lowercase for consistency
        flat_config["data_path"] = dataset.get("data_path")
        flat_config["batch_size"] = dataset.get("batch_size", 128)
        flat_config["num_workers"] = dataset.get("num_workers", 4)

        # Filter out DataLoader-specific parameters (not Dataset parameters)
        dataloader_params = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "drop_last",
            "persistent_workers",
            "prefetch_factor",
            "name",
            "dataset_name",
            "data_path",
        ]
        flat_config["dataset_config"] = {k: v for k, v in dataset.items() if k not in dataloader_params}
    else:
        # Handle flat structure where dataset fields are at top level
        dataset_name = nested_config.get("dataset_name", "MNIST")
        flat_config["dataset_name"] = dataset_name.lower()  # Lowercase for consistency
        flat_config["data_path"] = nested_config.get("data_path")
        flat_config["batch_size"] = nested_config.get("batch_size", 128)
        flat_config["num_workers"] = nested_config.get("num_workers", 4)
        flat_config["dataset_config"] = nested_config.get("dataset_config", {})

    # Map model configuration
    if "model" in nested_config:
        model = nested_config["model"]
        model_name = model.get("name", model.get("model_name", "mlp"))
        flat_config["model_name"] = model_name
        flat_config["model_config"] = {}

        # Handle different model types
        if "mlp_params" in model:
            flat_config["model_config"].update(model["mlp_params"])
        elif "cnn2p2_params" in model:
            flat_config["model_config"].update(model["cnn2p2_params"])
        elif "external_params" in model:
            # For torchvision models
            external = model["external_params"]
            if external.get("source") == "torchvision":
                flat_config["model_name"] = external.get("name_or_path", "resnet18")
                flat_config["pretrained"] = external.get("pretrained", False)

        # Add common model params
        if "output_dim" in model:
            flat_config["model_config"]["output_dim"] = model["output_dim"]
        if "dropout_rate" in model:
            flat_config["model_config"]["dropout_rate"] = model["dropout_rate"]
        if "alignment_layers" in model:
            flat_config["tracked_layers"] = (
                list(model["alignment_layers"].keys()) if isinstance(model["alignment_layers"], dict) else model["alignment_layers"]
            )
    else:
        # Handle flat structure where model_name is at top level
        model_name = nested_config.get("model_name", "mlp")
        flat_config["model_name"] = model_name
        flat_config["model_config"] = nested_config.get("model_config", {})

        # Handle different model types from flat structure
        if "mlp_params" in nested_config:
            flat_config["model_config"].update(nested_config["mlp_params"])
        elif "cnn2p2_params" in nested_config:
            flat_config["model_config"].update(nested_config["cnn2p2_params"])
        elif "external_params" in nested_config:
            # For torchvision models
            external = nested_config["external_params"]
            if external.get("source") == "torchvision":
                flat_config["model_name"] = external.get("name_or_path", "resnet18")
                flat_config["pretrained"] = external.get("pretrained", False)

        # Add common model params from flat structure
        if "output_dim" in nested_config:
            flat_config["model_config"]["output_dim"] = nested_config["output_dim"]
        if "dropout_rate" in nested_config:
            flat_config["model_config"]["dropout_rate"] = nested_config["dropout_rate"]
        if "alignment_layers" in nested_config:
            flat_config["tracked_layers"] = (
                list(nested_config["alignment_layers"].keys())
                if isinstance(nested_config["alignment_layers"], dict)
                else nested_config["alignment_layers"]
            )

    # Map training configuration
    if "training" in nested_config:
        training = nested_config["training"]
        flat_config["training_epochs"] = training.get("epochs", 10)
        flat_config["learning_rate"] = training.get("learning_rate", 0.001)
        flat_config["optimizer"] = training.get("optimizer", "Adam").lower()
        flat_config["train_before_dropout"] = training.get("train_before_dropout", True)

    # Map alignment settings
    flat_config["alignment_methods"] = nested_config.get("alignment_methods", ["rayleigh_quotient"])
    flat_config["alignment_data_num_samples"] = nested_config.get("alignment_data_num_samples", 1)
    if "alignment_settings" in nested_config:
        alignment = nested_config["alignment_settings"]
        # Map metric names
        metrics = alignment.get("metric", ["RQ"])
        metric_mapping = {
            "RQ": "rayleigh_quotient",
            "MI_G": "mutual_information",
            "Node_Redundancy": "redundancy",
            "PID_SI": "partial_information_decomposition",
        }
        flat_config["metrics"] = [metric_mapping.get(m, m) for m in metrics]
        flat_config["scale_by_norm"] = alignment.get("scale_by_norm", False)
        flat_config["force_cpu_for_large_metric_ops"] = alignment.get("force_cpu_for_large_metric_ops", False)
        flat_config["cnn_rq_aggregation_op"] = alignment.get("cnn_rq_aggregation_op", "mean")

    # Map pruning settings
    if "pruning_settings" in nested_config:
        pruning = nested_config["pruning_settings"]
        flat_config["exclude_classification_layer"] = pruning.get("exclude_classification_layer", True)

    # Map other settings
    flat_config["device"] = nested_config.get("device", "cuda")
    flat_config["seed"] = nested_config.get("seed", 42)

    # Map pruning configuration (check both top-level and nested 'pruning' block)
    pruning_block = nested_config.get("pruning", {})

    # Map top-level analysis flags
    flat_config["do_pruning_experiments"] = pruning_block.get("enabled", nested_config.get("do_pruning_experiments", False))
    flat_config["do_dropout_analysis"] = nested_config.get("dropout", {}).get("enabled", nested_config.get("do_dropout_analysis", False))
    flat_config["do_eigenfeature_analysis"] = nested_config.get("do_eigenfeature_analysis", False)

    # Map pruning parameters (prioritize nested pruning block, fallback to top-level)
    flat_config["pruning_strategies"] = pruning_block.get("algorithms", nested_config.get("pruning_strategies", ["magnitude", "random"]))
    flat_config["pruning_amounts"] = pruning_block.get("sparsity_levels", nested_config.get("pruning_amounts", [0.1, 0.3, 0.5, 0.7, 0.9]))
    flat_config["pruning_selection_mode"] = pruning_block.get("selection_modes", [nested_config.get("pruning_selection_mode", "low")])[0]
    flat_config["fine_tune_after_pruning"] = pruning_block.get("fine_tune_after_pruning", nested_config.get("fine_tune_after_pruning", True))
    flat_config["fine_tune_epochs"] = pruning_block.get("fine_tune_epochs", nested_config.get("fine_tune_epochs", 5))
    flat_config["pruning_alignment_metric"] = pruning_block.get(
        "alignment_metric", nested_config.get("pruning_alignment_metric", "rayleigh_quotient")
    )

    # Evaluation
    flat_config["do_perplexity_computation"] = nested_config.get("do_perplexity_computation", False)
    flat_config["evaluation_dataset"] = nested_config.get("evaluation_dataset", "wikitext")
    flat_config["evaluation_num_samples"] = nested_config.get("evaluation_num_samples", 100)

    # Map visualization settings
    flat_config["generate_plots"] = nested_config.get("generate_plots", True)
    flat_config["plot_format"] = nested_config.get("plot_format", "png")
    flat_config["plot_dpi"] = nested_config.get("plot_dpi", 300)

    # Map checkpointing
    if "checkpointing" in nested_config:
        checkpoint = nested_config["checkpointing"]
        flat_config["checkpoint_interval"] = checkpoint.get("checkpoint_frequency", 1) * 1000  # Convert to steps
        flat_config["save_best"] = checkpoint.get("save_checkpoints", True)

    # Map wandb settings
    if "wandb" in nested_config:
        wandb = nested_config["wandb"]
        if wandb.get("use_wandb", False):
            flat_config["wandb_project"] = wandb.get("wandb_project")
            flat_config["wandb_entity"] = wandb.get("wandb_entity")

    # Map paths
    flat_config["log_dir"] = nested_config.get("results_path", "./logs")
    flat_config["checkpoint_dir"] = os.path.join(flat_config["log_dir"], "checkpoints")

    return flat_config


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
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

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
    config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None, cli_args: Optional[List[str]] = None
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
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Convert value to appropriate type
                try:
                    value = eval(value)
                except Exception:
                    pass  # Keep as string

                # Handle nested keys (e.g., "model.hidden_dims=[300,200]")
                keys = key.split(".")
                target = config_dict
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                target[keys[-1]] = value

    return ExperimentConfig.from_dict(config_dict)
