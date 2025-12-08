"""
Configuration loading and saving utilities.

Supports both original format and unified format configs.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..experiments.base import ExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED FORMAT DETECTION AND CONVERSION
# =============================================================================

# Metric name mappings: unified -> original
METRIC_UNIFIED_TO_ORIGINAL = {
    "rayleigh_quotient": "rayleigh_quotient",
    "redundancy": "gaussian_mi_analytic",
    "average_redundancy": "average_redundancy",  # Keep as-is
    "synergy": "synergy_gaussian_mmi",
    "magnitude": "activation_l2_norm",
    "taylor": "taylor",  # Vision taylor importance
}

# Reverse mapping: original -> unified
METRIC_ORIGINAL_TO_UNIFIED = {v: k for k, v in METRIC_UNIFIED_TO_ORIGINAL.items()}
METRIC_ORIGINAL_TO_UNIFIED.update({
    "average_redundancy": "redundancy",
    "pairwise_redundancy_gaussian": "redundancy",
    "gaussian_mi": "redundancy",
})


def _is_unified_format(config_dict: Dict[str, Any]) -> bool:
    """
    Detect if config is in unified format.
    
    Unified format characteristics:
    - metrics block has nested dicts with 'enabled' keys (not a list)
    - Has 'extra' section for experiment-specific settings
    - Uses unified metric names (redundancy, magnitude, etc.)
    """
    metrics = config_dict.get("metrics", {})
    if not isinstance(metrics, dict):
        return False
    
    # Unified format: metrics.rayleigh_quotient.enabled exists
    # Original format: metrics.enabled is a list
    if "enabled" in metrics and isinstance(metrics["enabled"], list):
        return False
    
    # Check for unified metric structure
    unified_metrics = ["rayleigh_quotient", "redundancy", "synergy", "magnitude", "scar"]
    for metric in unified_metrics:
        if metric in metrics and isinstance(metrics[metric], dict):
            if "enabled" in metrics[metric]:
                return True
    
    # Check for 'extra' section (strong indicator of unified format)
    if "extra" in config_dict:
        return True
    
    return False


def _convert_unified_to_original(unified: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert unified format config to original format.
    
    This ensures that the unified config produces the exact same
    ExperimentConfig as the original format would.
    """
    original = {}
    
    # -------------------------------------------------------------------------
    # EXPERIMENT
    # -------------------------------------------------------------------------
    if "experiment" in unified:
        exp = unified["experiment"]
        original["experiment"] = {
            "name": exp.get("name", "experiment"),
            "type": exp.get("type", "alignment_analysis"),
        }
        original["seed"] = exp.get("seed", 42)
        original["device"] = exp.get("device", "cuda")
        if "output_dir" in exp:
            original["results_path"] = exp["output_dir"]
        if "num_networks" in exp:
            original["num_networks"] = exp["num_networks"]
        if "save_activations" in exp:
            original["save_activations"] = exp["save_activations"]
    
    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------
    if "model" in unified:
        model = unified["model"]
        original["model"] = {
            "name": model.get("name", "resnet18"),
            "pretrained": model.get("pretrained", True),
        }
        # LLM fields
        if "model_id" in model:
            original["model"]["model_id"] = model["model_id"]
        if "dtype" in model:
            original["model"]["dtype"] = model["dtype"]
        if "device_map" in model:
            original["model"]["device_map"] = model["device_map"]
        if "trust_remote_code" in model:
            original["model"]["trust_remote_code"] = model["trust_remote_code"]
        if "tracked_layers" in model:
            original["model"]["tracked_layers"] = model["tracked_layers"]
        if "num_classes" in model:
            original["model"]["num_classes"] = model["num_classes"]
    
    # -------------------------------------------------------------------------
    # DATASET
    # -------------------------------------------------------------------------
    if "dataset" in unified:
        dataset = unified["dataset"]
        original["dataset"] = {
            "name": dataset.get("name", "cifar10"),
            "batch_size": dataset.get("batch_size", 128),
            "num_workers": dataset.get("num_workers", 4),
        }
        if "subset" in dataset:
            original["dataset"]["subset"] = dataset["subset"]
        if "split" in dataset:
            original["dataset"]["split"] = dataset["split"]
        if "root" in dataset:
            original["dataset"]["data_path"] = dataset["root"]
    
    # -------------------------------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------------------------------
    if "calibration" in unified:
        cal = unified["calibration"]
        # Put calibration info in metrics block (where original format expects it)
        if "metrics" not in original:
            original["metrics"] = {}
        original["metrics"]["num_samples"] = cal.get("num_samples", 5000)
        # Also keep calibration block for LLM experiments
        original["calibration"] = cal
    
    # -------------------------------------------------------------------------
    # METRICS - Convert unified names to original names
    # -------------------------------------------------------------------------
    if "metrics" in unified:
        metrics = unified["metrics"]
        enabled_metrics = []
        metric_configs = {}
        
        # Check each unified metric
        for unified_name, original_name in METRIC_UNIFIED_TO_ORIGINAL.items():
            if unified_name in metrics:
                metric_cfg = metrics[unified_name]
                if isinstance(metric_cfg, dict):
                    if metric_cfg.get("enabled", True):
                        enabled_metrics.append(original_name)
                        # Copy metric-specific params
                        params = {k: v for k, v in metric_cfg.items() if k != "enabled"}
                        if params:
                            metric_configs[original_name] = params
                elif metric_cfg is True:
                    enabled_metrics.append(original_name)
        
        # Handle SCAR metrics (LLM-specific)
        if "scar" in metrics:
            scar = metrics["scar"]
            if isinstance(scar, dict) and scar.get("enabled", True):
                original["do_scar_metrics"] = True
                original["scar_num_samples"] = scar.get("num_samples", 64)
                original["scar_max_length"] = scar.get("max_length", 512)
        
        # Handle additional metrics
        # Note: Skip analysis-derived metrics that are computed by analysis pipelines
        # (not standalone metrics that can be computed independently)
        ANALYSIS_DERIVED_METRICS = {
            "supernode_protection_score",
            "supernode_connectivity_score", 
            "scar_activation_power",
            "scar_curvature",
            "scar_loss_proxy",
            "scar_taylor",
        }
        if "additional" in metrics:
            for name, cfg in metrics["additional"].items():
                if name in ANALYSIS_DERIVED_METRICS:
                    continue  # Skip analysis-derived metrics
                if isinstance(cfg, dict) and cfg.get("enabled", True):
                    enabled_metrics.append(name)
        
        original["metrics"] = {
            "enabled": enabled_metrics,
            **metric_configs,
        }
        
        # Composite weights - convert unified names to original
        if "composite_weights" in metrics:
            comp_weights = {}
            for name, weight in metrics["composite_weights"].items():
                original_name = METRIC_UNIFIED_TO_ORIGINAL.get(name, name)
                comp_weights[original_name] = weight
            original["metrics"]["composite_weights"] = comp_weights
    
    # -------------------------------------------------------------------------
    # SUPERNODE (LLM outlier detection)
    # -------------------------------------------------------------------------
    if "supernode" in unified:
        original["supernode"] = unified["supernode"]
    
    # -------------------------------------------------------------------------
    # CLUSTERING (Vision)
    # -------------------------------------------------------------------------
    if "clustering" in unified:
        original["clustering"] = unified["clustering"]
    
    # -------------------------------------------------------------------------
    # HALO ANALYSIS
    # -------------------------------------------------------------------------
    if "halo_analysis" in unified:
        original["halo_analysis"] = unified["halo_analysis"]
        if unified["halo_analysis"].get("enabled"):
            original["do_halo_analysis"] = True
    
    # -------------------------------------------------------------------------
    # CASCADE ANALYSIS
    # -------------------------------------------------------------------------
    if "cascade_analysis" in unified:
        original["cascade_analysis"] = unified["cascade_analysis"]
    
    # -------------------------------------------------------------------------
    # PRUNING - Convert unified metric names in algorithms/scoring_methods
    # -------------------------------------------------------------------------
    if "pruning" in unified:
        pruning = unified["pruning"]
        original_pruning = {
            "enabled": pruning.get("enabled", True),
        }
        
        # Ratios/sparsity levels
        if "ratios" in pruning:
            original_pruning["sparsity_levels"] = pruning["ratios"]
        elif "sparsity_levels" in pruning:
            original_pruning["sparsity_levels"] = pruning["sparsity_levels"]
        
        # Selection modes
        if "selection_modes" in pruning:
            original_pruning["selection_modes"] = pruning["selection_modes"]
        
        # Convert algorithm names
        if "algorithms" in pruning:
            converted_algorithms = []
            for alg in pruning["algorithms"]:
                converted_algorithms.append(METRIC_UNIFIED_TO_ORIGINAL.get(alg, alg))
            original_pruning["algorithms"] = converted_algorithms
        
        # Convert scoring methods
        if "scoring_methods" in pruning:
            converted_scoring = []
            for method in pruning["scoring_methods"]:
                converted_scoring.append(METRIC_UNIFIED_TO_ORIGINAL.get(method, method))
            original_pruning["scoring_methods"] = converted_scoring
        
        # Other pruning fields
        for key in ["distribution", "structured", "dependency_aware", "target", "single_strategy"]:
            if key in pruning:
                original_pruning[key] = pruning[key]
        
        # Fine-tune settings
        if "fine_tune" in pruning:
            original_pruning["fine_tune"] = pruning["fine_tune"]
        
        original["pruning"] = original_pruning
    
    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    if "evaluation" in unified:
        ev = unified["evaluation"]
        original["evaluation"] = {"enabled": ev.get("enabled", True)}
        
        # Perplexity (LLM)
        if ev.get("perplexity_enabled"):
            original["do_perplexity_computation"] = True
            if "perplexity_datasets" in ev:
                # Convert to original format
                original["evaluation"]["perplexity"] = {
                    "enabled": True,
                    "datasets": ev["perplexity_datasets"],
                }
        
        # bits_per_byte
        if ev.get("bits_per_byte"):
            original["evaluation"]["bits_per_byte"] = True
        
        # evaluation_num_samples
        if "evaluation_num_samples" in ev:
            original["evaluation_num_samples"] = ev["evaluation_num_samples"]
        
        # Benchmarks (LLM)
        if ev.get("benchmarks_enabled"):
            if "benchmark_tasks" in ev:
                original["evaluation"]["benchmarks"] = ev["benchmark_tasks"]
            original["evaluation"]["batch_size"] = ev.get("benchmark_batch_size", 8)
    
    # -------------------------------------------------------------------------
    # PERFORMANCE
    # -------------------------------------------------------------------------
    if "performance" in unified:
        original["performance"] = unified["performance"]
    
    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    if "visualization" in unified:
        viz = unified["visualization"]
        original["visualization"] = viz
        if viz.get("enabled", True):
            original["generate_plots"] = True
        if "format" in viz:
            original["plot_format"] = viz["format"]
        if "dpi" in viz:
            original["plot_dpi"] = viz["dpi"]
    
    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------
    if "output" in unified:
        out = unified["output"]
        if "dir" in out:
            original["results_path"] = out["dir"]
            # Also set experiment.output_dir for compatibility
            if "experiment" in original:
                original["experiment"]["output_dir"] = out["dir"]
    
    # -------------------------------------------------------------------------
    # EXTRA - Expand LLM-specific settings from extra block to top-level
    # -------------------------------------------------------------------------
    if "extra" in unified:
        extra = unified["extra"]
        
        # Analysis options (with all detailed settings) - TOP LEVEL
        if "analysis" in extra:
            original["analysis"] = extra["analysis"]
        
        # Supernode robustness - TOP LEVEL
        if "supernode_robustness" in extra:
            original["supernode_robustness"] = extra["supernode_robustness"]
        
        # Supernode summary - TOP LEVEL
        if "supernode_summary" in extra:
            original["supernode_summary"] = extra["supernode_summary"]
        
        # Multi-supernode - TOP LEVEL
        if "multi_supernode" in extra:
            original["multi_supernode"] = extra["multi_supernode"]
        
        # Cross-layer - TOP LEVEL
        if "cross_layer" in extra:
            original["cross_layer"] = extra["cross_layer"]
            if extra["cross_layer"].get("enabled"):
                original["do_connectivity_pruning"] = True
        
        # Generalized importance - TOP LEVEL
        if "generalized_importance" in extra:
            original["generalized_importance"] = extra["generalized_importance"]
            if extra["generalized_importance"].get("enabled"):
                original["do_generalized_importance"] = True
        
        # Halo analysis (detailed settings from extra override top-level)
        if "halo_analysis" in extra:
            if "halo_analysis" not in original:
                original["halo_analysis"] = {}
            original["halo_analysis"].update(extra["halo_analysis"])
            if extra["halo_analysis"].get("enabled"):
                original["do_halo_analysis"] = True
        
        # Visualization (detailed paper figure settings) - MERGE with top-level
        if "visualization" in extra:
            if "visualization" not in original:
                original["visualization"] = {}
            # Merge extra.visualization into top-level visualization
            extra_viz = extra["visualization"]
            for key, value in extra_viz.items():
                original["visualization"][key] = value
        
        # Top-level flags from extra
        for flag in ["do_scar_metrics", "do_directed_redundancy", "do_connectivity_pruning", 
                     "do_halo_analysis", "do_generalized_importance"]:
            if flag in extra:
                original[flag] = extra[flag]
        
        # Pretrain settings (for vision)
        if "pretrain_epochs" in extra:
            original["pretrain_epochs"] = extra["pretrain_epochs"]
        if "pretrain_lr" in extra:
            original["pretrain_lr"] = extra["pretrain_lr"]
        
        # Baselines (for vision)
        if "baselines" in extra:
            original["baselines"] = extra["baselines"]
        
        # Sensitivity analysis (for vision)
        if "sensitivity_analysis" in extra:
            original["sensitivity_analysis"] = extra["sensitivity_analysis"]
        
        # Structured pruning (for vision)
        if "structured_pruning" in extra:
            original["structured_pruning"] = extra["structured_pruning"]
        
        # Feature analysis (for vision)
        if "feature_analysis" in extra:
            original["feature_analysis"] = extra["feature_analysis"]
        
        # Efficiency tracking (for vision)
        if "efficiency" in extra:
            original["efficiency"] = extra["efficiency"]
    
    # -------------------------------------------------------------------------
    # BUILD LLM BLOCK - Reconstruct full llm: section as original expects
    # -------------------------------------------------------------------------
    llm_block = {}
    
    # SCAR settings
    if original.get("do_scar_metrics"):
        llm_block["scar_metrics"] = True
    if "scar_num_samples" in original:
        llm_block["scar_num_samples"] = original["scar_num_samples"]
    if "scar_max_length" in original:
        llm_block["scar_max_length"] = original["scar_max_length"]
    
    # Perplexity settings
    if original.get("do_perplexity_computation"):
        llm_block["evaluate_perplexity"] = True
    
    # Build evaluation_metrics list from evaluation.benchmarks
    evaluation_metrics = []
    if "evaluation" in original:
        ev = original["evaluation"]
        
        # Perplexity metrics
        if ev.get("perplexity", {}).get("enabled") or original.get("do_perplexity_computation"):
            evaluation_metrics.extend(["perplexity", "loss", "bits_per_byte"])
        
        # Build benchmark metrics from benchmark tasks
        if "benchmarks" in ev:
            for benchmark in ev["benchmarks"]:
                if isinstance(benchmark, dict):
                    task_name = benchmark.get("name", "")
                    # Map task name to evaluation_metrics format
                    if task_name == "mmlu":
                        evaluation_metrics.append("accuracy_mmlu")
                    elif task_name == "hellaswag":
                        evaluation_metrics.append("accuracy_hellaswag")
                    elif task_name == "piqa":
                        evaluation_metrics.append("accuracy_piqa")
                    elif task_name == "boolq":
                        evaluation_metrics.append("accuracy_boolq")
                    elif task_name == "winogrande":
                        evaluation_metrics.append("accuracy_winogrande")
                    elif task_name == "arc_easy":
                        evaluation_metrics.append("accuracy_arc_easy")
                    elif task_name == "arc_challenge":
                        evaluation_metrics.append("accuracy_arc_challenge")
                    elif task_name == "openbookqa":
                        evaluation_metrics.append("accuracy_openbookqa")
                    elif task_name == "gsm8k":
                        evaluation_metrics.append("accuracy_gsm8k")
                    elif task_name == "truthfulqa":
                        evaluation_metrics.append("accuracy_truthfulqa")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_metrics = []
    for m in evaluation_metrics:
        if m not in seen:
            seen.add(m)
            unique_metrics.append(m)
    
    if unique_metrics:
        llm_block["evaluation_metrics"] = unique_metrics
    
    if llm_block:
        original["llm"] = llm_block
    
    # -------------------------------------------------------------------------
    # ENSURE TOP-LEVEL FLAGS ARE SET based on section enables
    # -------------------------------------------------------------------------
    # Set do_scar_metrics if SCAR section is enabled
    if unified.get("metrics", {}).get("scar", {}).get("enabled"):
        original["do_scar_metrics"] = True
        original["scar_num_samples"] = unified["metrics"]["scar"].get("num_samples", 64)
        original["scar_max_length"] = unified["metrics"]["scar"].get("max_length", 512)
    
    # Set flags based on section enablement
    if unified.get("supernode", {}).get("enabled"):
        if unified["supernode"].get("cross_layer_analysis"):
            original["do_connectivity_pruning"] = True
    
    if unified.get("halo_analysis", {}).get("enabled"):
        original["do_halo_analysis"] = True
    
    logger.info("Converted unified config to original format")
    return original


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Supports both original format and unified format configs.
    Unified format configs are automatically detected and converted.

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

    # Detect and convert unified format to original format
    if _is_unified_format(config_dict):
        logger.info(f"Detected unified config format in {config_path}")
        config_dict = _convert_unified_to_original(config_dict)

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

    # Map experiment metadata (support nested "experiment" block)
    experiment_block = nested_config.get("experiment", {})
    name = (
        nested_config.get("experiment_name")
        or experiment_block.get("name")
        or nested_config.get("name")
        or "default_experiment"
    )
    flat_config["name"] = name

    flat_config["experiment_type"] = nested_config.get(
        "experiment_type",
        experiment_block.get("type", "alignment_analysis"),
    )

    flat_config["description"] = nested_config.get("description", experiment_block.get("description", ""))
    flat_config["tags"] = nested_config.get("tags", experiment_block.get("tags", []))

    # Map other top-level fields
    # Note: pretrained may be at top-level or in model block - handled below
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

    # Map metric configuration block (optional nested structure)
    metric_block = nested_config.get("metrics")
    if isinstance(metric_block, dict):
        enabled_metrics = metric_block.get("enabled")
        if enabled_metrics is not None:
            flat_config["metrics"] = enabled_metrics

        metric_configs = flat_config.get("metric_configs", {}).copy()
        for metric_name, metric_cfg in metric_block.items():
            if metric_name == "enabled" or metric_cfg is None:
                continue
            if isinstance(metric_cfg, dict):
                metric_configs[metric_name] = metric_cfg
        if metric_configs:
            flat_config["metric_configs"] = metric_configs

    # Map model configuration
    if "model" in nested_config:
        model = nested_config["model"]
        model_name = model.get("name", model.get("model_name", "mlp"))
        flat_config["model_name"] = model_name
        flat_config["model_config"] = {}
        
        # Handle pretrained flag directly in model block
        if "pretrained" in model:
            flat_config["pretrained"] = model["pretrained"]
        
        # Handle tracked_layers from model block
        if "tracked_layers" in model:
            flat_config["tracked_layers"] = model["tracked_layers"]

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
        
        # Handle HuggingFace model config (for LLMs)
        hf_fields = ["model_id", "model_backend", "dtype", "torch_dtype", "device_map"]
        for field in hf_fields:
            if field in model:
                # Normalize dtype field name (prefer 'dtype', but accept 'torch_dtype')
                if field == "dtype":
                    flat_config["model_config"]["torch_dtype"] = model[field]
                else:
                    flat_config["model_config"][field] = model[field]
        # Map hf_device_map -> device_map for backward compatibility
        if "hf_device_map" in model:
            flat_config["model_config"]["device_map"] = model["hf_device_map"]

        # Add common model params
        if "output_dim" in model:
            flat_config["model_config"]["output_dim"] = model["output_dim"]
        if "dropout_rate" in model:
            flat_config["model_config"]["dropout_rate"] = model["dropout_rate"]
        if "in_channels" in model:
            flat_config["model_config"]["in_channels"] = model["in_channels"]
        if "hidden_channels" in model:
            flat_config["model_config"]["hidden_channels"] = model["hidden_channels"]
        if "example_input_hw" in model:
            flat_config["model_config"]["example_input_hw"] = tuple(model["example_input_hw"])
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
    # Auto-disable training when using pretrained models (unless explicitly enabled)
    is_pretrained = flat_config.get("pretrained", False)
    
    if "training" in nested_config:
        training = nested_config["training"]
        # Handle both 'enabled' (new) and 'do_train' (old) keys
        if "enabled" in training:
            flat_config["do_train"] = training.get("enabled", False)
        elif "do_train" in training:
            flat_config["do_train"] = training.get("do_train", False)
        else:
            # If training block exists but enabled/do_train not specified,
            # default to False for pretrained models, True otherwise
            flat_config["do_train"] = not is_pretrained
        flat_config["training_epochs"] = training.get("epochs", 10)
        flat_config["learning_rate"] = training.get("learning_rate", 0.001)
        flat_config["optimizer"] = training.get("optimizer", "Adam").lower()
        flat_config["train_before_dropout"] = training.get("train_before_dropout", True)
        if "scheduler" in training:
            flat_config["scheduler"] = training.get("scheduler", "none")
        if "scheduler_config" in training:
            flat_config["scheduler_config"] = training["scheduler_config"]
        if "momentum" in training:
            flat_config["momentum"] = training["momentum"]
        if "weight_decay" in training:
            flat_config["weight_decay"] = training["weight_decay"]
        # Multi-network support for statistical error bars
        flat_config["num_networks"] = training.get("num_networks", 1)
    else:
        # No training block - auto-disable for pretrained models
        if is_pretrained:
            flat_config["do_train"] = False

    # Map alignment/metrics settings
    # Priority: metrics.enabled > alignment.methods > alignment_methods > default
    flat_config["alignment_methods"] = nested_config.get("alignment_methods", ["rayleigh_quotient"])
    flat_config["alignment_data_num_samples"] = nested_config.get("alignment_data_num_samples", 1)
    
    # Handle metrics block (new cleaner format)
    metrics_block = nested_config.get("metrics", {})
    if isinstance(metrics_block, dict):
        if "enabled" in metrics_block:
            flat_config["alignment_methods"] = metrics_block["enabled"]
        if "num_samples" in metrics_block:
            flat_config["alignment_data_num_samples"] = metrics_block["num_samples"]
        # Composite weights from metrics block
        if "composite_weights" in metrics_block:
            flat_config["alignment_composite_weights"] = metrics_block["composite_weights"]
    
    # Handle nested alignment block (backward compatibility)
    if "alignment" in nested_config and isinstance(nested_config["alignment"], dict):
        alignment_block = nested_config["alignment"]
        if "data_num_samples" in alignment_block:
            flat_config["alignment_data_num_samples"] = alignment_block["data_num_samples"]
        if "methods" in alignment_block:
            flat_config["alignment_methods"] = alignment_block["methods"]
    
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

    # Handle CNN-specific settings
    cnn_block = nested_config.get("cnn", {})
    if isinstance(cnn_block, dict):
        if "mode" in cnn_block:
            flat_config["cnn_rq_aggregation_op"] = cnn_block["mode"]
    # Also check layer_config block (backward compatibility)
    layer_config_block = nested_config.get("layer_config", {})
    if isinstance(layer_config_block, dict):
        if "cnn_mode" in layer_config_block:
            flat_config["cnn_rq_aggregation_op"] = layer_config_block["cnn_mode"]

    # Map pruning settings
    if "pruning_settings" in nested_config:
        pruning = nested_config["pruning_settings"]
        flat_config["exclude_classification_layer"] = pruning.get("exclude_classification_layer", True)

    # Map other settings (experiment-level overrides take precedence)
    flat_config["device"] = experiment_block.get("device", nested_config.get("device", "cuda"))
    flat_config["seed"] = experiment_block.get("seed", nested_config.get("seed", 42))
    # num_networks: experiment > training > top-level > default (1)
    if "num_networks" not in flat_config:
        flat_config["num_networks"] = experiment_block.get(
            "num_networks", nested_config.get("num_networks", 1)
        )
    
    # Composite weights: metrics.composite_weights > alignment.composite_weights > top-level
    if "alignment_composite_weights" not in flat_config:
        flat_config["alignment_composite_weights"] = nested_config.get("alignment_composite_weights", {})
    if "alignment" in nested_config and isinstance(nested_config["alignment"], dict):
        alignment_block = nested_config["alignment"]
        if "composite_weights" in alignment_block:
            flat_config["alignment_composite_weights"] = alignment_block["composite_weights"]

    flat_config["supernode_config"] = nested_config.get("supernode", {})

    # Map supernode-related nested configs directly
    flat_config["supernode"] = nested_config.get("supernode", {})
    flat_config["supernode_robustness"] = nested_config.get("supernode_robustness", {})
    flat_config["supernode_summary"] = nested_config.get("supernode_summary", {})
    flat_config["halo_analysis"] = nested_config.get("halo_analysis", {})
    flat_config["generalized_importance"] = nested_config.get("generalized_importance", {})
    
    # Map flags for these analyses
    if "halo_analysis" in nested_config and nested_config["halo_analysis"].get("enabled", False):
        flat_config["do_halo_analysis"] = True
    if "generalized_importance" in nested_config and nested_config["generalized_importance"].get("enabled", False):
        flat_config["do_generalized_importance"] = True

    # Map pruning configuration (check both top-level and nested 'pruning' block)
    pruning_block = nested_config.get("pruning", {})
    if not isinstance(pruning_block, dict):
        pruning_block = {}

    # Backward-compatibility: allow single-string keys used in older configs/template
    if "strategy" in pruning_block and "algorithms" not in pruning_block:
        strategy_value = pruning_block["strategy"]
        if isinstance(strategy_value, str):
            pruning_block["algorithms"] = [strategy_value]
        else:
            pruning_block["algorithms"] = list(strategy_value)
    if "target_sparsity" in pruning_block and "sparsity_levels" not in pruning_block:
        target_value = pruning_block["target_sparsity"]
        if isinstance(target_value, (int, float)):
            pruning_block["sparsity_levels"] = [float(target_value)]
        else:
            pruning_block["sparsity_levels"] = [float(v) for v in target_value]
    if "scoring" in pruning_block and "alignment_metric" not in pruning_block:
        pruning_block["alignment_metric"] = pruning_block["scoring"]
    if "direction" in pruning_block and "selection_modes" not in pruning_block:
        direction_value = pruning_block["direction"]
        pruning_block["selection_modes"] = [direction_value] if isinstance(direction_value, str) else list(direction_value)
    if "structured" in pruning_block:
        flat_config["alignment_structured_pruning"] = pruning_block["structured"]

    fine_tune_block = pruning_block.get("fine_tune")
    if isinstance(fine_tune_block, dict):
        if "enabled" in fine_tune_block:
            flat_config["fine_tune_after_pruning"] = fine_tune_block.get("enabled", True)
        if "epochs" in fine_tune_block:
            flat_config["fine_tune_epochs"] = fine_tune_block["epochs"]
        if "learning_rate" in fine_tune_block:
            flat_config["fine_tune_learning_rate"] = fine_tune_block["learning_rate"]

    # Map top-level analysis flags
    flat_config["do_pruning_experiments"] = pruning_block.get("enabled", nested_config.get("do_pruning_experiments", False))
    flat_config["do_dropout_analysis"] = nested_config.get("dropout", {}).get("enabled", nested_config.get("do_dropout_analysis", False))
    flat_config["do_eigenfeature_analysis"] = nested_config.get("do_eigenfeature_analysis", False)

    # Map pruning parameters (prioritize nested pruning block, fallback to top-level)
    # Pruning uses metrics from metrics.enabled as scoring criteria.
    # Random selection is handled via selection_modes, not as a separate strategy.
    if "algorithms" in pruning_block:
        # Backward compatibility: explicit algorithms list
        flat_config["pruning_strategies"] = pruning_block["algorithms"]
    elif flat_config.get("metrics"):
        # Use computed metrics as pruning strategies
        flat_config["pruning_strategies"] = list(flat_config["metrics"])
    else:
        # Fallback default
        flat_config["pruning_strategies"] = nested_config.get("pruning_strategies", ["rayleigh_quotient"])
    
    flat_config["pruning_amounts"] = pruning_block.get("sparsity_levels", nested_config.get("pruning_amounts", [0.1, 0.3, 0.5, 0.7, 0.9]))
    selection_modes = pruning_block.get("selection_modes", nested_config.get("pruning_selection_mode", "low"))
    flat_config["pruning_selection_mode"] = selection_modes
    # Only set fine_tune defaults if not already set from fine_tune block above
    if "fine_tune_after_pruning" not in flat_config:
        flat_config["fine_tune_after_pruning"] = pruning_block.get("fine_tune_after_pruning", nested_config.get("fine_tune_after_pruning", True))
    if "fine_tune_epochs" not in flat_config:
        flat_config["fine_tune_epochs"] = pruning_block.get("fine_tune_epochs", nested_config.get("fine_tune_epochs", 5))
    # Deprecated: alignment_metric is no longer needed since we prune by all metrics
    flat_config["pruning_alignment_metric"] = pruning_block.get(
        "alignment_metric", nested_config.get("pruning_alignment_metric", "rayleigh_quotient")
    )
    flat_config["dependency_aware_pruning"] = pruning_block.get(
        "dependency_aware", nested_config.get("dependency_aware_pruning", False)
    )
    
    # Single-layer pruning: specify a layer name to prune only that layer
    flat_config["pruning_target_layer"] = pruning_block.get(
        "target_layer", nested_config.get("pruning_target_layer", None)
    )

    # Performance settings (all optimizations enabled by default)
    # Check both old "optimization" block and new "performance" block
    perf_block = nested_config.get("performance", nested_config.get("optimization", {}))
    if isinstance(perf_block, dict):
        if "eval_batches" in perf_block:
            flat_config["eval_batches"] = perf_block["eval_batches"]
    
    # These are always enabled (no longer configurable)
    flat_config["use_tensorized_training"] = True
    flat_config["use_tensorized_pruning"] = True
    flat_config["use_ultra_parallel_eval"] = True

    # Evaluation settings
    flat_config["do_perplexity_computation"] = nested_config.get("do_perplexity_computation", False)
    flat_config["evaluation_dataset"] = nested_config.get("evaluation_dataset", "wikitext")
    flat_config["evaluation_num_samples"] = nested_config.get("evaluation_num_samples", 100)
    
    # Directed redundancy and connectivity pruning flags
    flat_config["do_directed_redundancy"] = nested_config.get("do_directed_redundancy", True)
    flat_config["do_connectivity_pruning"] = nested_config.get("do_connectivity_pruning", True)

    # SCAR metrics (LLM-specific) - check both old top-level and new llm block
    flat_config["do_scar_metrics"] = nested_config.get("do_scar_metrics", False)
    flat_config["scar_num_samples"] = nested_config.get("scar_num_samples", 0)
    flat_config["scar_max_length"] = nested_config.get("scar_max_length", 512)
    
    # Handle new llm block (cleaner format)
    llm_block = nested_config.get("llm", {})
    if isinstance(llm_block, dict):
        if "scar_metrics" in llm_block:
            flat_config["do_scar_metrics"] = llm_block["scar_metrics"]
        if "scar_num_samples" in llm_block:
            flat_config["scar_num_samples"] = llm_block["scar_num_samples"]
        if "scar_max_length" in llm_block:
            flat_config["scar_max_length"] = llm_block["scar_max_length"]
        if "evaluate_perplexity" in llm_block:
            flat_config["do_perplexity_computation"] = llm_block["evaluate_perplexity"]
        if "evaluation_dataset" in llm_block:
            flat_config["evaluation_dataset"] = llm_block["evaluation_dataset"]
        if "evaluation_num_samples" in llm_block:
            flat_config["evaluation_num_samples"] = llm_block["evaluation_num_samples"]
        if "evaluation_metrics" in llm_block:
            flat_config["evaluation_metrics"] = llm_block["evaluation_metrics"]
        # Few-shot and chain-of-thought settings (NVIDIA Minitron compatible)
        if "use_nvidia_fewshot" in llm_block:
            flat_config["use_nvidia_fewshot"] = llm_block["use_nvidia_fewshot"]
        if "use_chain_of_thought" in llm_block:
            flat_config["use_chain_of_thought"] = llm_block["use_chain_of_thought"]
        if "fewshot_settings" in llm_block:
            flat_config["fewshot_settings"] = llm_block["fewshot_settings"]
        # Preserve the entire llm block for direct access
        flat_config["llm"] = llm_block
    
    # Also check evaluation block (backward compatibility)
    eval_block = nested_config.get("evaluation", {})
    if isinstance(eval_block, dict):
        if "do_perplexity_computation" in eval_block:
            flat_config["do_perplexity_computation"] = eval_block["do_perplexity_computation"]
        if "evaluation_dataset" in eval_block:
            flat_config["evaluation_dataset"] = eval_block["evaluation_dataset"]
        if "evaluation_num_samples" in eval_block:
            flat_config["evaluation_num_samples"] = eval_block["evaluation_num_samples"]

    # Map visualization settings
    flat_config["generate_plots"] = nested_config.get("generate_plots", True)
    flat_config["plot_format"] = nested_config.get("plot_format", "png")
    flat_config["plot_dpi"] = nested_config.get("plot_dpi", 300)

    # Support optional nested visualization block for clearer configs
    viz_block = nested_config.get("visualization", {})
    if isinstance(viz_block, dict):
        # Mirror common options onto existing ExperimentConfig fields
        if "enabled" in viz_block:
            flat_config["generate_plots"] = viz_block.get("enabled", flat_config["generate_plots"])
        if "format" in viz_block:
            flat_config["plot_format"] = viz_block.get("format", flat_config["plot_format"])
        if "dpi" in viz_block:
            flat_config["plot_dpi"] = viz_block.get("dpi", flat_config["plot_dpi"])

        # Keep the full block around for experiment-specific logic
        flat_config["visualization_options"] = viz_block

    # Optional nested analysis block for higher-level analysis configuration
    analysis_block = nested_config.get("analysis", {})
    if isinstance(analysis_block, dict):
        flat_config["analysis_options"] = analysis_block
        if "generate_plots" in analysis_block:
            flat_config["generate_plots"] = analysis_block.get("generate_plots", flat_config.get("generate_plots", True))
    
    # Post-experiment analysis configuration (for AnalysisRunner)
    post_analysis_block = nested_config.get("post_analysis", {})
    if isinstance(post_analysis_block, dict) and post_analysis_block:
        flat_config["post_analysis"] = post_analysis_block

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
