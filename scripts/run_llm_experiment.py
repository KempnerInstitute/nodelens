#!/usr/bin/env python3
"""
Main entrypoint to run LLM alignment experiments.

Usage:
    python run_experiment.py --config configs/llm_alignment.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import yaml
import torch

from alignment.experiments.llm_experiments import LLMAlignmentExperiment
from alignment.experiments.base import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_experiment")


def parse_args():
    parser = argparse.ArgumentParser(description="Run an LLM alignment experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda).")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and flatten YAML experiment configuration for compatibility with BaseExperiment."""
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Flatten nested sections into a single dictionary like ExperimentConfig expects
    config = {}

    # Top-level experiment metadata
    if "experiment" in config_data:
        config.update(config_data["experiment"])

    # Model parameters
    if "model" in config_data:
        model_cfg = config_data["model"]
        config["model_name"] = model_cfg.get("model_name", model_cfg.get("name", "hf_causal_lm"))
        config["model_config"] = model_cfg
        config["pretrained"] = model_cfg.get("pretrained", True)
        config["model_id"] = model_cfg.get("model_id", None)

    # Dataset parameters
    if "dataset" in config_data:
        ds_cfg = config_data["dataset"]
        config["dataset_name"] = ds_cfg.get("dataset_name", ds_cfg.get("name", "wikitext"))
        config["dataset_config"] = ds_cfg
        config["data_path"] = ds_cfg.get("data_path", None)

    # Alignment metrics
    if "alignment" in config_data:
        align_cfg = config_data["alignment"]
        config["metrics"] = align_cfg.get("metrics", ["rayleigh_quotient"])
        config["alignment_methods"] = align_cfg.get("metrics", ["rayleigh_quotient"])
        config["metric_configs"] = align_cfg.get("metric_configs", {})

    # Pruning parameters
    if "pruning" in config_data:
        prune_cfg = config_data["pruning"]
        config["do_pruning_experiments"] = prune_cfg.get("enabled", False)
        config["pruning_strategies"] = prune_cfg.get("algorithms", ["alignment"])
        config["pruning_amounts"] = prune_cfg.get("sparsity_levels", [0.1, 0.3])
        config["pruning_selection_mode"] = prune_cfg.get("mode", "low")
        config["pruning_alignment_metric"] = prune_cfg.get("alignment_metric", "rayleigh_quotient")

    # Evaluation (perplexity, etc.)
    if "evaluation" in config_data:
        config["evaluation"] = config_data["evaluation"]

    # Importance computation
    if "importance_computation" in config_data:
        config["importance_computation"] = config_data["importance_computation"]

    # Wrapper / tracked layers
    if "wrapper" in config_data:
        config["tracked_layers"] = config_data["wrapper"].get("tracked_layers", None)

    # Generic defaults
    config.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    config.setdefault("batch_size", 1)
    config.setdefault("num_workers", 0)

    # Remove keys that are not accepted by ExperimentConfig / LLMAlignmentConfig
    unsupported_keys = [
        "evaluation",  # not a constructor field
        "importance_computation",
        "alignment_config",
        "pruning_config",
    ]
    for key in unsupported_keys:
        if key in config:
            del config[key]

    return config


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    # Override device if specified
    if args.device:
        config["device"] = args.device

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Using device: {config['device']}")

    # Initialize and run experiment
    experiment = LLMAlignmentExperiment(config)
    experiment.setup()
    results = experiment.run()

    # Save results
    output_dir = Path(config.get("log_dir", "./logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{config.get('name', 'llm_alignment')}_results.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Experiment completed. Results saved to {result_path}")


if __name__ == "__main__":
    main()
