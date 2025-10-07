#!/usr/bin/env python3
"""
Unified Alignment Experiment Runner

A single entry point for all alignment experiments that can handle:
- Any dataset (MNIST, CIFAR, ImageNet, etc.)
- Any model (MLP, CNN, ResNet, etc.)
- Any metric (Rayleigh Quotient, MI, CKA, etc.)
- Any pruning strategy (magnitude, gradient, fisher, etc.)
- Any experiment type (standard, progressive, layer-wise, etc.)

Usage:
    python scripts/run_experiment.py --config configs/unified_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "src"))

# Import from the alignment package
from alignment.experiments.general_alignment import GeneralAlignmentExperiment
from alignment.pruning.experiments.cascading_layer import CascadingLayerPruningExperiment
from alignment.pruning.experiments.layer_wise import LayerIsolatedPruningExperiment

logger = logging.getLogger(__name__)


def load_config(config_path, overrides=None):
    """Load and merge configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    return config


def create_experiment_config(unified_config):
    """Convert unified config to experiment config object."""
    from alignment.experiments.base import ExperimentConfig
    from alignment.experiments.general_alignment import GeneralAlignmentConfig

    # Extract model config and handle different naming conventions
    model_config = unified_config.get("model", {})
    model_name = model_config.get("name", model_config.get("architecture", "mlp"))

    # Extract dataset config
    dataset_config = unified_config.get("dataset", {})
    if isinstance(dataset_config, str):
        # Handle case where dataset is just a string
        dataset_name = dataset_config
        dataset_config = {"name": dataset_name}
    else:
        dataset_name = dataset_config.get("name", dataset_config.get("dataset", "mnist"))

    # Build base parameters common to all experiment types
    base_params = {
        "name": unified_config.get("experiment_name", "unified_experiment"),
        "seed": unified_config.get("seed", 42),
        "device": unified_config.get("device", "cuda"),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "batch_size": dataset_config.get("batch_size", unified_config.get("data", {}).get("batch_size", 128)),
        "num_workers": dataset_config.get("num_workers", unified_config.get("data", {}).get("num_workers", 4)),
        "metrics": unified_config.get("alignment", {}).get("metrics", unified_config.get("analysis", {}).get("metrics", ["rayleigh_quotient"])),
    }

    # Build model config with proper parameter names
    model_kwargs = {}
    if model_name == "mlp":
        # Map hidden_sizes to hidden_dims
        model_kwargs["hidden_dims"] = model_config.get("hidden_dims", model_config.get("hidden_sizes", [512, 256]))
        model_kwargs["activation_type"] = model_config.get("activation_type", model_config.get("activation", "relu"))
        model_kwargs["dropout_rate"] = model_config.get("dropout_rate", 0.0)
        # Determine input/output dims based on dataset
        if dataset_name == "mnist":
            model_kwargs["input_dim"] = 784
            model_kwargs["output_dim"] = 10
        elif dataset_name == "cifar10":
            model_kwargs["input_dim"] = 3072
            model_kwargs["output_dim"] = 10
        elif dataset_name == "cifar100":
            model_kwargs["input_dim"] = 3072
            model_kwargs["output_dim"] = 100
        else:
            model_kwargs["input_dim"] = model_config.get("input_dim", 784)
            model_kwargs["output_dim"] = model_config.get("output_dim", 10)

    base_params["model_config"] = model_kwargs

    # Add training config if present
    training_config = unified_config.get("training", {})
    if training_config:
        base_params["training_epochs"] = training_config.get("epochs", 10)
        # Handle different optimizer config formats
        if isinstance(training_config.get("optimizer"), dict):
            base_params["learning_rate"] = training_config["optimizer"].get("learning_rate", 0.001)
            base_params["optimizer"] = training_config["optimizer"].get("type", "adam")
        else:
            base_params["learning_rate"] = training_config.get("learning_rate", 0.001)
            base_params["optimizer"] = training_config.get("optimizer", "adam")

    # Infer experiment type based on enabled blocks
    pruning_cfg = unified_config.get("pruning", {})
    dropout_cfg = unified_config.get("dropout", {})

    pruning_enabled = pruning_cfg.get("enabled", False)
    cascading_scope = pruning_cfg.get("scope", "layer") == "cascading"
    dropout_enabled = dropout_cfg.get("enabled", False)

    # Determine if we need specialized experiment types or GeneralAlignmentConfig
    if cascading_scope and pruning_enabled:
        # For cascading layer pruning, use base ExperimentConfig
        config = ExperimentConfig(**base_params)
        inferred_type = "cascading_layer_pruning"
    elif pruning_cfg.get("scope") == "layer_isolated" and pruning_enabled:
        # For layer isolated pruning, use base ExperimentConfig
        config = ExperimentConfig(**base_params)
        inferred_type = "layer_isolated_pruning"
    else:
        # For standard experiments, use GeneralAlignmentConfig
        # Add GeneralAlignmentConfig-specific parameters
        general_params = base_params.copy()
        general_params["num_networks"] = unified_config.get("num_networks", 1)
        general_params["aggregate_metrics"] = unified_config.get("aggregate_metrics", True)
        general_params["save_individual_networks"] = unified_config.get("save_individual_networks", False)
        general_params["save_checkpoints"] = unified_config.get("save_checkpoints", False)

        config = GeneralAlignmentConfig(**general_params)

        # Set flags based on enabled blocks
        config.do_train = training_config.get("epochs", training_config.get("do_train", 0)) > 0
        config.do_dropout_analysis = dropout_enabled
        config.do_pruning_experiments = pruning_enabled

        # Determine inferred type for logging
        if pruning_enabled:
            inferred_type = "standard_pruning"
        elif dropout_enabled:
            inferred_type = "progressive_dropout"
        else:
            inferred_type = "alignment_analysis"

        # Handle plot generation
        visualization_config = unified_config.get("visualization", {})
        output_config = unified_config.get("output", {})
        analysis_config = unified_config.get("analysis", {})

        # Default to True if not explicitly set to False
        generate_plots = True
        if "generate_plots" in visualization_config:
            generate_plots = visualization_config["generate_plots"]
        elif "generate_plots" in output_config:
            generate_plots = output_config["generate_plots"]
        elif "generate_plots" in analysis_config:
            generate_plots = analysis_config["generate_plots"]

        config.generate_plots = generate_plots
        logger.info(f"Plot generation enabled: {config.generate_plots}")

        # Pruning specific configuration
        if pruning_enabled:
            # Get algorithms
            algorithms = pruning_cfg.get("algorithms", ["magnitude"])
            config.pruning_strategies = algorithms if isinstance(algorithms, list) else [algorithms]

            # Get sparsity levels
            sparsity_levels = pruning_cfg.get("sparsity_levels", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
            config.pruning_amounts = sparsity_levels

            # Get selection modes
            selection_modes = pruning_cfg.get("selection_modes", pruning_cfg.get("selection_mode", ["low"]))
            config.pruning_selection_mode = selection_modes if isinstance(selection_modes, list) else [selection_modes]

            # Fine-tuning settings
            config.fine_tune_after_pruning = pruning_cfg.get("fine_tune_after_pruning", True)
            config.fine_tune_epochs = pruning_cfg.get("fine_tune_epochs", 5)
            config.fine_tune_learning_rate = pruning_cfg.get("fine_tune_learning_rate", 0.0001)

            # Scope settings
            scope = pruning_cfg.get("scope", "layer")
            config.pruning_scope = scope

            # Alignment metric settings
            config.pruning_alignment_metric = pruning_cfg.get("alignment_metric", "rayleigh_quotient")
            config.pruning_hybrid_alpha = pruning_cfg.get("hybrid_alpha", 0.5)

            # Ultra-parallel evaluation settings
            config.use_ultra_parallel_eval = pruning_cfg.get("use_ultra_parallel_eval", False)
            config.eval_batches = pruning_cfg.get("eval_batches", None)

            logger.info(
                f"Pruning enabled: algorithms={config.pruning_strategies}, levels={config.pruning_amounts}, modes={config.pruning_selection_mode}"
            )
            logger.info(f"Ultra-parallel eval: {config.use_ultra_parallel_eval}, eval_batches: {config.eval_batches}")

        # Dropout specific configuration
        if dropout_enabled:
            config.dropout_rates = dropout_cfg.get("rates", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
            logger.info(f"Dropout enabled: rates={config.dropout_rates}")

    # Common configuration for all experiment types
    if not isinstance(config, GeneralAlignmentConfig):
        # Ensure plot generation is enabled for other experiment types too
        visualization_config = unified_config.get("visualization", {})
        output_config = unified_config.get("output", {})
        config.generate_plots = visualization_config.get("generate_plots", output_config.get("generate_plots", True))

    # Add additional attributes for compatibility
    config.training_config = training_config
    config.train_model = config.training_config.get("epochs", 0) > 0
    config.alignment_metrics = config.metrics
    config.apply_pruning = pruning_enabled

    # Legacy support for pruning_analysis and network_compression blocks
    pruning_analysis = unified_config.get("pruning_analysis", {})
    network_compression = unified_config.get("network_compression", {})

    # Use the appropriate config based on what's enabled
    if pruning_analysis.get("enabled", False):
        active_pruning_config = pruning_analysis
        config.pruning_strategy = active_pruning_config.get("algorithms", ["magnitude"])[0]
    elif network_compression.get("enabled", False):
        active_pruning_config = network_compression
        config.pruning_strategy = active_pruning_config.get("algorithms", ["magnitude"])[0]
    else:
        # Fallback to pruning config or empty
        active_pruning_config = pruning_cfg if pruning_enabled else {}
        config.pruning_strategy = pruning_cfg.get("algorithms", ["magnitude"])[0] if pruning_enabled else "magnitude"

    config.pruning_config = active_pruning_config
    config.analysis_config = unified_config.get("visualization", unified_config.get("analysis", {}))
    config.eval_model = True
    config.cnn_mode = model_config.get("cnn_mode", "unfold")

    # Set dropout rates (handle multiple sources)
    if hasattr(config, "dropout_rates"):
        # Already set for GeneralAlignmentConfig
        pass
    elif pruning_analysis.get("enabled", False):
        config.dropout_rates = pruning_analysis.get("dropout_rates", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    else:
        config.dropout_rates = dropout_cfg.get("rates", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])

    # Handle selection modes for compatibility
    if pruning_analysis.get("enabled", False):
        selection_mode = pruning_analysis.get("selection_strategies", ["low"])
    elif network_compression.get("enabled", False):
        selection_mode = [network_compression.get("selection_strategy", "low")]
    else:
        selection_mode = pruning_cfg.get("selection_modes", ["low"]) if pruning_enabled else ["low"]

    config.pruning_modes = selection_mode if isinstance(selection_mode, list) else [selection_mode]

    config.cascade_direction = unified_config.get("experiment_specific", {}).get("cascade_direction", "forward")
    config.recompute_scores = True

    # Note: Output directories are now set in main() after creating timestamped folders
    config.plot_dpi = unified_config.get("visualization", {}).get("plot_dpi", unified_config.get("output", {}).get("plot_dpi", 300))

    # Log the inferred experiment type
    logger.info(f"Inferred experiment type: {inferred_type} based on config blocks")
    config._inferred_experiment_type = inferred_type  # Store for later use

    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Alignment Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")

    args, unknown = parser.parse_known_args()

    # Parse additional overrides
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.seed:
        overrides["seed"] = args.seed

    # Load config using the proper config loader
    from alignment.configs.config_loader import load_config as proper_load_config

    config = proper_load_config(args.config)

    # Apply overrides to the loaded config
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = getattr(config, "name", "experiment")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a unique directory with experiment name and timestamp
        output_dir = Path(f"results/{experiment_name}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the configuration used
    config_save_path = output_dir / "experiment_config.yaml"
    config.save(config_save_path)

    # Update config with timestamped directories
    config.checkpoint_dir = str(output_dir / "checkpoints")
    config.log_dir = str(output_dir / "logs")
    config.experiment_dir = str(output_dir)  # Add experiment_dir for compatibility

    # Create plots directory in results folder (not in logs)
    plots_dir = output_dir / "plots"
    config.plots_dir = str(plots_dir)  # Add plots_dir for visualization

    # Ensure directories exist
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created plots directory: {plots_dir}")

    # Setup logging to both file and console
    log_file = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Print experiment info
    print(f"\n{'='*60}")
    print("Running Alignment Experiment")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config.device}")
    print(f"Plot generation: {getattr(config, 'generate_plots', True)}")
    print(f"Plots directory: {plots_dir}")
    print(f"{'='*60}\n")

    # Use the inferred experiment type from config
    experiment_type = getattr(config, "_inferred_experiment_type", "alignment_analysis")

    logger.info(f"Running {experiment_type} experiment")

    # Create experiment based on inferred type
    if experiment_type in ["standard_pruning", "progressive_dropout", "alignment_analysis"]:
        experiment = GeneralAlignmentExperiment(config)
    elif experiment_type == "layer_isolated_pruning":
        experiment = LayerIsolatedPruningExperiment(config)
    elif experiment_type == "cascading_layer_pruning":
        experiment = CascadingLayerPruningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Run experiment
    results = experiment.run()

    # Save results with timestamp
    results_file = output_dir / f"results_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Create experiment summary
    summary_file = output_dir / "experiment_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Experiment Type: {experiment_type} (inferred from config blocks)\n")
        f.write(f"Plot Generation: {getattr(config, 'generate_plots', True)}\n")
        f.write("=" * 50 + "\n\n")

        # Add results summary
        if "test_results" in results:
            f.write("Final Model Performance:\n")
            f.write(f"  - Accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%\n")
            f.write(f"  - Loss: {results['test_results'].get('final_loss', 'N/A'):.4f}\n\n")

        if "pruning_results" in results and results["pruning_results"]:
            f.write("Pruning Experiments:\n")
            strategies = results["pruning_results"].get("strategies", {})
            f.write(f"  - Strategies tested: {list(strategies.keys())}\n")
            f.write(f"  - Plots saved in: {config.log_dir}/plots/\n")

        # List plots created
        plots_created = list(plots_dir.glob("*"))
        if plots_created:
            f.write(f"\nPlots Generated ({len(plots_created)}):\n")
            for plot_file in sorted(plots_created):
                if plot_file.is_file():
                    f.write(f"  - {plot_file.name}\n")
        else:
            f.write("\nNo plots were generated.\n")

        f.write("\nGenerated Files:\n")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_dir)
                f.write(f"  - {relative_path}\n")

    # Print completion message
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")

    if "test_results" in results:
        print(f"Final model accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%")
        print(f"Final model loss: {results['test_results'].get('final_loss', 'N/A'):.4f}")

    print(f"\nAll results saved in: {output_dir}")
    print(f"  - Configuration: {config_save_path}")
    print(f"  - Results: {results_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Logs: {log_file}")

    # Check and report on plots
    plots_created = list(plots_dir.glob("*"))
    if plots_created:
        print(f"  - Plots ({len(plots_created)}): {plots_dir}/")
        for plot_file in sorted(plots_created):
            if plot_file.is_file():
                print(f"    * {plot_file.name}")
    else:
        print("  - No plots generated (check generate_plots setting and experiment configuration)")
        print(f"    * Pruning enabled: {getattr(config, 'do_pruning_experiments', False)}")
        print(f"    * Plot generation: {getattr(config, 'generate_plots', False)}")
        print(f"    * Pruning results in output: {'pruning_results' in results}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
