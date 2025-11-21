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

# Add the project root and src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "src"))

# Import from the alignment package
from alignment.experiments.general_alignment import GeneralAlignmentExperiment
from alignment.pruning.experiments.cascading_layer import CascadingLayerPruningExperiment
from alignment.pruning.experiments.layer_wise import LayerIsolatedPruningExperiment
from alignment.experiments.llm_experiments import LLMAlignmentExperiment

logger = logging.getLogger(__name__)


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

    # Determine experiment type
    experiment_type = getattr(config, "experiment_type", "alignment_analysis")
    logger.info(f"Running {experiment_type} experiment")
    logger.info(config)

    if experiment_type in {"llm_alignment", "llm_supernode", "llm"}:
        experiment = LLMAlignmentExperiment(config)
    elif experiment_type in {"alignment_analysis", "vision_synergy", "general_alignment"}:
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
