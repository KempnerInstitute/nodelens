#!/usr/bin/env python3
"""
Unified Alignment Experiment Runner

Run alignment experiments with configuration files.

Usage:
    python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
    python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml --device cuda:0
    python scripts/run_experiment.py --analysis-only --experiment-dir results/my_experiment_20240101
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

from alignment.experiments.general_alignment import GeneralAlignmentExperiment
from alignment.pruning.experiments.cascading_layer import CascadingLayerPruningExperiment
from alignment.pruning.experiments.layer_wise import LayerIsolatedPruningExperiment
from alignment.experiments.llm_experiments import LLMAlignmentExperiment

logger = logging.getLogger(__name__)


def run_post_analysis(config, results_file: Path, output_dir: Path):
    """Run post-experiment analysis using AnalysisRunner."""
    post_analysis_config = getattr(config, "post_analysis", {})
    if not post_analysis_config:
        return
    
    logger.info("Running post-experiment analysis...")
    
    try:
        from alignment.analysis import AnalysisRunner, AnalysisConfig
        
        # Build analysis config from post_analysis block
        analysis_config = AnalysisConfig(
            results_file=str(results_file),
            output_dir=str(output_dir / "analysis"),
            style=post_analysis_config.get("style", "seaborn-v0_8-paper"),
            format=post_analysis_config.get("format", config.plot_format),
            dpi=post_analysis_config.get("dpi", config.plot_dpi),
            analyses=post_analysis_config.get("analyses", ["all"]),
            histograms=post_analysis_config.get("histograms", {}),
            scatter_plots=post_analysis_config.get("scatter_plots", {}),
            heatmaps=post_analysis_config.get("heatmaps", {}),
            pruning_curves=post_analysis_config.get("pruning_curves", {}),
            layer_distributions=post_analysis_config.get("layer_distributions", {}),
            scar_analysis=post_analysis_config.get("scar_analysis", {}),
        )
        
        runner = AnalysisRunner(analysis_config)
        outputs = runner.run()
        
        total_files = sum(len(v) for v in outputs.values())
        logger.info(f"Post-analysis complete: generated {total_files} files in {output_dir / 'analysis'}")
        
    except Exception as e:
        logger.error(f"Post-analysis failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Alignment Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Load existing experiment and regenerate analysis/plots",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to existing experiment directory (with --analysis-only)",
    )

    args, unknown = parser.parse_known_args()

    # Parse overrides
    overrides = {}
    if args.device:
        overrides["device"] = args.device
    if args.seed:
        overrides["seed"] = args.seed

    # Load config
    from alignment.configs.config_loader import load_config as proper_load_config
    config = proper_load_config(args.config)

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    is_analysis_only = bool(args.analysis_only)

    if is_analysis_only:
        if not args.experiment_dir:
            raise ValueError("--analysis-only requires --experiment-dir")
        output_dir = Path(args.experiment_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {output_dir}")

        config.experiment_dir = str(output_dir)
        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        plots_dir = output_dir / "plots"
        config.plots_dir = str(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = output_dir / "experiment_config.yaml"
        timestamp = None
    else:
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = getattr(config, "name", "experiment")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"results/{experiment_name}_{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)

        config_save_path = output_dir / "experiment_config.yaml"
        config.save(config_save_path)

        config.checkpoint_dir = str(output_dir / "checkpoints")
        config.log_dir = str(output_dir / "logs")
        config.experiment_dir = str(output_dir)

        plots_dir = output_dir / "plots"
        config.plots_dir = str(plots_dir)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Print experiment info
    print(f"\n{'='*60}")
    print("Alignment Experiment" + (" (Analysis Only)" if is_analysis_only else ""))
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Experiment directory: {output_dir}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")

    # Determine experiment type
    experiment_type = getattr(config, "experiment_type", "alignment_analysis")
    logger.info(f"Running {experiment_type} experiment")

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

    # Analysis-only mode
    if is_analysis_only:
        if isinstance(experiment, GeneralAlignmentExperiment):
            result_files = sorted(output_dir.glob("results_*.json"))
            if not result_files:
                raise FileNotFoundError(f"No results_*.json found in {output_dir}")
            results_path = result_files[-1]
            with results_path.open("r") as f:
                results = json.load(f)

            experiment.train_results = results.get("train_results", {})
            experiment.test_results = results.get("test_results", {})
            experiment.dropout_results = results.get("dropout_results", {})
            experiment.pruning_results = results.get("pruning_results", {})
            experiment.eigenfeature_results = results.get("eigenfeature_results", {})

            if getattr(config, "generate_plots", True):
                experiment._generate_visualizations()
                logger.info("Regenerated visualizations from existing results")
            
            # Run post-analysis if configured
            run_post_analysis(config, results_path, output_dir)
        else:
            logger.warning(f"Analysis-only mode not supported for {experiment_type}")

        print(f"\n{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")
        return

    # Full experiment run
    results = experiment.run()

    # Save results
    results_file = output_dir / f"results_{timestamp}.json"

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

    # Run post-analysis if configured
    run_post_analysis(config, results_file, output_dir)

    # Print completion
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")

    if "test_results" in results:
        print(f"Final accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%")

    print(f"\nResults saved in: {output_dir}")
    print(f"  - Configuration: {config_save_path}")
    print(f"  - Results: {results_file}")
    print(f"  - Plots: {plots_dir}/")
    
    # Check for analysis output
    analysis_dir = output_dir / "analysis"
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.rglob("*"))
        print(f"  - Analysis ({len([f for f in analysis_files if f.is_file()])} files): {analysis_dir}/")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
