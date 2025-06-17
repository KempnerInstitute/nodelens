#!/usr/bin/env python3
"""
Example script showing how to run alignment experiments using configuration files.

This script demonstrates:
1. Loading configuration from YAML files
2. Running different types of experiments
3. Using command-line overrides
4. Saving and visualizing results
"""

import argparse
import sys
from pathlib import Path
import logging
import torch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from alignment_refactor.configs import load_config, validate_config
from alignment_refactor.experiments import (
    ProgressiveDropoutExperiment,
    EigenvectorAlignment,
    LayerIsolatedPruning,
    ExperimentConfig
)
from alignment_refactor.models.architectures.standard_models import MLP, CNN2P2
from alignment_refactor.analysis.visualizers import MetricVisualizer


def setup_logging(verbose=True):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_model_from_config(config: ExperimentConfig):
    """Create model based on configuration."""
    model_name = config.model_name
    model_config = config.model_config
    
    if model_name == "mlp":
        return MLP(**model_config)
    elif model_name == "cnn2p2":
        return CNN2P2(**model_config)
    else:
        # Try torchvision
        try:
            import torchvision.models as models
            if hasattr(models, model_name):
                model_fn = getattr(models, model_name)
                return model_fn(pretrained=config.pretrained, **model_config)
        except ImportError:
            pass
        
        raise ValueError(f"Unknown model: {model_name}")


def select_experiment_class(config: ExperimentConfig):
    """Select appropriate experiment class based on config."""
    # Check which experiment-specific options are present
    if hasattr(config, 'dropout_fractions') and config.dropout_fractions:
        return ProgressiveDropoutExperiment
    elif hasattr(config, 'num_components'):
        return EigenvectorAlignment
    elif hasattr(config, 'pruning_percentages') and config.pruning_percentages:
        return LayerIsolatedPruning
    else:
        # Default to progressive dropout
        return ProgressiveDropoutExperiment


def run_experiment(config_path: str, overrides: dict = None):
    """
    Run experiment from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        overrides: Dictionary of configuration overrides
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)
    
    # Validate configuration
    errors = validate_config(config.to_dict())
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Create model
    print(f"Creating model: {config.model_name}")
    model = create_model_from_config(config)
    
    # Select experiment class
    experiment_class = select_experiment_class(config)
    print(f"Running experiment type: {experiment_class.__name__}")
    
    # Create and run experiment
    experiment = experiment_class(model=model, config=config)
    
    # Train model if requested
    if config.train_before_dropout:
        print(f"Training model for {config.training_epochs} epochs...")
        # Note: Actual training would require dataset loading and training loop
        # This is a placeholder
        print("Training complete (placeholder)")
    
    # Run experiment
    print("Running experiment...")
    results = experiment.run()
    
    # Save results
    experiment.save_results()
    print(f"Results saved to: {config.log_dir}")
    
    # Create visualizations if requested
    if hasattr(config, 'plotting') and config.plotting.get('save_plots', True):
        print("Creating visualizations...")
        visualizer = MetricVisualizer()
        
        # Create plots based on experiment type
        if isinstance(experiment, ProgressiveDropoutExperiment):
            # Plot metric vs dropout
            for metric_name in config.metrics:
                if metric_name in results:
                    visualizer.plot_metric_vs_dropout(
                        results[metric_name],
                        metric_name,
                        save_path=Path(config.log_dir) / f"{metric_name}_vs_dropout.png"
                    )
        
        print("Visualizations saved")
    
    return results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Run alignment experiments from configuration files"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training phase"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Build overrides
    overrides = {}
    if args.device:
        overrides['device'] = args.device
    if args.batch_size:
        overrides['batch_size'] = args.batch_size
    if args.epochs:
        overrides['training_epochs'] = args.epochs
    if args.seed:
        overrides['seed'] = args.seed
    if args.output_dir:
        overrides['log_dir'] = args.output_dir
        overrides['checkpoint_dir'] = args.output_dir
    if args.no_train:
        overrides['train_before_dropout'] = False
    
    # Run experiment
    run_experiment(args.config, overrides)


if __name__ == "__main__":
    main() 