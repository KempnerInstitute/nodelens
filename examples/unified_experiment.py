#!/usr/bin/env python3
"""
Unified Alignment Experiment

This is the main experiment script that can run any configuration specified
in the master config file. It supports all features of the alignment framework:
- Multiple model architectures
- Various datasets
- All alignment metrics
- All pruning strategies
- Comprehensive analysis and visualization

Usage:
    python unified_experiment.py --config configs/master_config.yaml
    
    # Override specific parameters:
    python unified_experiment.py --config configs/master_config.yaml \
        --model_name resnet50 \
        --dataset_name cifar100 \
        --training_config.epochs 200
"""

import argparse
import logging
import yaml
import json
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.analysis import HTMLReporter, MarkdownReporter
from alignment.analysis.visualization import (
    MetricVisualizer,
    PruningVisualizer,
    AlignmentVisualizer
)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(config.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific log directory
    exp_name = config.get("name", "experiment")
    timestamp = datetime.now().strftime(config.get("timestamp_format", "%Y%m%d_%H%M%S"))
    if config.get("use_timestamp", True):
        log_subdir = log_dir / f"{exp_name}_{timestamp}"
    else:
        log_subdir = log_dir / exp_name
    log_subdir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = log_subdir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    
    # Save config for reference
    config_file = log_subdir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return logger, log_subdir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Alignment Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Allow overriding any config parameter from command line
    # Examples of common overrides:
    parser.add_argument("--name", type=str, help="Override experiment name")
    parser.add_argument("--model_name", type=str, help="Override model architecture")
    parser.add_argument("--dataset_name", type=str, help="Override dataset")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    
    # Training overrides
    parser.add_argument("--training_config.epochs", type=int, dest="epochs",
                       help="Override number of epochs")
    parser.add_argument("--training_config.batch_size", type=int, dest="batch_size",
                       help="Override batch size")
    parser.add_argument("--training_config.learning_rate", type=float, dest="learning_rate",
                       help="Override learning rate")
    
    # Pruning overrides
    parser.add_argument("--pruning_strategy", type=str, help="Override pruning strategy")
    parser.add_argument("--pruning_config.amount", type=float, dest="pruning_amount",
                       help="Override pruning amount")
    
    # Workflow overrides
    parser.add_argument("--train_model", type=lambda x: x.lower() == 'true',
                       help="Whether to train the model")
    parser.add_argument("--apply_pruning", type=lambda x: x.lower() == 'true',
                       help="Whether to apply pruning")
    
    return parser.parse_args()


def load_and_merge_config(args) -> Dict[str, Any]:
    """Load config from file and merge with command line overrides."""
    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    overrides = vars(args)
    for key, value in overrides.items():
        if value is not None and key != 'config':
            # Handle nested keys (e.g., training_config.epochs)
            if '.' in key:
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                # Handle special mappings for common overrides
                if key == 'epochs' and 'training_config' in config:
                    config['training_config']['epochs'] = value
                elif key == 'batch_size' and 'training_config' in config:
                    config['training_config']['batch_size'] = value
                elif key == 'learning_rate' and 'training_config' in config:
                    config['training_config']['learning_rate'] = value
                elif key == 'pruning_amount' and 'pruning_config' in config:
                    config['pruning_config']['amount'] = value
                else:
                    config[key] = value
    
    return config


def create_experiment_config(config: Dict[str, Any]) -> GeneralAlignmentConfig:
    """Create GeneralAlignmentConfig from master config dictionary."""
    # Extract the parameters that GeneralAlignmentConfig actually accepts
    experiment_config = GeneralAlignmentConfig(
        # Basic info from ExperimentConfig
        name=config.get('name', 'unified_experiment'),
        model_name=config.get('model_name', 'resnet18'),
        model_config=config.get('model_config', {}),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_dir=config.get('log_dir', './logs'),
        device=config.get('training_config', {}).get('device', 'cuda'),
        seed=config.get('seed', 42),
        
        # Dataset configuration
        dataset_name=config.get('dataset_name', 'cifar10'),
        dataset_config=config.get('dataset_config', {}),
        
        # Training configuration
        training_config=config.get('training_config', {}),
        
        # Metrics configuration
        alignment_metrics=config.get('alignment_metrics', ['rayleigh_quotient']),
        compute_metrics_on=config.get('compute_metrics_on'),  # None means all layers
        
        # Pruning configuration
        pruning_strategy=config.get('pruning_strategy', 'magnitude'),
        pruning_config=config.get('pruning_config', {}),
        pruning_based_on_metric=config.get('pruning_based_on_metric'),
        
        # Experiment flow
        train_model=config.get('train_model', True),
        compute_initial_metrics=config.get('compute_initial_metrics', True),
        apply_pruning=config.get('apply_pruning', True),
        fine_tune_after_pruning=config.get('fine_tune_after_pruning', True),
        fine_tune_epochs=config.get('pruning_config', {}).get('fine_tune_epochs', 10),
        
        # Analysis configuration
        track_performance=config.get('analysis_config', {}).get('save_predictions', True),
        save_checkpoints=config.get('analysis_config', {}).get('save_weights', True),
        save_metrics_history=config.get('analysis_config', {}).get('save_activations', True),
    )
    
    # Store additional config parameters that might be used elsewhere
    experiment_config._full_config = config
    
    return experiment_config


def generate_comprehensive_report(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate comprehensive analysis report with visualizations."""
    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive analysis...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizers
    metric_viz = MetricVisualizer()
    pruning_viz = PruningVisualizer()
    alignment_viz = AlignmentVisualizer()
    
    # 1. Generate metric visualizations
    if 'initial_metrics' in results or 'final_metrics' in results:
        logger.info("Creating metric visualizations...")
        
        # Plot each metric
        for metric_name in config.get('alignment_metrics', []):
            if 'initial_metrics' in results and metric_name in results['initial_metrics']:
                # Plot layer comparison for initial metrics
                initial_data = results['initial_metrics'][metric_name]
                fig = metric_viz.plot_layer_comparison(
                    initial_data,
                    title=f"{metric_name} - Initial Values by Layer",
                    ylabel=metric_name
                )
                if fig:
                    fig.savefig(
                        viz_dir / f"{metric_name}_initial.png",
                        dpi=config.get('analysis_config', {}).get('plot_dpi', 300),
                        bbox_inches='tight'
                    )
                    plt.close(fig)
                
                # If we have both initial and final, create comparison
                if 'final_metrics' in results and metric_name in results['final_metrics']:
                    final_data = results['final_metrics'][metric_name]
                    
                    # Create a combined plot showing before/after
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Initial values
                    layers = list(initial_data.keys())
                    initial_values = list(initial_data.values())
                    ax1.bar(range(len(layers)), initial_values, color='blue', alpha=0.7)
                    ax1.set_xticks(range(len(layers)))
                    ax1.set_xticklabels(layers, rotation=45, ha='right')
                    ax1.set_title(f"{metric_name} - Initial", fontsize=12)
                    ax1.set_ylabel("Value")
                    
                    # Final values
                    final_values = list(final_data.values())
                    ax2.bar(range(len(layers)), final_values, color='green', alpha=0.7)
                    ax2.set_xticks(range(len(layers)))
                    ax2.set_xticklabels(layers, rotation=45, ha='right')
                    ax2.set_title(f"{metric_name} - After Pruning", fontsize=12)
                    ax2.set_ylabel("Value")
                    
                    fig.suptitle(f"{metric_name} Comparison", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    fig.savefig(
                        viz_dir / f"{metric_name}_comparison.png",
                        dpi=config.get('analysis_config', {}).get('plot_dpi', 300),
                        bbox_inches='tight'
                    )
                    plt.close(fig)
    
    # 2. Generate pruning visualizations
    if 'pruning_results' in results and config.get('apply_pruning', False):
        logger.info("Creating pruning visualizations...")
        
        # Create sparsity visualization
        if 'sparsity' in results['pruning_results']:
            sparsity_data = results['pruning_results']['sparsity']
            fig = pruning_viz.plot_layer_wise_pruning(
                sparsity_by_layer=sparsity_data,
                title="Layer-wise Sparsity Distribution"
            )
            if fig:
                fig.savefig(
                    viz_dir / "sparsity_distribution.png",
                    dpi=config.get('analysis_config', {}).get('plot_dpi', 300),
                    bbox_inches='tight'
                )
                plt.close(fig)
    
    # 3. Generate HTML report
    if config.get('analysis_config', {}).get('generate_html_report', True):
        logger.info("Generating HTML report...")
        
        reporter = HTMLReporter(f"{config['name']} - Comprehensive Analysis")
        
        # Add configuration section
        config_content = f"<pre>{yaml.dump(config, default_flow_style=False)}</pre>"
        reporter.add_section("Experiment Configuration", config_content)
        
        # Add results summary
        summary_content = generate_results_summary(results, config)
        reporter.add_section("Results Summary", summary_content)
        
        # Add visualizations
        for img_file in viz_dir.glob("*.png"):
            reporter.add_figure(
                str(img_file),
                caption=img_file.stem.replace('_', ' ').title()
            )
        
        # Generate report
        report_path = output_dir / "report.html"
        reporter.generate(str(report_path))
        logger.info(f"HTML report saved to {report_path}")
    
    # 4. Save raw results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Raw results saved to {results_path}")


def generate_results_summary(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate HTML summary of results."""
    summary = ["<div class='results-summary'>"]
    
    # Training results
    if 'performance_history' in results:
        perf = results['performance_history']
        summary.append("<h3>Training Performance</h3>")
        
        if 'val_acc' in perf and perf['val_acc']:
            final_acc = perf['val_acc'][-1]
            best_acc = max(perf['val_acc'])
            summary.append(f"<p>Final Accuracy: {final_acc:.2%}</p>")
            summary.append(f"<p>Best Accuracy: {best_acc:.2%}</p>")
        
        if 'train_loss' in perf and perf['train_loss']:
            final_loss = perf['train_loss'][-1]
            summary.append(f"<p>Final Training Loss: {final_loss:.4f}</p>")
    
    # Pruning results
    if 'pruning_results' in results and 'sparsity' in results['pruning_results']:
        summary.append("<h3>Pruning Results</h3>")
        sparsity = results['pruning_results']['sparsity']
        summary.append(f"<p>Overall Sparsity: {sparsity.get('overall', 0):.2%}</p>")
        
        if 'performance_retention' in results.get('analysis', {}):
            retention = results['analysis']['performance_retention']
            summary.append(f"<p>Performance Retention: {retention:.2%}</p>")
    
    # Metric changes
    if 'analysis' in results and 'metric_changes' in results['analysis']:
        summary.append("<h3>Metric Changes</h3>")
        for metric, changes in results['analysis']['metric_changes'].items():
            avg_change = sum(c.get('percent_change', 0) for c in changes.values()) / len(changes)
            summary.append(f"<p>{metric}: {avg_change:+.1f}% average change</p>")
    
    summary.append("</div>")
    return "\n".join(summary)


def main():
    """Main experiment execution."""
    # Parse arguments
    args = parse_args()
    
    # Load and merge configuration
    config = load_and_merge_config(args)
    
    # Setup logging
    logger, output_dir = setup_logging(config)
    
    # Log experiment info
    logger.info("=" * 80)
    logger.info("UNIFIED ALIGNMENT EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Experiment: {config['name']}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Device: {config.get('training_config', {}).get('device', 'cuda')}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # Set random seed
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.get('seed', 42))
        
        # Create experiment configuration
        experiment_config = create_experiment_config(config)
        
        # Create and run experiment
        logger.info("Creating experiment...")
        experiment = GeneralAlignmentExperiment(experiment_config)
        
        logger.info("Running experiment...")
        results = experiment.run()
        
        # Generate comprehensive report
        if config.get('analysis_config', {}).get('generate_plots', True):
            generate_comprehensive_report(results, config, output_dir)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Print key results
        if 'performance_history' in results:
            perf = results['performance_history']
            if 'val_acc' in perf and perf['val_acc']:
                logger.info(f"Final Validation Accuracy: {perf['val_acc'][-1]:.2%}")
        
        if 'pruning_results' in results and 'sparsity' in results['pruning_results']:
            sparsity = results['pruning_results']['sparsity'].get('overall', 0)
            logger.info(f"Achieved Sparsity: {sparsity:.1%}")
        
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 