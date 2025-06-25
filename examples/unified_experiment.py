#!/usr/bin/env python3
"""
Unified Alignment Experiment

This is the main experiment script that can run any configuration specified
in the master config file. It supports all features of the alignment framework:
- Multiple model architectures
- Various datasets
- All alignment metrics
- All pruning strategies
- Specialized pruning experiments (cascading, layer-isolated)
- Comprehensive analysis and visualization

Usage:
    python unified_experiment.py --config configs/master_config.yaml
    
    # Use specialized pruning experiments:
    python unified_experiment.py --config configs/master_config.yaml \
        --pruning_experiment cascading_layer \
        --dropout_rates 0.1 0.3 0.5 0.7 0.9
    
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
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.experiments.runner import ExperimentRunner
from alignment.pruning.experiments import CascadingLayerPruningExperiment, LayerIsolatedPruningExperiment
from alignment.pruning.experiments.cascading_layer import CascadingConfig
from alignment.pruning.experiments.layer_wise import LayerIsolatedConfig
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
    
    # Pruning experiment type
    parser.add_argument(
        "--pruning_experiment",
        type=str,
        choices=['standard', 'cascading_layer', 'layer_isolated'],
        default='standard',
        help="Type of pruning experiment to run"
    )
    
    # Specialized pruning experiment parameters
    parser.add_argument(
        "--dropout_rates",
        type=float,
        nargs='+',
        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        help="Dropout rates for specialized pruning experiments"
    )
    
    parser.add_argument(
        "--cascade_direction",
        type=str,
        choices=['forward', 'backward'],
        default='forward',
        help="Direction for cascading layer pruning"
    )
    
    parser.add_argument(
        "--recompute_scores",
        type=lambda x: x.lower() == 'true',
        default=True,
        help="Whether to recompute scores after each layer in cascading"
    )
    
    parser.add_argument(
        "--pruning_modes",
        type=str,
        nargs='+',
        default=['low', 'high', 'random'],
        help="Pruning modes to evaluate"
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
    
    # Get pruning experiment parameters from config file first
    # These can be overridden by command line arguments
    if 'pruning_experiment' not in config:
        config['pruning_experiment'] = 'standard'
    if 'dropout_rates' not in config:
        config['dropout_rates'] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    if 'cascade_direction' not in config:
        config['cascade_direction'] = 'forward'
    if 'recompute_scores' not in config:
        config['recompute_scores'] = True
    if 'pruning_modes' not in config:
        config['pruning_modes'] = ['low', 'high', 'random']
    
    # Override with command line arguments if provided
    if args.pruning_experiment != 'standard':  # Only override if not default
        config['pruning_experiment'] = args.pruning_experiment
    if args.dropout_rates != [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:  # Only override if not default
        config['dropout_rates'] = args.dropout_rates
    if args.cascade_direction != 'forward':  # Only override if not default
        config['cascade_direction'] = args.cascade_direction
    if args.recompute_scores is not True:  # Only override if not default
        config['recompute_scores'] = args.recompute_scores
    if args.pruning_modes != ['low', 'high', 'random']:  # Only override if not default
        config['pruning_modes'] = args.pruning_modes
    
    # Apply other command line overrides
    overrides = vars(args)
    for key, value in overrides.items():
        if value is not None and key not in ['config', 'pruning_experiment', 'dropout_rates', 
                                              'cascade_direction', 'recompute_scores', 'pruning_modes']:
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


def create_cascading_config(config: Dict[str, Any]) -> CascadingConfig:
    """Create CascadingConfig from master config dictionary."""
    return CascadingConfig(
        # Basic info
        name=config.get('name', 'cascading_experiment'),
        model_name=config.get('model_name', 'resnet18'),
        model_config=config.get('model_config', {}),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_dir=config.get('log_dir', './logs'),
        device=config.get('training_config', {}).get('device', 'cuda'),
        seed=config.get('seed', 42),
        
        # Dataset
        dataset_name=config.get('dataset_name', 'cifar10'),
        dataset_config=config.get('dataset_config', {}),
        
        # Dropout configuration
        dropout_rates=config.get('dropout_rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]),
        dropout_mode=config.get('dropout_mode', 'scaled'),
        cascade_direction=config.get('cascade_direction', 'forward'),
        
        # Pruning configuration
        pruning_metric=config.get('alignment_metrics', ['rayleigh_quotient'])[0],
        pruning_strategy='low',  # Will be varied in the experiment
        exclude_classification_layer=config.get('exclude_classification_layer', True),
        recompute_scores=config.get('recompute_scores', True),
        
        # Training configuration
        train_before_dropout=config.get('train_model', True),
        training_epochs=config.get('training_config', {}).get('epochs', 10),
        learning_rate=config.get('training_config', {}).get('learning_rate', 0.001),
        optimizer=config.get('training_config', {}).get('optimizer', 'adam'),
        
        # Evaluation
        eval_batches=config.get('eval_batches'),
        num_random_trials=config.get('num_random_trials', 3),
        
        # Metrics
        metrics=config.get('alignment_metrics', ['rayleigh_quotient']),
    )


def create_layer_isolated_config(config: Dict[str, Any]) -> LayerIsolatedConfig:
    """Create LayerIsolatedConfig from master config dictionary."""
    return LayerIsolatedConfig(
        # Basic info
        name=config.get('name', 'layer_isolated_experiment'),
        model_name=config.get('model_name', 'resnet18'),
        model_config=config.get('model_config', {}),
        checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
        log_dir=config.get('log_dir', './logs'),
        device=config.get('training_config', {}).get('device', 'cuda'),
        seed=config.get('seed', 42),
        
        # Dataset
        dataset_name=config.get('dataset_name', 'cifar10'),
        dataset_config=config.get('dataset_config', {}),
        
        # Dropout configuration
        dropout_rates=config.get('dropout_rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]),
        dropout_mode=config.get('dropout_mode', 'scaled'),
        
        # Pruning configuration
        pruning_metric=config.get('alignment_metrics', ['rayleigh_quotient'])[0],
        pruning_strategy='low',  # Will be varied in the experiment
        exclude_classification_layer=config.get('exclude_classification_layer', True),
        
        # Training configuration
        train_before_dropout=config.get('train_model', True),
        training_epochs=config.get('training_config', {}).get('epochs', 10),
        learning_rate=config.get('training_config', {}).get('learning_rate', 0.001),
        optimizer=config.get('training_config', {}).get('optimizer', 'adam'),
        
        # Evaluation
        eval_batches=config.get('eval_batches'),
        num_random_trials=config.get('num_random_trials', 3),
        
        # Metrics
        metrics=config.get('alignment_metrics', ['rayleigh_quotient']),
    )


def generate_specialized_pruning_report(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    experiment_type: str
) -> None:
    """Generate visualizations for specialized pruning experiments."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating visualizations for {experiment_type} experiment...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizers
    pruning_viz = PruningVisualizer()
    alignment_viz = AlignmentVisualizer()
    
    # 1. Performance comparison across dropout rates using standard visualization
    if 'accuracies' in results and 'losses' in results:
        # Convert results to format expected by PruningVisualizer
        # Format: strategy -> sparsity -> metric -> value
        formatted_results = {}
        
        dropout_rates = results.get('dropout_rates', [])
        
        for strategy in ['low', 'high', 'random']:
            if strategy in results['accuracies']:
                formatted_results[strategy] = {}
                for i, rate in enumerate(dropout_rates):
                    formatted_results[strategy][rate] = {
                        'accuracy': results['accuracies'][strategy][i] if i < len(results['accuracies'][strategy]) else 0,
                        'loss': results['losses'][strategy][i] if strategy in results['losses'] and i < len(results['losses'][strategy]) else 0
                    }
        
        # Use standard pruning performance plot
        if formatted_results:
            fig = pruning_viz.plot_pruning_performance(
                formatted_results,
                metrics=['accuracy', 'loss'],
                title=f'{experiment_type.replace("_", " ").title()} - Performance vs Dropout Rate',
                save_path=viz_dir / "performance_comparison.png"
            )
            plt.close(fig)
    
    # 2. Layer scores visualization using AlignmentVisualizer
    if 'layer_scores' in results and results['layer_scores']:
        # Convert layer scores to expected format
        layer_scores_tensors = {}
        for layer_name, scores in results['layer_scores'].items():
            if isinstance(scores, list) and scores:
                # Convert list to tensor
                layer_scores_tensors[layer_name] = torch.tensor(scores)
            elif isinstance(scores, torch.Tensor):
                layer_scores_tensors[layer_name] = scores
        
        if layer_scores_tensors:
            fig = alignment_viz.plot_layer_scores(
                layer_scores_tensors,
                title=f'{experiment_type.replace("_", " ").title()} - Layer Alignment Scores',
                save_path=viz_dir / "layer_scores_distribution.png"
            )
            plt.close(fig)
    
    # 3. Generate comparison grid if we have enough data
    if 'accuracies' in results and len(results.get('dropout_rates', [])) > 2:
        # Use the formatted results from above
        if formatted_results:
            fig = pruning_viz.plot_pruning_comparison_grid(
                formatted_results,
                save_path=viz_dir / "comprehensive_comparison.png"
            )
            plt.close(fig)
    
    # 4. Specialized visualizations based on experiment type
    if experiment_type == 'cascading_layer' and 'cascade_masks' in results:
        # For cascading experiments, create a custom visualization showing the cascade effect
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dropout_rates_with_masks = []
        layer_active_neurons = {}
        
        # Extract data from cascade masks
        for dropout_key, cascade_info in results['cascade_masks'].items():
            if 'active_neurons' in cascade_info:
                try:
                    dropout_rate = float(dropout_key.split('_')[1])
                    dropout_rates_with_masks.append(dropout_rate)
                    
                    for layer, active_count in cascade_info['active_neurons'].items():
                        if layer not in layer_active_neurons:
                            layer_active_neurons[layer] = []
                        layer_active_neurons[layer].append((dropout_rate, active_count))
                except (ValueError, IndexError):
                    continue
        
        # Sort and plot
        for layer, data_points in layer_active_neurons.items():
            if data_points:
                data_points.sort(key=lambda x: x[0])
                rates, counts = zip(*data_points)
                ax.plot(rates, counts, marker='o', label=layer, linewidth=2)
        
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Active Neurons')
        ax.set_title('Cascading Effect: Active Neurons per Layer')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(viz_dir / "cascading_effect.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    elif experiment_type == 'layer_isolated' and 'layer_sparsities' in results:
        # For layer-isolated experiments, use the layer-wise pruning visualization
        # Format: strategy -> layer -> sparsity
        layer_sparsity_data = {}
        model_accuracy = {}
        
        for strategy in ['low', 'high', 'random']:
            if strategy in results.get('layer_sparsities', {}):
                layer_sparsity_data[strategy] = results['layer_sparsities'][strategy]
                # Get final accuracy for this strategy
                if 'accuracies' in results and strategy in results['accuracies']:
                    accuracies = results['accuracies'][strategy]
                    model_accuracy[strategy] = accuracies[-1] if accuracies else 0
        
        if layer_sparsity_data:
            fig = pruning_viz.plot_layer_wise_pruning(
                layer_sparsity_data,
                model_accuracy,
                save_path=viz_dir / "layer_wise_patterns.png"
            )
            plt.close(fig)
    
    # 5. Summary statistics
    summary_stats = {
        'experiment_type': experiment_type,
        'dropout_rates': results.get('dropout_rates', []),
        'performance_summary': {}
    }
    
    for strategy in ['low', 'high', 'random']:
        if strategy in results.get('accuracies', {}):
            accs = results['accuracies'][strategy]
            if accs:
                summary_stats['performance_summary'][strategy] = {
                    'best_accuracy': max(accs),
                    'worst_accuracy': min(accs),
                    'accuracy_drop': max(accs) - min(accs),
                    'final_accuracy': accs[-1]
                }
    
    # Save summary statistics
    stats_path = output_dir / "summary_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    logger.info(f"Visualizations saved to {viz_dir}")
    logger.info(f"Summary statistics saved to {stats_path}")


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
            
            # Use PruningVisualizer for layer-wise sparsity visualization
            # Convert to expected format: strategy -> layer -> sparsity
            layer_sparsity_by_strategy = {
                config.get('pruning_strategy', 'magnitude_based'): {
                    k: v for k, v in sparsity_data.items() if k != 'overall'
                }
            }
            
            # Get model accuracy if available
            model_accuracy = {}
            if 'performance_after' in results['pruning_results']:
                strategy = config.get('pruning_strategy', 'magnitude_based')
                model_accuracy[strategy] = results['pruning_results']['performance_after'].get('val_acc', 0) * 100
            
            # Use the standard layer-wise pruning visualization
            fig = pruning_viz.plot_layer_wise_pruning(
                layer_sparsity_by_strategy,
                model_accuracy,
                save_path=viz_dir / "layer_wise_sparsity.png"
            )
            plt.close(fig)
        
        # If we have performance comparison data, create pruning performance plot
        if 'performance_history' in results and 'val_acc' in results['performance_history']:
            # Create a simple before/after comparison
            performance_data = {
                'before_pruning': {
                    0.0: {
                        'accuracy': results['performance_history']['val_acc'][-1] * 100,
                        'loss': results['performance_history']['val_loss'][-1] if 'val_loss' in results['performance_history'] else 0
                    }
                }
            }
            
            if 'performance_after' in results['pruning_results']:
                sparsity = results['pruning_results']['sparsity'].get('overall', 0)
                performance_data['after_pruning'] = {
                    sparsity: {
                        'accuracy': results['pruning_results']['performance_after'].get('val_acc', 0) * 100,
                        'loss': results['pruning_results']['performance_after'].get('val_loss', 0)
                    }
                }
            
            fig = pruning_viz.plot_pruning_performance(
                performance_data,
                metrics=['accuracy', 'loss'],
                title='Pruning Impact on Performance',
                save_path=viz_dir / "pruning_performance.png"
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
    
    # Check if this is a specialized pruning experiment
    if 'dropout_rates' in results:
        # Specialized pruning experiment summary
        summary.append("<h3>Pruning Experiment Results</h3>")
        
        if 'accuracies' in results:
            summary.append("<table border='1' style='border-collapse: collapse;'>")
            summary.append("<tr><th>Dropout Rate</th><th>Low Mode</th><th>High Mode</th><th>Random Mode</th></tr>")
            
            dropout_rates = results.get('dropout_rates', [])
            for i, rate in enumerate(dropout_rates):
                summary.append(f"<tr><td>{rate:.1%}</td>")
                for mode in ['low', 'high', 'random']:
                    if mode in results['accuracies'] and i < len(results['accuracies'][mode]):
                        acc = results['accuracies'][mode][i]
                        summary.append(f"<td>{acc:.2f}%</td>")
                    else:
                        summary.append("<td>-</td>")
                summary.append("</tr>")
            
            summary.append("</table>")
    else:
        # Standard experiment summary
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
    logger.info(f"Pruning Experiment: {config['pruning_experiment']}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("=" * 80)
    
    try:
        # Set random seed
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.get('seed', 42))
        
        # Choose experiment type based on pruning_experiment parameter
        if config['pruning_experiment'] == 'cascading_layer':
            logger.info("Running Cascading Layer Pruning Experiment")
            experiment_config = create_cascading_config(config)
            experiment = CascadingLayerPruningExperiment(experiment_config)
            
        elif config['pruning_experiment'] == 'layer_isolated':
            logger.info("Running Layer-Isolated Pruning Experiment")
            experiment_config = create_layer_isolated_config(config)
            experiment = LayerIsolatedPruningExperiment(experiment_config)
            
        else:  # standard
            logger.info("Running Standard General Alignment Experiment")
            experiment_config = create_experiment_config(config)
            experiment = GeneralAlignmentExperiment(experiment_config)
        
        logger.info("Running experiment...")
        results = experiment.run()
        
        # Generate appropriate report based on experiment type
        if config['pruning_experiment'] in ['cascading_layer', 'layer_isolated']:
            generate_specialized_pruning_report(
                results, config, output_dir, config['pruning_experiment']
            )
        else:
            if config.get('analysis_config', {}).get('generate_plots', True):
                generate_comprehensive_report(results, config, output_dir)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Print key results based on experiment type
        if config['pruning_experiment'] in ['cascading_layer', 'layer_isolated']:
            # Print specialized experiment results
            if 'accuracies' in results:
                logger.info("Performance Summary:")
                for mode in ['low', 'high', 'random']:
                    if mode in results['accuracies'] and results['accuracies'][mode]:
                        best_acc = max(results['accuracies'][mode])
                        worst_acc = min(results['accuracies'][mode])
                        logger.info(f"  {mode} mode: Best={best_acc:.2f}%, Worst={worst_acc:.2f}%")
        else:
            # Print standard experiment results
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