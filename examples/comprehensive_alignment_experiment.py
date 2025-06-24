#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Comprehensive Alignment Experiment Script

This script demonstrates ALL features of the alignment framework in a single,
configurable experiment. It supports:

1. Multiple model architectures (MLP, CNNs, Vision Transformers)
2. Various datasets (MNIST, CIFAR, ImageNet, etc.)
3. All available alignment metrics (36+ metrics)
4. Multiple pruning strategies
5. Advanced training options
6. Comprehensive analysis and visualization

Usage:
    python comprehensive_alignment_experiment.py --config configs/comprehensive_alignment_config.yaml
    
    # Override specific parameters:
    python comprehensive_alignment_experiment.py --config configs/comprehensive_alignment_config.yaml \
        --model_name resnet50 --dataset_name cifar100 --training_config.epochs 50
        
    # Run without training (just compute metrics):
    python comprehensive_alignment_experiment.py --config configs/comprehensive_alignment_config.yaml \
        --train_model false --compute_initial_metrics true

Configuration:
    This script requires a YAML configuration file. Two configs are provided:
    - configs/comprehensive_alignment_config.yaml: Full config with ALL options documented
    - configs/quick_test_config.yaml: Minimal config for quick testing
    
    Any parameter in the config can be overridden from the command line.

Requirements:
    - PyTorch with CUDA support (optional, will use CPU if not available)
    - torchvision (for models and datasets)
    - alignment package installed
    - matplotlib (for visualizations)
    - pyyaml (for config loading)

Output:
    Results are saved to: logs/<experiment_name>/
    ├── <experiment_name>_<timestamp>.log    # Detailed execution log
    ├── results.json                         # Complete experiment results
    ├── report.html                          # Comprehensive HTML report
    └── visualizations/                      # Generated plots
        ├── summary.png                      # Quick summary visualization
        ├── <metric>_comparison.png          # Per-metric comparisons
        ├── sparsity_by_layer.png           # Pruning visualization
        └── alignment_heatmap.png           # Comprehensive metric heatmap

Example Configurations:
    See configs/comprehensive_alignment_config.yaml for all available options.
    Each option is documented with its purpose and available values.
"""

import argparse
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to Python path if running from examples
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Import alignment framework components
from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.analysis import ResultAggregator, HTMLReporter
from alignment.analysis.visualization import (
    MetricVisualizer,
    PruningVisualizer,
    AlignmentVisualizer,
    plot_quick_summary
)


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_dir = Path(config.get("log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{config['name']}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if config.get("verbose", True) else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Alignment Experiment",
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
    parser.add_argument(
        "--name",
        type=str,
        help="Override experiment name"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Override model architecture"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Override dataset"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Override device (cuda/cpu)"
    )
    
    # Training overrides
    parser.add_argument(
        "--training_config.epochs",
        type=int,
        dest="training_epochs",
        help="Override number of training epochs"
    )
    
    parser.add_argument(
        "--training_config.batch_size",
        type=int,
        dest="batch_size",
        help="Override batch size"
    )
    
    parser.add_argument(
        "--training_config.learning_rate",
        type=float,
        dest="learning_rate",
        help="Override learning rate"
    )
    
    # Workflow overrides
    parser.add_argument(
        "--train_model",
        type=lambda x: x.lower() == 'true',
        help="Whether to train the model"
    )
    
    parser.add_argument(
        "--apply_pruning",
        type=lambda x: x.lower() == 'true',
        help="Whether to apply pruning"
    )
    
    parser.add_argument(
        "--pruning_config.amount",
        type=float,
        dest="pruning_amount",
        help="Override pruning amount"
    )
    
    return parser.parse_args()


def load_and_update_config(args) -> Dict[str, Any]:
    """Load config from YAML and apply command-line overrides."""
    # Load base config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
    if args.name:
        config['name'] = args.name
    if args.model_name:
        config['model_name'] = args.model_name
    if args.dataset_name:
        config['dataset_name'] = args.dataset_name
    if args.device:
        config['device'] = args.device
    
    # Training overrides
    if args.training_epochs is not None:
        config['training_config']['epochs'] = args.training_epochs
    if args.batch_size is not None:
        config['training_config']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training_config']['learning_rate'] = args.learning_rate
    
    # Workflow overrides
    if args.train_model is not None:
        config['train_model'] = args.train_model
    if args.apply_pruning is not None:
        config['apply_pruning'] = args.apply_pruning
    if args.pruning_amount is not None:
        config['pruning_config']['amount'] = args.pruning_amount
    
    return config


def create_experiment_config(config_dict: Dict[str, Any]) -> GeneralAlignmentConfig:
    """Create experiment configuration from dictionary."""
    # Extract relevant sections
    config = GeneralAlignmentConfig(
        # Basic info
        name=config_dict['name'],
        description=config_dict.get('description', ''),
        tags=config_dict.get('tags', []),
        
        # Model
        model_name=config_dict['model_name'],
        model_config=config_dict.get('model_config', {}),
        pretrained=config_dict.get('pretrained', False),  # Only use top-level pretrained if exists
        
        # Dataset
        dataset_name=config_dict['dataset_name'],
        dataset_config=config_dict.get('dataset_config', {}),
        data_path=config_dict.get('dataset_config', {}).get('data_path'),
        
        # Training
        training_config=config_dict.get('training_config', {}),
        
        # Metrics
        alignment_metrics=config_dict.get('alignment_metrics', ['rayleigh_quotient']),
        metric_configs=config_dict.get('metric_configs', {}),
        compute_metrics_on=config_dict.get('compute_metrics_on'),
        
        # Pruning
        pruning_strategy=config_dict.get('pruning_strategy', 'magnitude'),
        pruning_config=config_dict.get('pruning_config', {}),
        pruning_based_on_metric=config_dict.get('pruning_based_on_metric'),
        
        # Workflow
        train_model=config_dict.get('train_model', True),
        compute_initial_metrics=config_dict.get('compute_initial_metrics', True),
        apply_pruning=config_dict.get('apply_pruning', True),
        fine_tune_after_pruning=config_dict.get('fine_tune_after_pruning', True),
        fine_tune_epochs=config_dict.get('fine_tune_epochs', 10),
        
        # Analysis
        track_performance=config_dict.get('track_performance', True),
        save_checkpoints=config_dict.get('save_checkpoints', True),
        save_metrics_history=config_dict.get('save_metrics_history', True),
        
        # Resources
        device=config_dict.get('device', 'cuda'),
        batch_size=config_dict.get('training_config', {}).get('batch_size', 128),
        num_workers=config_dict.get('num_workers', 4),
        seed=config_dict.get('seed', 42),
        
        # Paths
        checkpoint_dir=config_dict.get('checkpoint_dir', './checkpoints'),
        log_dir=config_dict.get('log_dir', './logs'),
        
        # Additional metric options
        scale_by_norm=config_dict.get('metric_configs', {}).get('rayleigh_quotient', {}).get('scale_by_norm', False),
        force_cpu_for_large_metric_ops=config_dict.get('force_cpu_for_large_metric_ops', False),
        cnn_rq_aggregation_op=config_dict.get('metric_configs', {}).get('rayleigh_quotient', {}).get('aggregation_op', 'mean'),
        exclude_classification_layer=config_dict.get('exclude_classification_layer', True)
    )
    
    return config


def generate_visualizations(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Generate comprehensive visualizations from results."""
    logger = logging.getLogger(__name__)
    logger.info("Generating visualizations...")
    
    # Create output directory
    output_dir = Path(config.get('log_dir', './logs')) / config['name'] / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Quick summary plot
        if 'initial_metrics' in results:
            summary_path = output_dir / 'summary.png'
            plot_quick_summary(
                results,
                save_path=str(summary_path),
                title=f"{config['name']} - Summary"
            )
            logger.info(f"Summary plot saved to {summary_path}")
        
        # 2. Metric visualizations
        if 'initial_metrics' in results or 'final_metrics' in results:
            metric_viz = MetricVisualizer()
            
            # Plot each metric
            for metric_name in config.get('alignment_metrics', []):
                if 'initial_metrics' in results and metric_name in results['initial_metrics']:
                    fig = metric_viz.plot_layer_comparison(
                        results['initial_metrics'][metric_name],
                        results.get('final_metrics', {}).get(metric_name, {}),
                        metric_name=metric_name,
                        title=f"{metric_name} - Initial vs Final"
                    )
                    if fig:
                        fig.savefig(
                            output_dir / f"{metric_name}_comparison.png",
                            dpi=config.get('plot_dpi', 300),
                            bbox_inches='tight'
                        )
                        plt.close(fig)
        
        # 3. Pruning visualization
        if 'pruning_results' in results and 'sparsity' in results['pruning_results']:
            pruning_viz = PruningVisualizer()
            
            # Create sparsity visualization
            sparsity_data = results['pruning_results']['sparsity']
            fig = pruning_viz.plot_sparsity_by_layer(
                sparsity_data,
                title="Sparsity by Layer"
            )
            if fig:
                fig.savefig(
                    output_dir / "sparsity_by_layer.png",
                    dpi=config.get('plot_dpi', 300),
                    bbox_inches='tight'
                )
                plt.close(fig)
        
        # 4. Alignment visualization
        if 'initial_metrics' in results:
            align_viz = AlignmentVisualizer()
            
            # Create comprehensive alignment plot
            fig = align_viz.plot_metric_heatmap(
                results['initial_metrics'],
                title="Alignment Metrics Heatmap"
            )
            if fig:
                fig.savefig(
                    output_dir / "alignment_heatmap.png",
                    dpi=config.get('plot_dpi', 300),
                    bbox_inches='tight'
                )
                plt.close(fig)
            
        logger.info(f"Visualizations saved to {output_dir}")
            
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def generate_report(results: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Generate comprehensive HTML report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating comprehensive report...")
    
    # Create output directory
    output_dir = Path(config.get('log_dir', './logs')) / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create report
    reporter = HTMLReporter(f"{config['name']} - Alignment Analysis Report")
    
    # Add configuration section
    reporter.add_section("Experiment Configuration")
    reporter.add_code_block(yaml.dump(config, default_flow_style=False), language='yaml')
    
    # Add results summary
    reporter.add_section("Results Summary")
    
    # Performance metrics
    if 'performance_history' in results:
        perf = results['performance_history']
        if perf.get('val_acc'):
            initial_acc = max(perf['val_acc'][:config['training_config']['epochs']]) if perf['val_acc'] else 0
            final_acc = max(perf['val_acc'][-config.get('fine_tune_epochs', 10):]) if len(perf['val_acc']) > config['training_config']['epochs'] else initial_acc
            
            reporter.add_text(f"Initial Accuracy: {initial_acc:.2%}")
            reporter.add_text(f"Final Accuracy: {final_acc:.2%}")
            reporter.add_text(f"Accuracy Drop: {(initial_acc - final_acc):.2%}")
    
    # Sparsity achieved
    if 'pruning_results' in results and 'sparsity' in results['pruning_results']:
        sparsity = results['pruning_results']['sparsity']
        reporter.add_text(f"Overall Sparsity: {sparsity.get('overall', 0):.2%}")
    
    # Add detailed metrics
    reporter.add_section("Alignment Metrics Analysis")
    
    if 'analysis' in results and 'metric_changes' in results['analysis']:
        for metric_name, changes in results['analysis']['metric_changes'].items():
            reporter.add_subsection(metric_name)
            
            # Create summary table
            table_data = []
            for layer, change_data in changes.items():
                table_data.append({
                    'Layer': layer,
                    'Absolute Change': f"{change_data['absolute_change']:.4f}",
                    'Percent Change': f"{change_data['percent_change']:.2f}%"
                })
            
            if table_data:
                reporter.add_dataframe(table_data)
    
    # Add raw results
    reporter.add_section("Raw Results")
    reporter.add_code_block(json.dumps(results, indent=2, default=str), language='json')
    
    # Generate report
    report_path = output_dir / 'report.html'
    reporter.generate(str(report_path))
    logger.info(f"Report saved to {report_path}")


def main():
    """Main experiment execution."""
    # Parse arguments
    args = parse_args()
    
    # Load and update configuration
    config = load_and_update_config(args)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ALIGNMENT EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Experiment: {config['name']}")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Device: {config['device']}")
    logger.info("=" * 80)
    
    try:
        # Create experiment configuration
        experiment_config = create_experiment_config(config)
        
        # Create and run experiment
        logger.info("Creating experiment...")
        experiment = GeneralAlignmentExperiment(experiment_config)
        
        logger.info("Running experiment...")
        results = experiment.run()
        
        # Save results
        results_dir = Path(config.get('log_dir', './logs')) / config['name']
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_path = results_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_path}")
        
        # Generate visualizations
        if config.get('generate_plots', True):
            generate_visualizations(results, config)
        
        # Generate report
        generate_report(results, config)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Print key results
        if 'analysis' in results:
            analysis = results['analysis']
            
            # Sparsity achieved
            if 'sparsity_impact' in analysis:
                sparsity = analysis['sparsity_impact']['achieved_sparsity'].get('overall', 0)
                retention = analysis['sparsity_impact'].get('performance_retention', 0)
                logger.info(f"Sparsity Achieved: {sparsity:.1%}")
                logger.info(f"Performance Retention: {retention:.1%}")
            
            # Metric changes
            if 'metric_changes' in analysis:
                logger.info("\nKey Metric Changes:")
                for metric, changes in analysis['metric_changes'].items():
                    avg_change = sum(c['percent_change'] for c in changes.values()) / len(changes)
                    logger.info(f"  {metric}: {avg_change:+.1f}% average change")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 