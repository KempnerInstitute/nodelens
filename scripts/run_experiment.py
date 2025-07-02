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
import logging
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Import from the alignment package
from alignment.experiments.general_alignment import GeneralAlignmentExperiment
from alignment.pruning.experiments.layer_wise import LayerIsolatedPruningExperiment
from alignment.pruning.experiments.cascading_layer import CascadingLayerPruningExperiment

logger = logging.getLogger(__name__)


def load_config(config_path, overrides=None):
    """Load and merge configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
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
    
    experiment_type = unified_config.get('experiment_type', 'alignment_analysis')
    
    # Map experiment types to what the script expects
    experiment_type_mapping = {
        'general_alignment': 'standard_pruning',  # Map general_alignment to standard_pruning
        'standard_pruning': 'standard_pruning',
        'progressive_dropout': 'progressive_dropout',
        'alignment_analysis': 'alignment_analysis',
        'layer_isolated_pruning': 'layer_isolated_pruning',
        'cascading_layer_pruning': 'cascading_layer_pruning'
    }
    
    mapped_experiment_type = experiment_type_mapping.get(experiment_type, experiment_type)
    
    # Extract model config and handle different naming conventions
    model_config = unified_config.get('model', {})
    model_name = model_config.get('name', model_config.get('architecture', 'mlp'))
    
    # Extract dataset config
    dataset_config = unified_config.get('dataset', {})
    if isinstance(dataset_config, str):
        # Handle case where dataset is just a string
        dataset_name = dataset_config
        dataset_config = {'name': dataset_name}
    else:
        dataset_name = dataset_config.get('name', dataset_config.get('dataset', 'mnist'))
    
    # Build base parameters for all experiment types
    base_params = {
        'name': unified_config.get('experiment_name', 'unified_experiment'),
        'seed': unified_config.get('seed', 42),
        'device': unified_config.get('device', 'cuda'),
        'model_name': model_name,
        'dataset_name': dataset_name,
        'batch_size': dataset_config.get('batch_size', unified_config.get('data', {}).get('batch_size', 128)),
        'num_workers': dataset_config.get('num_workers', unified_config.get('data', {}).get('num_workers', 4)),
        'metrics': unified_config.get('alignment', {}).get('metrics', unified_config.get('analysis', {}).get('metrics', ['rayleigh_quotient'])),
    }
    
    # Build model config with proper parameter names
    model_kwargs = {}
    if model_name == 'mlp':
        # Map hidden_sizes to hidden_dims
        model_kwargs['hidden_dims'] = model_config.get('hidden_dims', model_config.get('hidden_sizes', [512, 256]))
        model_kwargs['activation_type'] = model_config.get('activation_type', model_config.get('activation', 'relu'))
        model_kwargs['dropout_rate'] = model_config.get('dropout_rate', 0.0)
        # Determine input/output dims based on dataset
        if dataset_name == 'mnist':
            model_kwargs['input_dim'] = 784
            model_kwargs['output_dim'] = 10
        elif dataset_name == 'cifar10':
            model_kwargs['input_dim'] = 3072
            model_kwargs['output_dim'] = 10
        elif dataset_name == 'cifar100':
            model_kwargs['input_dim'] = 3072
            model_kwargs['output_dim'] = 100
        else:
            model_kwargs['input_dim'] = model_config.get('input_dim', 784)
            model_kwargs['output_dim'] = model_config.get('output_dim', 10)
    
    base_params['model_config'] = model_kwargs
    
    # Add training config if present
    training_config = unified_config.get('training', {})
    if training_config:
        base_params['training_epochs'] = training_config.get('epochs', 10)
        # Handle different optimizer config formats
        if isinstance(training_config.get('optimizer'), dict):
            base_params['learning_rate'] = training_config['optimizer'].get('learning_rate', 0.001)
            base_params['optimizer'] = training_config['optimizer'].get('type', 'adam')
        else:
            base_params['learning_rate'] = training_config.get('learning_rate', 0.001)
            base_params['optimizer'] = training_config.get('optimizer', 'adam')
    
    # Create the appropriate config type based on experiment
    if mapped_experiment_type in ['standard_pruning', 'progressive_dropout', 'alignment_analysis']:
        # Create GeneralAlignmentConfig for general experiments
        config = GeneralAlignmentConfig(**base_params)
        
        # Set GeneralAlignmentConfig specific fields
        config.do_train = training_config.get('epochs', training_config.get('do_train', 0)) > 0
        config.do_dropout_analysis = mapped_experiment_type == 'progressive_dropout'
        config.do_pruning_experiments = mapped_experiment_type == 'standard_pruning' or unified_config.get('pruning', {}).get('enabled', False)
        config.generate_plots = unified_config.get('visualization', {}).get('generate_plots', unified_config.get('output', {}).get('generate_plots', True))
        
        # Pruning specific
        if config.do_pruning_experiments:
            pruning = unified_config.get('pruning', {})
            # Handle both single values and lists for algorithms and sparsity levels
            algorithms = pruning.get('algorithms', pruning.get('strategy', 'magnitude'))  # Support old name for compatibility
            config.pruning_strategies = algorithms if isinstance(algorithms, list) else [algorithms]
            
            sparsity_levels = pruning.get('sparsity_levels', pruning.get('amount', pruning.get('pruning_amounts', 0.5)))  # Support old names
            config.pruning_amounts = sparsity_levels if isinstance(sparsity_levels, list) else [sparsity_levels]
            
            config.fine_tune_after_pruning = pruning.get('fine_tune_after_pruning', pruning.get('fine_tune', True))
            config.fine_tune_epochs = pruning.get('fine_tune_epochs', 5)
            
            # Selection mode can be single value or list
            selection_mode = pruning.get('selection_mode', 'low')
            config.pruning_selection_mode = selection_mode  # Keep as-is, the experiment will handle list vs single
            
            # Alignment-based pruning settings
            config.pruning_alignment_metric = pruning.get('alignment_metric', 'rayleigh_quotient')
            config.pruning_hybrid_alpha = pruning.get('hybrid_alpha', 0.5)
            
            # Pruning scope
            config.pruning_scope = pruning.get('scope', 'layer')
    else:
        # Create base ExperimentConfig for other experiment types
        config = ExperimentConfig(**base_params)
    
    # Add additional attributes for compatibility with specialized experiments
    config.training_config = training_config
    config.train_model = config.training_config.get('epochs', 0) > 0
    config.alignment_metrics = config.metrics
    config.apply_pruning = True
    
    # Get pruning configuration with backward compatibility
    pruning_config = unified_config.get('pruning', {})
    config.pruning_strategy = pruning_config.get('algorithms', pruning_config.get('strategy', 'magnitude'))
    if isinstance(config.pruning_strategy, list):
        config.pruning_strategy = config.pruning_strategy[0]  # Use first algorithm as default
    
    config.pruning_config = pruning_config
    config.analysis_config = unified_config.get('visualization', unified_config.get('analysis', {}))
    config.eval_model = True
    config.cnn_mode = model_config.get('cnn_mode', 'unfold')
    config.dropout_rates = unified_config.get('dropout', {}).get('rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Handle selection modes (which importance values to prune)
    # Support both single value and list
    selection_mode = pruning_config.get('selection_mode', 'low')
    config.pruning_modes = selection_mode if isinstance(selection_mode, list) else [selection_mode]
    
    config.cascade_direction = unified_config.get('experiment_specific', {}).get('cascade_direction', 'forward')
    config.recompute_scores = True
    
    # Note: Output directories are now set in main() after creating timestamped folders
    config.plot_dpi = unified_config.get('visualization', {}).get('plot_dpi', unified_config.get('output', {}).get('plot_dpi', 300))
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Alignment Experiment Runner')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--experiment_type', type=str, help='Override experiment type')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    
    args, unknown = parser.parse_known_args()
    
    # Parse additional overrides
    overrides = {}
    if args.experiment_type:
        overrides['experiment_type'] = args.experiment_type
    if args.device:
        overrides['device'] = args.device
    if args.seed:
        overrides['seed'] = args.seed
    
    # Load config
    unified_config = load_config(args.config, overrides)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = unified_config.get('experiment_name', 'experiment')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create a unique directory with experiment name and timestamp
        output_dir = Path(f"results/{experiment_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration used
    config_save_path = output_dir / 'experiment_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(unified_config, f, default_flow_style=False, sort_keys=False)
    
    # Create experiment config
    config = create_experiment_config(unified_config)
    
    # Update config with timestamped directories
    config.checkpoint_dir = str(output_dir / 'checkpoints')
    config.log_dir = str(output_dir / 'logs')
    config.experiment_dir = str(output_dir)  # Add experiment_dir for compatibility
    
    # Ensure directories exist
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging to both file and console
    log_file = output_dir / 'experiment.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Print experiment info
    print(f"\n{'='*60}")
    print(f"Running Alignment Experiment")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config.device}")
    print(f"Experiment type: {unified_config.get('experiment_type', 'alignment_analysis')}")
    print(f"{'='*60}\n")
    
    # Create experiment based on type
    experiment_type = unified_config.get('experiment_type', 'alignment_analysis')
    
    # Map experiment types
    experiment_type_mapping = {
        'general_alignment': 'standard_pruning',
        'standard_pruning': 'standard_pruning',
        'progressive_dropout': 'progressive_dropout',
        'alignment_analysis': 'alignment_analysis',
        'layer_isolated_pruning': 'layer_isolated_pruning',
        'cascading_layer_pruning': 'cascading_layer_pruning'
    }
    
    mapped_experiment_type = experiment_type_mapping.get(experiment_type, experiment_type)
    
    logger.info(f"Running {experiment_type} experiment (mapped to {mapped_experiment_type})")
    
    if mapped_experiment_type in ['standard_pruning', 'progressive_dropout', 'alignment_analysis']:
        experiment = GeneralAlignmentExperiment(config)
    elif mapped_experiment_type == 'layer_isolated_pruning':
        experiment = LayerIsolatedPruningExperiment(config)
    elif mapped_experiment_type == 'cascading_layer_pruning':
        experiment = CascadingLayerPruningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {mapped_experiment_type}")
    
    # Run experiment
    results = experiment.run()
    
    # Save results with timestamp
    results_file = output_dir / f'results_{timestamp}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create experiment summary
    summary_file = output_dir / 'experiment_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Experiment Type: {experiment_type}\n")
        f.write("=" * 50 + "\n\n")
        
        # Add results summary
        if 'test_results' in results:
            f.write("Final Model Performance:\n")
            f.write(f"  - Accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%\n")
            f.write(f"  - Loss: {results['test_results'].get('final_loss', 'N/A'):.4f}\n\n")
        
        if 'pruning_results' in results and results['pruning_results']:
            f.write("Pruning Experiments:\n")
            strategies = results['pruning_results'].get('strategies', {})
            f.write(f"  - Strategies tested: {list(strategies.keys())}\n")
            f.write(f"  - Plots saved in: {config.log_dir}/plots/\n")
        
        f.write("\nGenerated Files:\n")
        for file_path in sorted(output_dir.rglob('*')):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_dir)
                f.write(f"  - {relative_path}\n")
    
    # Print completion message
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    
    if 'test_results' in results:
        print(f"Final model accuracy: {results['test_results'].get('final_accuracy', 'N/A'):.2f}%")
        print(f"Final model loss: {results['test_results'].get('final_loss', 'N/A'):.4f}")
    
    print(f"\nAll results saved in: {output_dir}")
    print(f"  - Configuration: {config_save_path}")
    print(f"  - Results: {results_file}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Logs: {log_file}")
    
    if Path(config.log_dir, 'plots').exists():
        print(f"  - Plots: {config.log_dir}/plots/")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main() 