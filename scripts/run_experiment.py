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
        'num_networks': unified_config.get('num_networks', 1),
        'aggregate_metrics': unified_config.get('aggregate_metrics', True),
        'save_individual_networks': unified_config.get('save_individual_networks', False),
        'save_checkpoints': unified_config.get('save_checkpoints', False),
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
        
        # CRITICAL FIX: Ensure plot generation is enabled
        # Check multiple possible locations for plot configuration
        visualization_config = unified_config.get('visualization', {})
        output_config = unified_config.get('output', {})
        analysis_config = unified_config.get('analysis', {})
        
        # Default to True if not explicitly set to False
        generate_plots = True
        if 'generate_plots' in visualization_config:
            generate_plots = visualization_config['generate_plots']
        elif 'generate_plots' in output_config:
            generate_plots = output_config['generate_plots']
        elif 'generate_plots' in analysis_config:
            generate_plots = analysis_config['generate_plots']
        
        config.generate_plots = generate_plots
        logger.info(f"Plot generation enabled: {config.generate_plots}")
        
        # Pruning specific - handle our new clean structure
        pruning_cfg = unified_config.get('pruning', {})
        
        # Check if pruning is enabled
        if pruning_cfg.get('enabled', False):
            config.do_pruning_experiments = True
            
            # Get algorithms
            algorithms = pruning_cfg.get('algorithms', ['magnitude'])
            config.pruning_strategies = algorithms if isinstance(algorithms, list) else [algorithms]
            
            # Get sparsity levels
            sparsity_levels = pruning_cfg.get('sparsity_levels', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
            config.pruning_amounts = sparsity_levels
            
            # Get selection modes (note the plural form)
            selection_modes = pruning_cfg.get('selection_modes', pruning_cfg.get('selection_mode', ['low']))
            config.pruning_selection_mode = selection_modes if isinstance(selection_modes, list) else [selection_modes]
            
            # Fine-tuning settings
            config.fine_tune_after_pruning = pruning_cfg.get('fine_tune_after_pruning', True)
            config.fine_tune_epochs = pruning_cfg.get('fine_tune_epochs', 5)
            config.fine_tune_learning_rate = pruning_cfg.get('fine_tune_learning_rate', 0.0001)
            
            # Scope settings
            scope = pruning_cfg.get('scope', 'layer')
            config.pruning_scope = scope
            
            # Alignment metric settings
            config.pruning_alignment_metric = pruning_cfg.get('alignment_metric', 'rayleigh_quotient')
            config.pruning_hybrid_alpha = pruning_cfg.get('hybrid_alpha', 0.5)
            
            logger.info(f"Pruning enabled: algorithms={config.pruning_strategies}, levels={config.pruning_amounts}, modes={config.pruning_selection_mode}")
        
        # Legacy support for pruning_analysis and network_compression blocks
        pruning_analysis = unified_config.get('pruning_analysis', {})
        network_compression = unified_config.get('network_compression', {})
    else:
        # Create base ExperimentConfig for other experiment types
        config = ExperimentConfig(**base_params)
        # Ensure plot generation is enabled for other experiment types too
        visualization_config = unified_config.get('visualization', {})
        output_config = unified_config.get('output', {})
        config.generate_plots = visualization_config.get('generate_plots', output_config.get('generate_plots', True))
    
    # Add additional attributes for compatibility with specialized experiments
    config.training_config = training_config
    config.train_model = config.training_config.get('epochs', 0) > 0
    config.alignment_metrics = config.metrics
    config.apply_pruning = True
    
    # Get pruning configuration with our new clean structure
    pruning_analysis = unified_config.get('pruning_analysis', {})
    network_compression = unified_config.get('network_compression', {})
    
    # Use the appropriate config based on what's enabled
    if pruning_analysis.get('enabled', False):
        active_pruning_config = pruning_analysis
        config.pruning_strategy = active_pruning_config.get('algorithms', ['magnitude'])[0]
    elif network_compression.get('enabled', False):
        active_pruning_config = network_compression
        config.pruning_strategy = active_pruning_config.get('algorithms', ['magnitude'])[0]
    else:
        # Fallback to empty config
        active_pruning_config = {}
        config.pruning_strategy = 'magnitude'
    
    config.pruning_config = active_pruning_config
    config.analysis_config = unified_config.get('visualization', unified_config.get('analysis', {}))
    config.eval_model = True
    config.cnn_mode = model_config.get('cnn_mode', 'unfold')
    # Set dropout rates based on our new structure
    if pruning_analysis.get('enabled', False):
        config.dropout_rates = pruning_analysis.get('dropout_rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    else:
        config.dropout_rates = unified_config.get('dropout', {}).get('rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    
    # Handle selection modes (which importance values to prune)
    # Support both single value and list from our new structure
    if pruning_analysis.get('enabled', False):
        selection_mode = pruning_analysis.get('selection_strategies', ['low'])
    elif network_compression.get('enabled', False):
        selection_mode = [network_compression.get('selection_strategy', 'low')]
    else:
        selection_mode = ['low']
    
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
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    
    args, unknown = parser.parse_known_args()
    
    # Parse additional overrides
    overrides = {}
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
    
    # CRITICAL FIX: Create plots directory explicitly
    plots_dir = Path(config.log_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created plots directory: {plots_dir}")
    
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
    # ------------------------------------------------------------------
    # Infer experiment_type automatically if the user did not specify it
    # Priority:
    #   1. cascading pruning  -> 'cascading_layer_pruning'
    #   2. standard pruning   -> 'standard_pruning'
    #   3. progressive dropout-> 'progressive_dropout'
    #   4. default            -> 'alignment_analysis'
    # ------------------------------------------------------------------
    if 'experiment_type' in unified_config:
        experiment_type = unified_config['experiment_type']
    else:
        pruning_cfg = unified_config.get('pruning', {})
        dropout_cfg = unified_config.get('dropout', {})

        pruning_enabled  = pruning_cfg.get('enabled', False)
        cascading_scope  = pruning_cfg.get('scope', 'layer') == 'cascading'
        dropout_enabled  = dropout_cfg.get('enabled', False)

        if cascading_scope and pruning_enabled:
            experiment_type = 'cascading_layer_pruning'
        elif pruning_enabled:
            experiment_type = 'standard_pruning'
        elif dropout_enabled:
            experiment_type = 'progressive_dropout'
        else:
            experiment_type = 'alignment_analysis'
# ------------------------------------------------------------------
    print(f"Plot generation: {getattr(config, 'generate_plots', True)}")
    print(f"Plots directory: {plots_dir}")
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
    
    # DEBUGGING: Check pruning configuration before creating experiment
    pruning_analysis = unified_config.get('pruning_analysis', {})
    network_compression = unified_config.get('network_compression', {})
    
    logger.info("=== PRUNING CONFIGURATION DEBUG ===")
    logger.info(f"pruning_analysis.enabled: {pruning_analysis.get('enabled', False)}")
    logger.info(f"network_compression.enabled: {network_compression.get('enabled', False)}")
    logger.info(f"config.do_pruning_experiments: {getattr(config, 'do_pruning_experiments', 'NOT SET')}")
    logger.info(f"config.generate_plots: {getattr(config, 'generate_plots', 'NOT SET')}")
    
    if hasattr(config, 'pruning_strategies'):
        logger.info(f"config.pruning_strategies: {config.pruning_strategies}")
    if hasattr(config, 'pruning_amounts'):
        logger.info(f"config.pruning_amounts: {config.pruning_amounts}")
    if hasattr(config, 'pruning_selection_mode'):
        logger.info(f"config.pruning_selection_mode: {config.pruning_selection_mode}")
    
    logger.info("=== END PRUNING DEBUG ===")
    
    if mapped_experiment_type in ['standard_pruning', 'progressive_dropout', 'alignment_analysis']:
        experiment = GeneralAlignmentExperiment(config)
    elif mapped_experiment_type == 'layer_isolated_pruning':
        experiment = LayerIsolatedPruningExperiment(config)
    elif mapped_experiment_type == 'cascading_layer_pruning':
        experiment = CascadingLayerPruningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {mapped_experiment_type}")
    
    # DEBUGGING: Log configuration state before running
    logger.info(f"Final config state:")
    logger.info(f"  - generate_plots: {getattr(config, 'generate_plots', 'NOT SET')}")
    logger.info(f"  - do_pruning_experiments: {getattr(config, 'do_pruning_experiments', 'NOT SET')}")
    logger.info(f"  - log_dir: {config.log_dir}")
    logger.info(f"  - plots_dir exists: {plots_dir.exists()}")
    
    # DEBUGGING: Check if the experiment object has the right configuration
    if hasattr(experiment, 'config'):
        exp_config = experiment.config
        logger.info(f"Experiment object config:")
        logger.info(f"  - generate_plots: {getattr(exp_config, 'generate_plots', 'NOT SET')}")
        logger.info(f"  - do_pruning_experiments: {getattr(exp_config, 'do_pruning_experiments', 'NOT SET')}")
        logger.info(f"  - log_dir: {getattr(exp_config, 'log_dir', 'NOT SET')}")
        
        # Check if pruning methods exist
        if hasattr(experiment, 'run_pruning_experiments'):
            logger.info("  - run_pruning_experiments method exists")
        else:
            logger.warning("  - run_pruning_experiments method MISSING")
            
        if hasattr(experiment, 'visualize_pruning_results'):
            logger.info("  - visualize_pruning_results method exists")
        else:
            logger.warning("  - visualize_pruning_results method MISSING")
    
    # Run experiment
    results = experiment.run()
    
    # DEBUGGING: Check if plots were actually created and log detailed info
    plots_created = list(plots_dir.glob('*.png')) + list(plots_dir.glob('*.pdf')) + list(plots_dir.glob('*.jpg'))
    logger.info(f"Plots created after experiment: {len(plots_created)} files")
    for plot_file in plots_created:
        logger.info(f"  - {plot_file.name} (size: {plot_file.stat().st_size} bytes)")
    
    # DEBUGGING: Check if pruning results exist in the results
    if 'pruning_results' in results:
        logger.info("Pruning results found in experiment results")
        pruning_results = results['pruning_results']
        if isinstance(pruning_results, dict):
            logger.info(f"Pruning results keys: {list(pruning_results.keys())}")
            if 'strategies' in pruning_results:
                strategies = pruning_results['strategies']
                logger.info(f"Pruning strategies in results: {list(strategies.keys()) if isinstance(strategies, dict) else strategies}")
        else:
            logger.info(f"Pruning results type: {type(pruning_results)}")
    else:
        logger.warning("NO pruning results found in experiment results")
        logger.info(f"Available result keys: {list(results.keys()) if isinstance(results, dict) else 'Results not a dict'}")
    
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
        f.write(f"Plot Generation: {getattr(config, 'generate_plots', True)}\n")
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
        
        # List plots created
        plots_created = list(plots_dir.glob('*'))
        if plots_created:
            f.write(f"\nPlots Generated ({len(plots_created)}):\n")
            for plot_file in sorted(plots_created):
                if plot_file.is_file():
                    f.write(f"  - {plot_file.name}\n")
        else:
            f.write("\nNo plots were generated.\n")
        
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
    
    # Check and report on plots
    plots_created = list(plots_dir.glob('*'))
    if plots_created:
        print(f"  - Plots ({len(plots_created)}): {plots_dir}/")
        for plot_file in sorted(plots_created):
            if plot_file.is_file():
                print(f"    * {plot_file.name}")
    else:
        print(f"  - No plots generated (check generate_plots setting and experiment configuration)")
        print(f"    * Pruning enabled: {getattr(config, 'do_pruning_experiments', False)}")
        print(f"    * Plot generation: {getattr(config, 'generate_plots', False)}")
        print(f"    * Pruning results in output: {'pruning_results' in results}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main() 