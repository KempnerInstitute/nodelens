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
    python run_unified_experiment.py --config configs/unified_config.yaml
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
    
    # Build base parameters for all experiment types
    base_params = {
        'name': unified_config.get('experiment_name', 'unified_experiment'),
        'seed': unified_config.get('seed', 42),
        'device': unified_config.get('device', 'cuda'),
        'model_name': unified_config['model']['name'],
        'model_config': unified_config['model'],
        'dataset_name': unified_config['dataset']['name'],
        'dataset_config': unified_config['dataset'],
        'batch_size': unified_config['dataset'].get('batch_size', 128),
        'num_workers': unified_config['dataset'].get('num_workers', 4),
        'metrics': unified_config['alignment'].get('metrics', ['rayleigh_quotient']),
    }
    
    # Add training config if present
    if 'training' in unified_config:
        training = unified_config['training']
        base_params['training_epochs'] = training.get('epochs', 10)
        base_params['learning_rate'] = training.get('learning_rate', 0.001)
        base_params['optimizer'] = training.get('optimizer', 'adam')
    
    # Create the appropriate config type based on experiment
    if experiment_type in ['standard_pruning', 'progressive_dropout', 'alignment_analysis']:
        # Create GeneralAlignmentConfig for general experiments
        config = GeneralAlignmentConfig(**base_params)
        
        # Set GeneralAlignmentConfig specific fields
        config.do_train = unified_config.get('training', {}).get('epochs', 0) > 0
        config.do_dropout_analysis = experiment_type == 'progressive_dropout'
        config.do_pruning_experiments = experiment_type == 'standard_pruning'
        config.generate_plots = unified_config.get('visualization', {}).get('generate_plots', True)
        
        # Pruning specific
        if experiment_type == 'standard_pruning':
            pruning = unified_config.get('pruning', {})
            # Handle both single values and lists for strategies and amounts
            strategies = pruning.get('strategy', 'magnitude')
            config.pruning_strategies = strategies if isinstance(strategies, list) else [strategies]
            amounts = pruning.get('amount', 0.5)
            config.pruning_amounts = amounts if isinstance(amounts, list) else [amounts]
            config.fine_tune_after_pruning = pruning.get('fine_tune', True)
            config.fine_tune_epochs = pruning.get('fine_tune_epochs', 5)
    else:
        # Create base ExperimentConfig for other experiment types
        config = ExperimentConfig(**base_params)
    
    # Add additional attributes for compatibility with specialized experiments
    config.training_config = unified_config.get('training', {})
    config.train_model = config.training_config.get('epochs', 0) > 0
    config.alignment_metrics = config.metrics
    config.apply_pruning = True
    config.pruning_strategy = unified_config.get('pruning', {}).get('strategy', 'magnitude')
    config.pruning_config = unified_config.get('pruning', {})
    config.analysis_config = unified_config.get('visualization', {})
    config.eval_model = True
    config.cnn_mode = unified_config['model'].get('cnn_mode', 'unfold')
    config.dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    config.pruning_modes = config.pruning_config.get('strategies', ['high', 'low', 'random'])
    config.cascade_direction = unified_config.get('experiment_specific', {}).get('cascade_direction', 'forward')
    config.recompute_scores = True
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Alignment Experiment Runner')
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument('--experiment_type', type=str, help='Override experiment type')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--seed', type=int, help='Override seed')
    
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
    config = create_experiment_config(unified_config)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create experiment based on type
    experiment_type = unified_config.get('experiment_type', 'alignment_analysis')
    
    logger.info(f"Running {experiment_type} experiment")
    
    if experiment_type in ['standard_pruning', 'progressive_dropout', 'alignment_analysis']:
        experiment = GeneralAlignmentExperiment(config)
    elif experiment_type == 'layer_isolated_pruning':
        experiment = LayerIsolatedPruningExperiment(config)
    elif experiment_type == 'cascading_layer_pruning':
        experiment = CascadingLayerPruningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Run experiment
    results = experiment.run()
    
    # Get the appropriate directory based on experiment type
    if hasattr(experiment, 'experiment_dir'):
        save_dir = experiment.experiment_dir
    elif hasattr(experiment, 'results_dir'):
        save_dir = experiment.results_dir
    else:
        save_dir = "logs"
    
    logger.info(f"Experiment completed. Results saved to: {save_dir}")


if __name__ == '__main__':
    main() 