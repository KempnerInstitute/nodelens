#!/usr/bin/env python3
"""Run a pruning experiment using the general alignment framework."""

import argparse
import yaml
import torch
from pathlib import Path
import json
from datetime import datetime

from src.alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig


def main():
    parser = argparse.ArgumentParser(description="Run pruning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Update device
    config_dict['device'] = args.device
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config_dict.get('experiment_name', 'pruning_experiment')
    output_name = f"{experiment_name}_{timestamp}"
    
    # Map config fields from YAML to GeneralAlignmentConfig
    # Remove fields that don't belong to the config class
    config_dict.pop('experiment_type', None)
    config_dict.pop('experiment_name', None)
    
    # Map the config fields
    mapped_config = {
        'name': experiment_name,  # Use experiment_name as the name field
        'seed': config_dict.get('seed', 42),
        'device': config_dict.get('device', 'cpu'),
        
        # Data configuration
        'dataset_name': config_dict.get('data', {}).get('dataset', 'mnist'),
        'batch_size': config_dict.get('data', {}).get('batch_size', 128),
        'num_workers': config_dict.get('data', {}).get('num_workers', 4),
        
        # Model configuration
        'model_name': config_dict.get('model', {}).get('architecture', 'mlp'),
        'model_config': {
            'hidden_dims': config_dict.get('model', {}).get('hidden_sizes', [512, 256]),  # MLP uses hidden_dims
            'activation_type': config_dict.get('model', {}).get('activation', 'relu'),  # MLP uses activation_type
            'dropout_rate': config_dict.get('model', {}).get('dropout_rate', 0.2),
            'input_dim': 784,  # MNIST flattened
            'output_dim': 10,  # MNIST classes
        },
        
        # Training configuration
        'do_train': config_dict.get('training', {}).get('do_train', True),
        'training_epochs': config_dict.get('training', {}).get('epochs', 10),
        'learning_rate': config_dict.get('training', {}).get('optimizer', {}).get('learning_rate', 0.001),
        'optimizer': config_dict.get('training', {}).get('optimizer', {}).get('type', 'adam'),
        
        # Pruning configuration
        'do_pruning_experiments': config_dict.get('pruning', {}).get('enabled', True),
        'pruning_strategies': ['magnitude', 'alignment'],  # Test both strategies
        'pruning_amounts': [0.3, 0.5, 0.7],  # Test fewer levels for faster execution
        'fine_tune_after_pruning': True,
        'fine_tune_epochs': config_dict.get('pruning', {}).get('fine_tune_epochs', 5),
        
        # Analysis configuration
        'measure_alignment_during_training': config_dict.get('analysis', {}).get('compute_metrics', True),
        'do_dropout_analysis': False,  # Disable dropout analysis for pruning experiments
        'do_eigenfeature_analysis': False,  # Disable eigenfeature analysis
        
        # Metrics - only use actual alignment metrics from the registry
        'metrics': ['rayleigh_quotient'],  # Only alignment metrics, not accuracy/loss
        'alignment_methods': ['rayleigh_quotient'],  # Default alignment method
        
        # Output configuration
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'log_dir': str(output_dir / 'logs'),
        'save_intermediate_results': config_dict.get('output', {}).get('save_results', True),
    }
    
    # Create config object
    print(f"Creating experiment config...")
    config = GeneralAlignmentConfig(**mapped_config)
    
    # Create and run experiment
    print(f"Running experiment on {args.device}...")
    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()
    
    # Save results
    results_path = output_dir / f"{output_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\n=== Experiment Summary ===")
    if 'test_results' in results:
        print(f"Final accuracy: {results['test_results']['final_accuracy']:.2f}%")
        print(f"Final loss: {results['test_results']['final_loss']:.4f}")
    
    if 'pruning_results' in results and 'strategies' in results['pruning_results']:
        print("\nPruning Results:")
        for strategy, strategy_results in results['pruning_results']['strategies'].items():
            print(f"\n{strategy}:")
            if 'pruning_amounts' in strategy_results:
                for i, amount in enumerate(strategy_results['pruning_amounts']):
                    sparsity = strategy_results['sparsities'][i] if 'sparsities' in strategy_results else 0
                    acc_before = strategy_results['accuracies_before_finetune'][i]
                    acc_after = strategy_results['accuracies_after_finetune'][i]
                    print(f"  {amount*100:.0f}% pruning: sparsity={sparsity:.2%}, "
                          f"acc_before={acc_before:.2f}%, acc_after={acc_after:.2f}%")


if __name__ == "__main__":
    main() 