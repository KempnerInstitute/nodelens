#!/usr/bin/env python3
"""
Run alignment experiments using unified_config.yaml with timestamped output folders.
"""

import argparse
import logging
import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch

# Add the src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file and apply overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
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


def create_experiment_config(unified_config: Dict[str, Any], output_dir: Path) -> GeneralAlignmentConfig:
    """Convert unified config format to GeneralAlignmentConfig."""
    
    # Extract configuration sections
    data_config = unified_config.get('data', {})
    model_config = unified_config.get('model', {})
    training_config = unified_config.get('training', {})
    analysis_config = unified_config.get('analysis', {})
    pruning_config = unified_config.get('pruning', {})
    dropout_config = unified_config.get('dropout', {})
    eigenfeature_config = unified_config.get('eigenfeature', {})
    output_config = unified_config.get('output', {})
    
    # Build GeneralAlignmentConfig parameters
    config_dict = {
        'name': unified_config.get('experiment_name', 'unified_experiment'),
        'seed': unified_config.get('seed', 42),
        'device': unified_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        
        # Data configuration
        'dataset_name': data_config.get('dataset', 'mnist'),
        'batch_size': data_config.get('batch_size', 128),
        'num_workers': data_config.get('num_workers', 4),
        
        # Model configuration
        'model_name': model_config.get('architecture', 'mlp'),
        'model_config': {
            'hidden_dims': model_config.get('hidden_sizes', [512, 256]),
            'activation_type': model_config.get('activation', 'relu'),
            'dropout_rate': model_config.get('dropout_rate', 0.0),
            'input_dim': 784 if data_config.get('dataset') == 'mnist' else 3072 if data_config.get('dataset') == 'cifar10' else 784,
            'output_dim': 10 if data_config.get('dataset') in ['mnist', 'cifar10'] else 100,
        },
        
        # Training configuration
        'do_train': training_config.get('do_train', True),
        'training_epochs': training_config.get('epochs', 10),
        'learning_rate': training_config.get('optimizer', {}).get('learning_rate', 0.001),
        'optimizer': training_config.get('optimizer', {}).get('type', 'adam'),
        
        # Analysis configuration
        'measure_alignment_during_training': analysis_config.get('compute_metrics', True),
        'alignment_methods': analysis_config.get('metrics', ['rayleigh_quotient']),
        
        # Pruning configuration
        'do_pruning_experiments': pruning_config.get('enabled', False),
        'pruning_strategies': pruning_config.get('algorithms', ['magnitude']),
        'pruning_amounts': pruning_config.get('sparsity_levels', [0.1, 0.3, 0.5, 0.7, 0.9]),
        'pruning_selection_mode': pruning_config.get('selection_mode', 'low'),
        'pruning_alignment_metric': pruning_config.get('alignment_metric', 'rayleigh_quotient'),
        'pruning_scope': pruning_config.get('scope', 'layer'),
        'fine_tune_after_pruning': pruning_config.get('fine_tune_after_pruning', True),
        'fine_tune_epochs': pruning_config.get('fine_tune_epochs', 10),
        
        # Dropout analysis
        'do_dropout_analysis': dropout_config.get('enabled', False),
        'dropout_rates': dropout_config.get('rates', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]),
        
        # Eigenfeature analysis
        'do_eigenfeature_analysis': eigenfeature_config.get('enabled', False),
        
        # Output configuration
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'log_dir': str(output_dir / 'logs'),
        'save_intermediate_results': output_config.get('save_results', True),
        'generate_plots': output_config.get('generate_plots', True),
        'plot_dpi': output_config.get('plot_dpi', 300),
    }
    
    return GeneralAlignmentConfig(**config_dict)


def organize_results(output_dir: Path, experiment_name: str) -> None:
    """Organize experiment results into a clean structure."""
    
    # Create summary files
    summary_file = output_dir / 'experiment_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # List all generated files
        f.write("Generated Files:\n")
        for file_path in sorted(output_dir.rglob('*')):
            if file_path.is_file():
                relative_path = file_path.relative_to(output_dir)
                f.write(f"  - {relative_path}\n")
    
    print(f"\n✓ Results organized in: {output_dir}")
    print(f"✓ Summary created: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run unified alignment experiment')
    parser.add_argument('--config', type=str, default='configs/unified_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--experiment-name', type=str, help='Override experiment name')
    
    # Add common overrides
    parser.add_argument('--device', type=str, help='Override device (cuda/cpu)')
    parser.add_argument('--epochs', type=int, help='Override training epochs')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning experiments')
    parser.add_argument('--no-pruning', action='store_true', help='Disable pruning experiments')
    
    args = parser.parse_args()
    
    # Load configuration
    overrides = {}
    if args.device:
        overrides['device'] = args.device
    if args.epochs:
        overrides['training.epochs'] = args.epochs
    if args.pruning:
        overrides['pruning.enabled'] = True
    if args.no_pruning:
        overrides['pruning.enabled'] = False
    if args.experiment_name:
        overrides['experiment_name'] = args.experiment_name
    
    config = load_config(args.config, overrides)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get('experiment_name', 'unified_experiment')
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/{experiment_name}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the configuration used
    config_save_path = output_dir / 'experiment_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    
    print(f"\n{'='*60}")
    print(f"Running Unified Alignment Experiment")
    print(f"{'='*60}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {config.get('device', 'cuda')}")
    print(f"{'='*60}\n")
    
    # Create experiment configuration
    experiment_config = create_experiment_config(config, output_dir)
    
    # Run experiment
    experiment = GeneralAlignmentExperiment(experiment_config)
    results = experiment.run()
    
    # Save results
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
    
    # Organize results
    organize_results(output_dir, experiment_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    
    if 'test_results' in results:
        print(f"Final model accuracy: {results['test_results']['final_accuracy']:.2f}%")
        print(f"Final model loss: {results['test_results']['final_loss']:.4f}")
    
    if 'pruning_results' in results and results['pruning_results']:
        print("\nPruning experiments completed.")
        strategies = results['pruning_results'].get('strategies', {})
        print(f"Strategies tested: {list(strategies.keys())}")
    
    print(f"\nAll results saved in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main() 