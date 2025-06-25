#!/usr/bin/env python3
"""
Unified Alignment Experiment Runner

This script provides a single entry point for all alignment experiments.
It can handle any combination of:
- Datasets: MNIST, CIFAR10/100, ImageNet, etc.
- Models: MLP, CNN, ResNet, VGG, etc.
- Metrics: Rayleigh Quotient, Mutual Information, CKA, etc.
- Pruning: Magnitude, Gradient, Fisher, Random, etc.
- Experiment Types: Standard pruning, Progressive dropout, Layer-wise analysis, etc.

Usage:
    python unified_experiment.py --config configs/unified_config.yaml
"""

import argparse
import logging
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alignment.experiments.base import BaseExperiment
from alignment.experiments.runner import ExperimentRunner
from alignment.pruning.experiments import (
    LayerIsolatedPruningExperiment,
    CascadingLayerPruningExperiment
)
from alignment.models import AlignmentModel
from alignment.data import create_dataloader
from alignment.metrics import RayleighQuotient, WeightCosineSimilarity
from alignment.pruning import PruningStrategy
from alignment.analysis.reporting import ReportGenerator
from alignment.analysis.visualization import (
    plot_pruning_results, 
    plot_layer_importance,
    plot_alignment_heatmap
)

logger = logging.getLogger(__name__)


class UnifiedExperiment:
    """Unified experiment class that can run any type of alignment experiment."""
    
    def __init__(self, config_path: str, overrides: Optional[Dict] = None):
        """Initialize unified experiment.
        
        Args:
            config_path: Path to configuration file
            overrides: Optional dictionary of config overrides
        """
        self.config = self._load_config(config_path, overrides)
        self._setup_logging()
        self._setup_paths()
        self.device = torch.device(self.config['device'])
        
        logger.info(f"Initialized unified experiment: {self.config['experiment_name']}")
        logger.info(f"Experiment type: {self.config['experiment_type']}")
        
    def _load_config(self, config_path: str, overrides: Optional[Dict] = None) -> Dict:
        """Load and merge configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        if overrides:
            def update_nested(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_nested(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            config = update_nested(config, overrides)
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        
        # Console handler
        handlers = []
        if log_config.get('console', True):
            handlers.append(logging.StreamHandler(sys.stdout))
        
        # File handler
        if log_config.get('file', True):
            log_file = os.path.join(self.results_path, 'experiment.log')
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def _setup_paths(self):
        """Setup experiment paths."""
        base_path = self.config.get('results_path', 'results')
        
        if self.config.get('use_timestamp', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.results_path = os.path.join(base_path, f"{self.config['experiment_name']}_{timestamp}")
        else:
            self.results_path = os.path.join(base_path, self.config['experiment_name'])
        
        os.makedirs(self.results_path, exist_ok=True)
        
        # Save config
        config_save_path = os.path.join(self.results_path, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f)
    
    def create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_config = self.config['model']
        model_name = model_config['name'].lower()
        
        if model_name == 'mlp':
            from alignment.models.architectures import create_mlp
            mlp_cfg = model_config.get('mlp_config', {})
            model = create_mlp(
                input_dim=mlp_cfg['input_dim'],
                hidden_dims=mlp_cfg['hidden_dims'],
                output_dim=model_config['output_dim'],
                activation=mlp_cfg.get('activation', 'relu'),
                dropout_rate=model_config.get('dropout_rate', 0.0)
            )
        
        elif model_name == 'cnn':
            from alignment.models.architectures import create_cnn
            cnn_cfg = model_config.get('cnn_config', {})
            model = create_cnn(
                in_channels=cnn_cfg['in_channels'],
                conv_channels=cnn_cfg['conv_channels'],
                kernel_sizes=cnn_cfg['kernel_sizes'],
                output_dim=model_config['output_dim'],
                fc_dims=cnn_cfg.get('fc_dims', [128])
            )
        
        elif model_name in ['resnet18', 'resnet50', 'alexnet', 'vgg16']:
            # Use torchvision models
            import torchvision.models as models
            model_class = getattr(models, model_name)
            model = model_class(
                pretrained=model_config.get('pretrained', False),
                num_classes=model_config['output_dim']
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Wrap in AlignmentModel
        model = AlignmentModel(model)
        return model.to(self.device)
    
    def create_dataset(self):
        """Create dataset based on configuration."""
        dataset_config = self.config['dataset']
        
        train_loader = create_dataloader(
            dataset_name=dataset_config['name'],
            data_path=dataset_config.get('data_path', './data'),
            batch_size=dataset_config['batch_size'],
            train=True,
            num_workers=dataset_config.get('num_workers', 4),
            download=dataset_config.get('download', True),
            normalize=dataset_config.get('normalize', True)
        )
        
        test_loader = create_dataloader(
            dataset_name=dataset_config['name'],
            data_path=dataset_config.get('data_path', './data'),
            batch_size=dataset_config['batch_size'],
            train=False,
            num_workers=dataset_config.get('num_workers', 4),
            download=dataset_config.get('download', True),
            normalize=dataset_config.get('normalize', True)
        )
        
        return train_loader, test_loader
    
    def create_metrics(self) -> List:
        """Create metrics based on configuration."""
        metrics = []
        metric_names = self.config['alignment'].get('metrics', ['rayleigh_quotient'])
        
        for metric_name in metric_names:
            if metric_name.lower() in ['rayleigh_quotient', 'rq']:
                metrics.append(RayleighQuotient())
            elif metric_name.lower() in ['weight_cosine_similarity', 'wcs']:
                metrics.append(WeightCosineSimilarity())
            # Add more metrics as needed
        
        return metrics
    
    def run(self):
        """Run the experiment based on configuration."""
        experiment_type = self.config['experiment_type']
        
        logger.info(f"Running {experiment_type} experiment")
        
        if experiment_type == 'standard_pruning':
            results = self._run_standard_pruning()
        
        elif experiment_type == 'progressive_dropout':
            results = self._run_progressive_dropout()
        
        elif experiment_type == 'alignment_analysis':
            results = self._run_alignment_analysis()
        
        elif experiment_type == 'layer_isolated_pruning':
            results = self._run_layer_isolated_pruning()
        
        elif experiment_type == 'cascading_layer_pruning':
            results = self._run_cascading_layer_pruning()
        
        elif experiment_type == 'training_alignment':
            results = self._run_training_alignment()
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        if self.config['visualization'].get('generate_plots', True):
            self._generate_visualizations(results)
        
        # Generate report
        self._generate_report(results)
        
        logger.info("Experiment completed successfully")
        return results
    
    def _run_standard_pruning(self) -> Dict:
        """Run standard pruning experiment."""
        # Create model and dataset
        model = self.create_model()
        train_loader, test_loader = self.create_dataset()
        
        # Train model
        if self.config['training'].get('epochs', 0) > 0:
            model = self._train_model(model, train_loader, test_loader)
        
        # Evaluate baseline
        baseline_acc = self._evaluate_model(model, test_loader)
        
        # Apply pruning
        pruning_config = self.config['pruning']
        strategy = PruningStrategy(pruning_config['strategy'])
        
        pruned_model = strategy.prune(
            model,
            amount=pruning_config['amount'],
            scope=pruning_config.get('scope', 'global')
        )
        
        # Evaluate pruned model
        pruned_acc = self._evaluate_model(pruned_model, test_loader)
        
        # Fine-tune if configured
        if pruning_config.get('fine_tune', True):
            pruned_model = self._fine_tune_model(
                pruned_model, 
                train_loader, 
                test_loader,
                epochs=pruning_config.get('fine_tune_epochs', 5)
            )
            fine_tuned_acc = self._evaluate_model(pruned_model, test_loader)
        else:
            fine_tuned_acc = pruned_acc
        
        return {
            'baseline_accuracy': baseline_acc,
            'pruned_accuracy': pruned_acc,
            'fine_tuned_accuracy': fine_tuned_acc,
            'pruning_amount': pruning_config['amount'],
            'pruning_strategy': pruning_config['strategy']
        }
    
    def _run_progressive_dropout(self) -> Dict:
        """Run progressive dropout experiment."""
        # Create model and dataset
        model = self.create_model()
        train_loader, test_loader = self.create_dataset()
        metrics = self.create_metrics()
        
        # Train model
        if self.config['training'].get('epochs', 0) > 0:
            model = self._train_model(model, train_loader, test_loader)
        
        # Progressive dropout parameters
        pruning_config = self.config['pruning']
        dropout_rates = np.linspace(
            pruning_config['dropout_min'],
            pruning_config['dropout_max'],
            pruning_config['dropout_steps']
        )
        
        results = {
            'dropout_rates': dropout_rates.tolist(),
            'strategies': {}
        }
        
        # Run each strategy
        for strategy in pruning_config.get('strategies', ['high', 'low', 'random']):
            logger.info(f"Running {strategy} strategy")
            
            accuracies = []
            for rate in tqdm(dropout_rates, desc=f"{strategy} dropout"):
                # Apply dropout based on strategy
                pruned_model = self._apply_strategic_dropout(
                    model.clone(), 
                    rate, 
                    strategy, 
                    metrics[0]
                )
                
                # Evaluate
                acc = self._evaluate_model(pruned_model, test_loader)
                accuracies.append(acc)
            
            results['strategies'][strategy] = accuracies
        
        return results
    
    def _run_alignment_analysis(self) -> Dict:
        """Run comprehensive alignment analysis."""
        # This combines multiple analyses
        results = {}
        
        # Run progressive dropout
        results['progressive_dropout'] = self._run_progressive_dropout()
        
        # Add layer importance analysis
        model = self.create_model()
        train_loader, test_loader = self.create_dataset()
        metrics = self.create_metrics()
        
        if self.config['training'].get('epochs', 0) > 0:
            model = self._train_model(model, train_loader, test_loader)
        
        # Compute layer importance
        layer_scores = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                scores = metrics[0].compute_scores(module, train_loader)
                layer_scores[name] = {
                    'mean': float(scores.mean()),
                    'std': float(scores.std()),
                    'max': float(scores.max()),
                    'min': float(scores.min())
                }
        
        results['layer_importance'] = layer_scores
        
        return results
    
    def _run_layer_isolated_pruning(self) -> Dict:
        """Run layer-isolated pruning experiment."""
        config = self._create_experiment_config()
        experiment = LayerIsolatedPruningExperiment(config)
        return experiment.run()
    
    def _run_cascading_layer_pruning(self) -> Dict:
        """Run cascading layer pruning experiment."""
        config = self._create_experiment_config()
        experiment = CascadingLayerPruningExperiment(config)
        return experiment.run()
    
    def _run_training_alignment(self) -> Dict:
        """Track alignment metrics during training."""
        model = self.create_model()
        train_loader, test_loader = self.create_dataset()
        metrics = self.create_metrics()
        
        training_config = self.config['training']
        track_interval = self.config['experiment_specific'].get('track_interval', 10)
        
        results = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'alignment_scores': {metric.name: [] for metric in metrics}
        }
        
        optimizer = self._create_optimizer(model, training_config)
        
        for epoch in range(training_config['epochs']):
            # Training
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, epoch
            )
            
            # Evaluation
            test_loss, test_acc = self._evaluate_with_loss(model, test_loader)
            
            # Compute alignment metrics
            metric_scores = {}
            for metric in metrics:
                scores = metric.compute(model, train_loader)
                metric_scores[metric.name] = float(scores.mean())
            
            # Store results
            results['epochs'].append(epoch)
            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
            
            for metric_name, score in metric_scores.items():
                results['alignment_scores'][metric_name].append(score)
            
            logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%"
            )
        
        return results
    
    def _train_model(self, model: nn.Module, train_loader, test_loader) -> nn.Module:
        """Train the model."""
        training_config = self.config['training']
        optimizer = self._create_optimizer(model, training_config)
        
        for epoch in range(training_config['epochs']):
            train_loss, train_acc = self._train_epoch(
                model, train_loader, optimizer, epoch
            )
            
            if epoch % training_config.get('eval_interval', 1) == 0:
                test_acc = self._evaluate_model(model, test_loader)
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                    f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%"
                )
        
        return model
    
    def _create_optimizer(self, model: nn.Module, training_config: Dict):
        """Create optimizer based on config."""
        opt_name = training_config.get('optimizer', 'adam').lower()
        lr = training_config.get('learning_rate', 0.001)
        
        if opt_name == 'adam':
            return torch.optim.Adam(
                model.parameters(), 
                lr=lr,
                weight_decay=training_config.get('weight_decay', 0)
            )
        elif opt_name == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=training_config.get('momentum', 0.9),
                weight_decay=training_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _train_epoch(self, model, train_loader, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _evaluate_model(self, model, test_loader):
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def _evaluate_with_loss(self, model, test_loader):
        """Evaluate model with loss."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _apply_strategic_dropout(self, model, rate, strategy, metric):
        """Apply dropout based on strategy."""
        # Compute importance scores
        scores = []
        modules = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module_scores = metric.compute_scores(module)
                scores.extend(module_scores.flatten().cpu().numpy())
                modules.append((module, len(module_scores.flatten())))
        
        scores = np.array(scores)
        
        # Determine indices to drop based on strategy
        num_to_drop = int(len(scores) * rate)
        
        if strategy == 'high':
            # Drop highest scoring weights
            indices = np.argsort(scores)[-num_to_drop:]
        elif strategy == 'low':
            # Drop lowest scoring weights
            indices = np.argsort(scores)[:num_to_drop]
        else:  # random
            indices = np.random.choice(len(scores), num_to_drop, replace=False)
        
        # Create mask
        mask = np.ones(len(scores), dtype=bool)
        mask[indices] = False
        
        # Apply mask to model
        idx = 0
        for module, size in modules:
            module_mask = mask[idx:idx+size].reshape(module.weight.shape)
            module.weight.data *= torch.tensor(module_mask, device=module.weight.device).float()
            idx += size
        
        return model
    
    def _save_results(self, results: Dict):
        """Save experiment results."""
        # Save as JSON
        results_path = os.path.join(self.results_path, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as compressed numpy if configured
        if self.config['logging'].get('compression', True):
            np_results_path = os.path.join(self.results_path, 'results.npz')
            np.savez_compressed(np_results_path, **results)
        
        logger.info(f"Results saved to {self.results_path}")
    
    def _generate_visualizations(self, results: Dict):
        """Generate visualization plots."""
        viz_config = self.config['visualization']
        plot_types = viz_config.get('plot_types', ['dropout_accuracy'])
        
        for plot_type in plot_types:
            if plot_type == 'dropout_accuracy' and 'progressive_dropout' in results:
                self._plot_dropout_accuracy(results['progressive_dropout'])
            
            elif plot_type == 'layer_importance' and 'layer_importance' in results:
                self._plot_layer_importance(results['layer_importance'])
            
            elif plot_type == 'training_curves' and 'train_loss' in results:
                self._plot_training_curves(results)
        
        logger.info(f"Visualizations saved to {self.results_path}")
    
    def _plot_dropout_accuracy(self, dropout_results):
        """Plot dropout vs accuracy curves."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['figure_size'])
        
        dropout_rates = dropout_results['dropout_rates']
        
        for strategy, accuracies in dropout_results['strategies'].items():
            ax.plot(dropout_rates, accuracies, marker='o', label=strategy)
        
        ax.set_xlabel('Dropout Rate')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Progressive Dropout Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.results_path, 'dropout_accuracy.png')
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def _plot_layer_importance(self, layer_scores):
        """Plot layer importance scores."""
        import matplotlib.pyplot as plt
        
        layers = list(layer_scores.keys())
        means = [layer_scores[l]['mean'] for l in layers]
        stds = [layer_scores[l]['std'] for l in layers]
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['figure_size'])
        
        x = np.arange(len(layers))
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Importance Score')
        ax.set_title('Layer Importance Analysis')
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'layer_importance.png')
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def _plot_training_curves(self, results):
        """Plot training curves."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = results['epochs']
        
        # Loss curves
        ax1.plot(epochs, results['train_loss'], label='Train')
        ax1.plot(epochs, results['test_loss'], label='Test')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, results['train_acc'], label='Train')
        ax2.plot(epochs, results['test_acc'], label='Test')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_path, 'training_curves.png')
        plt.savefig(save_path, dpi=self.config['visualization']['dpi'])
        plt.close()
    
    def _generate_report(self, results: Dict):
        """Generate experiment report."""
        report = ReportGenerator(self.config, results)
        report_path = os.path.join(self.results_path, 'report.html')
        report.generate(report_path)
        logger.info(f"Report generated: {report_path}")
    
    def _create_experiment_config(self):
        """Create config object for legacy experiment classes."""
        # Convert dict config to object-like config for compatibility
        class ConfigObject:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, ConfigObject(v))
                    else:
                        setattr(self, k, v)
        
        return ConfigObject(self.config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Unified Alignment Experiment Runner'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to configuration file'
    )
    
    # Allow config overrides from command line
    parser.add_argument('--experiment_type', type=str, help='Override experiment type')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--dataset.name', type=str, help='Override dataset name')
    parser.add_argument('--model.name', type=str, help='Override model name')
    parser.add_argument('--training.epochs', type=int, help='Override training epochs')
    
    args = parser.parse_args()
    
    # Build overrides dict from command line args
    overrides = {}
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            # Handle nested keys like 'dataset.name'
            keys = key.split('.')
            d = overrides
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
    
    # Run experiment
    experiment = UnifiedExperiment(args.config, overrides)
    results = experiment.run()
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {experiment.results_path}")


if __name__ == '__main__':
    main() 