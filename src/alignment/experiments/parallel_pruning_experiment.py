"""
Parallel experiment runner for multi-network pruning analysis.

This module provides efficient parallel training and analysis of multiple networks
with different seeds, computing metrics and performing pruning experiments.
"""

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..analysis.visualization.pruning_plots import PruningVisualizer
from ..metrics import get_metric
from ..models import ModelWrapper
from ..pruning import PruningConfig, get_pruning_strategy
from ..training.multi_network import train_networks_fully_tensorized

logger = logging.getLogger(__name__)


@dataclass
class ParallelExperimentConfig:
    """Configuration for parallel pruning experiments."""
    num_networks: int = 5
    seeds: Optional[List[int]] = None
    model_class: type = None
    model_kwargs: Dict[str, Any] = None
    dataset_name: str = 'mnist'
    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    device: str = 'cuda'
    
    # Pruning settings
    pruning_strategies: List[str] = None
    pruning_modes: List[str] = None
    sparsity_levels: List[float] = None
    fine_tune_epochs: int = 5
    
    # Metrics settings
    metrics_to_compute: List[str] = None
    compute_rayleigh: bool = True
    
    # Output settings
    output_dir: str = 'results/parallel_pruning'
    save_checkpoints: bool = True
    create_visualizations: bool = True
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = list(range(42, 42 + self.num_networks))
        if self.pruning_strategies is None:
            self.pruning_strategies = ['magnitude', 'gradient', 'random']
        if self.pruning_modes is None:
            self.pruning_modes = ['low', 'high']
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        if self.metrics_to_compute is None:
            self.metrics_to_compute = ['rayleigh_quotient', 'mutual_information']


class ParallelPruningExperiment:
    """
    Run parallel pruning experiments on multiple networks.
    
    This class handles:
    1. Training multiple networks with different seeds
    2. Computing alignment metrics (RQ, MI, etc.)
    3. Applying various pruning strategies
    4. Generating comprehensive visualizations
    """
    
    def __init__(self, config: ParallelExperimentConfig):
        """
        Initialize parallel experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = PruningVisualizer()
        
        # Setup multiprocessing
        self.num_workers = min(config.num_networks, mp.cpu_count())
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete parallel experiment.
        
        Returns:
            Dictionary containing all results
        """
        logger.info(f"Starting parallel pruning experiment with {self.config.num_networks} networks")
        
        # Phase 1: Train networks in parallel
        logger.info("Phase 1: Training networks...")
        networks, training_history = self._train_networks_parallel()
        
        # Phase 2: Compute initial metrics
        logger.info("Phase 2: Computing initial metrics...")
        initial_metrics = self._compute_metrics_parallel(networks)
        
        # Phase 3: Run pruning experiments
        logger.info("Phase 3: Running pruning experiments...")
        pruning_results = self._run_pruning_experiments_parallel(networks)
        
        # Phase 4: Generate visualizations
        logger.info("Phase 4: Generating visualizations...")
        if self.config.create_visualizations:
            self._create_visualizations(pruning_results)
        
        # Phase 5: Save results
        logger.info("Phase 5: Saving results...")
        results = {
            'config': self.config.__dict__,
            'training_history': training_history,
            'initial_metrics': initial_metrics,
            'pruning_results': pruning_results,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S')
        }
        self._save_results(results)
        
        logger.info("Experiment complete!")
        return results
    
    def _train_networks_parallel(self) -> Tuple[List[nn.Module], Dict[str, Any]]:
        """Train multiple networks in parallel."""
        # Get data loaders
        train_loader, val_loader = self._get_data_loaders()
        
        # Create networks with different seeds
        networks = []
        for seed in self.config.seeds[:self.config.num_networks]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = self.config.model_class(**self.config.model_kwargs)
            networks.append(model)
        
        # Train using tensorized approach if possible
        if len(networks) > 1 and self._can_use_tensorized_training(networks):
            logger.info(f"Using tensorized training for {len(networks)} networks")
            networks, history = train_networks_fully_tensorized(
                networks=networks,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.epochs,
                optimizer_kwargs={'lr': self.config.learning_rate},
                device=self.config.device,
                checkpoint_dir=self.output_dir / 'checkpoints' if self.config.save_checkpoints else None
            )
        else:
            # Fallback to parallel training with multiprocessing
            logger.info(f"Using parallel training with {self.num_workers} workers")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for i, (network, seed) in enumerate(zip(networks, self.config.seeds)):
                    future = executor.submit(
                        self._train_single_network,
                        network, train_loader, val_loader, seed, i
                    )
                    futures.append(future)
                
                trained_networks = []
                histories = []
                for future in futures:
                    net, hist = future.result()
                    trained_networks.append(net)
                    histories.append(hist)
                
                networks = trained_networks
                history = self._aggregate_histories(histories)
        
        return networks, history
    
    def _compute_metrics_parallel(self, networks: List[nn.Module]) -> Dict[str, Any]:
        """Compute metrics for all networks in parallel."""
        results = {}
        
        # Use thread pool for metric computation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for metric_name in self.config.metrics_to_compute:
                for i, network in enumerate(networks):
                    key = (metric_name, i)
                    future = executor.submit(
                        self._compute_single_metric,
                        network, metric_name
                    )
                    futures[key] = future
            
            # Collect results
            for (metric_name, net_idx), future in futures.items():
                if metric_name not in results:
                    results[metric_name] = []
                results[metric_name].append(future.result())
        
        # Compute statistics
        stats = {}
        for metric_name, values in results.items():
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        return stats
    
    def _run_pruning_experiments_parallel(
        self, 
        networks: List[nn.Module]
    ) -> Dict[str, Any]:
        """Run pruning experiments on all networks."""
        results = {}
        
        # Get data loaders for evaluation
        train_loader, val_loader = self._get_data_loaders()
        
        # Iterate through strategies and modes
        for strategy_name in self.config.pruning_strategies:
            for mode in self.config.pruning_modes:
                strategy_key = f"{strategy_name}_{mode}"
                logger.info(f"Running {strategy_key} pruning...")
                
                # Results for this strategy across all networks
                strategy_results = []
                
                # Process each network
                for net_idx, network in enumerate(networks):
                    seed_results = {}
                    
                    # Test different sparsity levels
                    for sparsity in self.config.sparsity_levels:
                        # Clone network for this experiment
                        net_copy = self._clone_network(network)
                        
                        # Apply pruning
                        config = PruningConfig(
                            amount=sparsity,
                            pruning_mode=mode
                        )
                        strategy = get_pruning_strategy(strategy_name, config=config)
                        
                        # Prune all layers
                        for name, module in net_copy.named_modules():
                            if isinstance(module, (nn.Linear, nn.Conv2d)):
                                strategy.prune(module)
                        
                        # Fine-tune if requested
                        if self.config.fine_tune_epochs > 0:
                            self._fine_tune_network(
                                net_copy, train_loader, val_loader,
                                epochs=self.config.fine_tune_epochs
                            )
                        
                        # Evaluate
                        accuracy, loss = self._evaluate_network(net_copy, val_loader)
                        
                        seed_results[sparsity] = {
                            'accuracy': accuracy,
                            'loss': loss
                        }
                    
                    strategy_results.append(seed_results)
                
                # Aggregate results across seeds
                results[strategy_key] = self._aggregate_pruning_results(strategy_results)
        
        return results
    
    def _aggregate_pruning_results(
        self, 
        seed_results: List[Dict[float, Dict[str, float]]]
    ) -> Dict[float, Dict[str, Any]]:
        """Aggregate pruning results across multiple seeds."""
        aggregated = {}
        
        # Get all sparsity levels
        sparsities = sorted(seed_results[0].keys())
        
        for sparsity in sparsities:
            metrics = {}
            
            # Collect values for each metric
            for metric in ['accuracy', 'loss']:
                values = [sr[sparsity][metric] for sr in seed_results]
                
                metrics[metric] = values
                metrics[f'{metric}_mean'] = np.mean(values)
                metrics[f'{metric}_std'] = np.std(values)
            
            aggregated[sparsity] = {
                'mean': {
                    'accuracy': metrics['accuracy_mean'],
                    'loss': metrics['loss_mean']
                },
                'std': {
                    'accuracy': metrics['accuracy_std'],
                    'loss': metrics['loss_std']
                },
                'raw_values': {
                    'accuracy': metrics['accuracy'],
                    'loss': metrics['loss']
                }
            }
        
        return aggregated
    
    def _create_visualizations(self, pruning_results: Dict[str, Any]):
        """Create comprehensive visualizations."""
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # 1. Main performance comparison
        self.visualizer.plot_pruning_performance(
            pruning_results,
            metrics=['accuracy', 'loss'],
            save_path=vis_dir / 'performance_comparison.png',
            title='Pruning Strategy Performance Comparison',
            show_confidence=True
        )
        
        # 2. Comprehensive comparison grid
        # Convert format for grid plot
        grid_results = {}
        for strategy_key, strategy_data in pruning_results.items():
            grid_results[strategy_key] = {}
            for sparsity, data in strategy_data.items():
                grid_results[strategy_key][sparsity] = {
                    'accuracy': data['mean']['accuracy'],
                    'loss': data['mean']['loss']
                }
        
        self.visualizer.plot_pruning_comparison_grid(
            grid_results,
            save_path=vis_dir / 'comparison_grid.png'
        )
        
        # 3. Multi-seed analysis
        # Reorganize data by strategy
        seed_results = {}
        for strategy_key in pruning_results:
            seed_results[strategy_key] = []
            
            # Extract raw values for each seed
            for seed_idx in range(self.config.num_networks):
                seed_data = {}
                for sparsity, data in pruning_results[strategy_key].items():
                    seed_data[sparsity] = {
                        'accuracy': data['raw_values']['accuracy'][seed_idx],
                        'loss': data['raw_values']['loss'][seed_idx]
                    }
                seed_results[strategy_key].append(seed_data)
        
        self.visualizer.plot_multi_seed_results(
            seed_results,
            metric='accuracy',
            save_path=vis_dir / 'multi_seed_accuracy.png'
        )
        
        self.visualizer.plot_multi_seed_results(
            seed_results,
            metric='loss',
            save_path=vis_dir / 'multi_seed_loss.png'
        )
    
    def _save_results(self, results: Dict[str, Any]):
        """Save all results to disk."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        # Save as JSON
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Parallel Pruning Experiment Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Networks: {self.config.num_networks}\n")
            f.write(f"  - Strategies: {self.config.pruning_strategies}\n")
            f.write(f"  - Modes: {self.config.pruning_modes}\n")
            f.write(f"  - Sparsity levels: {self.config.sparsity_levels}\n")
            f.write(f"\nBest performing strategies:\n")
            
            # Find best strategies
            for sparsity in [0.5, 0.7, 0.9]:
                f.write(f"\nAt {sparsity*100:.0f}% sparsity:\n")
                best_acc = 0
                best_strategy = None
                
                for strategy, data in results['pruning_results'].items():
                    if sparsity in data:
                        acc = data[sparsity]['mean']['accuracy']
                        if acc > best_acc:
                            best_acc = acc
                            best_strategy = strategy
                
                if best_strategy:
                    f.write(f"  Best: {best_strategy} ({best_acc:.2f}%)\n")
        
        logger.info(f"Results saved to {self.output_dir}")
    
    # Helper methods
    def _get_data_loaders(self):
        """Get data loaders for the specified dataset."""
        # This is a placeholder - implement based on your data module
        from ..data.datasets import get_dataset
        return get_dataset(
            self.config.dataset_name,
            batch_size=self.config.batch_size
        )
    
    def _can_use_tensorized_training(self, networks: List[nn.Module]) -> bool:
        """Check if networks can be trained using tensorized approach."""
        # Simple check - all networks should have same architecture
        if len(networks) < 2:
            return False
        
        base_arch = str(networks[0])
        for net in networks[1:]:
            if str(net) != base_arch:
                return False
        return True
    
    def _train_single_network(
        self, 
        network: nn.Module,
        train_loader,
        val_loader,
        seed: int,
        idx: int
    ) -> Tuple[nn.Module, Dict]:
        """Train a single network (for multiprocessing)."""
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Move to device
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        network = network.to(device)
        
        # Training loop
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.config.epochs):
            # Train
            network.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
            
            # Validate
            val_loss, val_acc = self._evaluate_network(network, val_loader)
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(100. * correct / total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        
        return network.cpu(), history
    
    def _compute_single_metric(self, network: nn.Module, metric_name: str) -> float:
        """Compute a single metric for a network."""
        metric = get_metric(metric_name)()
        
        # Get a sample batch for metric computation
        train_loader, _ = self._get_data_loaders()
        inputs, _ = next(iter(train_loader))
        inputs = inputs.to(self.device)
        
        network = network.to(self.device)
        
        # Compute metric (simplified - implement based on your metrics)
        with torch.no_grad():
            if metric_name == 'rayleigh_quotient':
                # Example for first layer
                if hasattr(network, 'fc1'):
                    scores = metric.compute(inputs=inputs, weights=network.fc1.weight)
                else:
                    # Find first linear layer
                    for module in network.modules():
                        if isinstance(module, nn.Linear):
                            scores = metric.compute(inputs=inputs, weights=module.weight)
                            break
                return scores.mean().item()
            else:
                # Implement other metrics
                return 0.0
    
    def _clone_network(self, network: nn.Module) -> nn.Module:
        """Create a deep copy of a network."""
        import copy
        return copy.deepcopy(network)
    
    def _fine_tune_network(
        self,
        network: nn.Module,
        train_loader,
        val_loader,
        epochs: int
    ):
        """Fine-tune a pruned network."""
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        network = network.to(device)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate * 0.1)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            network.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def _evaluate_network(
        self,
        network: nn.Module,
        val_loader
    ) -> Tuple[float, float]:
        """Evaluate a network."""
        device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        network = network.to(device)
        network.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return accuracy, avg_loss
    
    def _aggregate_histories(self, histories: List[Dict]) -> Dict:
        """Aggregate training histories from multiple networks."""
        aggregated = {}
        
        for key in histories[0].keys():
            values = [h[key] for h in histories]
            aggregated[key] = {
                'mean': np.mean(values, axis=0).tolist(),
                'std': np.std(values, axis=0).tolist(),
                'all': values
            }
        
        return aggregated


def run_parallel_pruning_experiment(
    model_class: type,
    model_kwargs: Dict[str, Any],
    num_networks: int = 5,
    dataset_name: str = 'mnist',
    output_dir: str = 'results/parallel_pruning',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run a parallel pruning experiment.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model construction
        num_networks: Number of networks to train
        dataset_name: Name of dataset to use
        output_dir: Directory to save results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing all results
    """
    config = ParallelExperimentConfig(
        num_networks=num_networks,
        model_class=model_class,
        model_kwargs=model_kwargs,
        dataset_name=dataset_name,
        output_dir=output_dir,
        **kwargs
    )
    
    experiment = ParallelPruningExperiment(config)
    return experiment.run() 