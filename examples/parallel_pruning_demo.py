"""
Demo script for parallel pruning experiments with visualization.

This script demonstrates:
1. Training multiple networks in parallel
2. Applying different pruning strategies
3. Creating comprehensive visualizations
4. Statistical analysis across multiple seeds
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

from alignment.experiments.parallel_pruning_experiment import (
    ParallelPruningExperiment,
    ParallelExperimentConfig,
    run_parallel_pruning_experiment
)
from alignment.analysis.visualization import PruningVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNet(nn.Module):
    """Simple network for MNIST."""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def demo_basic_parallel_experiment():
    """Run a basic parallel pruning experiment."""
    print("=== Basic Parallel Pruning Experiment ===\n")
    
    # Run experiment with convenience function
    results = run_parallel_pruning_experiment(
        model_class=SimpleNet,
        model_kwargs={'hidden_size': 256},
        num_networks=3,  # Use 3 networks for quick demo
        dataset_name='mnist',
        epochs=5,  # Fewer epochs for demo
        sparsity_levels=[0.3, 0.5, 0.7, 0.9],
        pruning_strategies=['magnitude', 'random'],
        pruning_modes=['low', 'high'],
        output_dir='results/demo_parallel_basic'
    )
    
    print("\nExperiment completed!")
    print(f"Results saved to: results/demo_parallel_basic")
    
    return results


def demo_advanced_parallel_experiment():
    """Run an advanced parallel experiment with custom configuration."""
    print("\n\n=== Advanced Parallel Pruning Experiment ===\n")
    
    # Create custom configuration
    config = ParallelExperimentConfig(
        num_networks=5,
        seeds=[42, 43, 44, 45, 46],
        model_class=SimpleNet,
        model_kwargs={'hidden_size': 512},
        dataset_name='mnist',
        batch_size=128,
        epochs=10,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        
        # Pruning settings
        pruning_strategies=['magnitude', 'gradient', 'fisher', 'random'],
        pruning_modes=['low', 'high'],
        sparsity_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        fine_tune_epochs=3,
        
        # Metrics settings
        metrics_to_compute=['rayleigh_quotient', 'mutual_information'],
        compute_rayleigh=True,
        
        # Output settings
        output_dir='results/demo_parallel_advanced',
        save_checkpoints=True,
        create_visualizations=True
    )
    
    # Create and run experiment
    experiment = ParallelPruningExperiment(config)
    results = experiment.run()
    
    print("\nAdvanced experiment completed!")
    print(f"Check results/demo_parallel_advanced for detailed results and visualizations")
    
    return results


def demo_custom_visualization():
    """Demonstrate custom visualization capabilities."""
    print("\n\n=== Custom Visualization Demo ===\n")
    
    # Create sample results for visualization
    sample_results = {
        'magnitude_low': {
            0.3: {'mean': {'accuracy': 95.2, 'loss': 0.15}, 'std': {'accuracy': 0.8, 'loss': 0.02}},
            0.5: {'mean': {'accuracy': 93.1, 'loss': 0.22}, 'std': {'accuracy': 1.2, 'loss': 0.03}},
            0.7: {'mean': {'accuracy': 88.5, 'loss': 0.35}, 'std': {'accuracy': 1.5, 'loss': 0.04}},
            0.9: {'mean': {'accuracy': 75.3, 'loss': 0.68}, 'std': {'accuracy': 2.1, 'loss': 0.06}}
        },
        'magnitude_high': {
            0.3: {'mean': {'accuracy': 85.7, 'loss': 0.42}, 'std': {'accuracy': 1.5, 'loss': 0.05}},
            0.5: {'mean': {'accuracy': 72.3, 'loss': 0.78}, 'std': {'accuracy': 2.3, 'loss': 0.08}},
            0.7: {'mean': {'accuracy': 45.2, 'loss': 1.45}, 'std': {'accuracy': 3.5, 'loss': 0.12}},
            0.9: {'mean': {'accuracy': 15.8, 'loss': 2.35}, 'std': {'accuracy': 2.8, 'loss': 0.15}}
        },
        'random': {
            0.3: {'mean': {'accuracy': 91.5, 'loss': 0.28}, 'std': {'accuracy': 1.0, 'loss': 0.03}},
            0.5: {'mean': {'accuracy': 85.2, 'loss': 0.48}, 'std': {'accuracy': 1.8, 'loss': 0.05}},
            0.7: {'mean': {'accuracy': 72.1, 'loss': 0.85}, 'std': {'accuracy': 2.5, 'loss': 0.08}},
            0.9: {'mean': {'accuracy': 35.6, 'loss': 1.75}, 'std': {'accuracy': 3.2, 'loss': 0.12}}
        }
    }
    
    # Create visualizer
    visualizer = PruningVisualizer()
    
    # Create output directory
    output_dir = Path('results/demo_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance comparison with confidence intervals
    fig1 = visualizer.plot_pruning_performance(
        sample_results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'performance_comparison.png',
        title='Pruning Strategy Comparison (Low vs High vs Random)',
        show_confidence=True
    )
    
    # 2. Comprehensive comparison grid
    fig2 = visualizer.plot_pruning_comparison_grid(
        sample_results,
        save_path=output_dir / 'comparison_grid.png'
    )
    
    # 3. Multi-seed analysis (simulate multiple seeds)
    seed_results = {}
    for strategy in ['magnitude_low', 'magnitude_high', 'random']:
        seed_results[strategy] = []
        for seed in range(5):
            seed_data = {}
            for sparsity in [0.3, 0.5, 0.7, 0.9]:
                # Add some random variation
                base_acc = sample_results[strategy][sparsity]['mean']['accuracy']
                base_loss = sample_results[strategy][sparsity]['mean']['loss']
                
                seed_data[sparsity] = {
                    'accuracy': base_acc + np.random.randn() * 2,
                    'loss': base_loss + np.random.randn() * 0.05
                }
            seed_results[strategy].append(seed_data)
    
    fig3 = visualizer.plot_multi_seed_results(
        seed_results,
        metric='accuracy',
        save_path=output_dir / 'multi_seed_analysis.png'
    )
    
    print(f"Visualizations saved to {output_dir}")
    print("Generated plots:")
    print("  - performance_comparison.png: Accuracy and loss curves with confidence")
    print("  - comparison_grid.png: Comprehensive strategy comparison")
    print("  - multi_seed_analysis.png: Statistical analysis across seeds")


def demo_quick_analysis():
    """Quick analysis of pruning impact on different layers."""
    print("\n\n=== Quick Pruning Analysis ===\n")
    
    # Create a simple model
    model = SimpleNet(hidden_size=256)
    
    # Analyze pruning impact
    from alignment.pruning import get_pruning_strategy, PruningConfig
    
    sparsity_levels = [0.3, 0.5, 0.7, 0.9]
    strategies = ['magnitude', 'random']
    modes = ['low', 'high']
    
    print("Layer-wise parameter counts:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: {param.numel()} parameters")
    
    print("\nPruning analysis:")
    for strategy in strategies:
        for mode in modes:
            print(f"\n{strategy.capitalize()} pruning (mode={mode}):")
            
            for sparsity in sparsity_levels:
                config = PruningConfig(amount=sparsity, pruning_mode=mode)
                pruner = get_pruning_strategy(strategy, config=config)
                
                # Count zeros after pruning
                total_params = 0
                pruned_params = 0
                
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Clone weights
                        weights = module.weight.data.clone()
                        
                        # Apply pruning
                        mask = pruner.create_pruning_mask(
                            pruner.compute_importance_scores(module)
                        )
                        
                        pruned_weights = weights * mask
                        zeros = (pruned_weights == 0).sum().item()
                        
                        total_params += weights.numel()
                        pruned_params += zeros
                
                actual_sparsity = pruned_params / total_params
                print(f"  Target: {sparsity*100:.0f}%, Actual: {actual_sparsity*100:.1f}%")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Parallel Pruning Experiments Demo")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demos
    try:
        # 1. Basic parallel experiment
        basic_results = demo_basic_parallel_experiment()
        
        # 2. Advanced parallel experiment (optional - takes longer)
        # Uncomment to run:
        # advanced_results = demo_advanced_parallel_experiment()
        
        # 3. Custom visualization
        demo_custom_visualization()
        
        # 4. Quick analysis
        demo_quick_analysis()
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nCheck the 'results/' directory for:")
    print("  - results/demo_parallel_basic/: Basic experiment results")
    print("  - results/demo_visualization/: Custom visualization examples")
    print("\nKey features demonstrated:")
    print("  ✓ Parallel training of multiple networks")
    print("  ✓ Multiple pruning strategies (magnitude, gradient, random)")
    print("  ✓ High/low pruning modes")
    print("  ✓ Statistical analysis across seeds")
    print("  ✓ Comprehensive visualizations")
    print("  ✓ Automatic result aggregation")


if __name__ == "__main__":
    main() 