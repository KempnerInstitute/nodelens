"""
Simple demo for pruning visualization without circular imports.

This demonstrates the visualization capabilities without running
the full parallel experiment framework.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import only what we need
from alignment.analysis.visualization.pruning_plots import PruningVisualizer
from alignment.pruning.strategies import (
    MagnitudePruning, 
    RandomPruning,
    ParallelModePruning
)
from alignment.pruning.base import PruningConfig


class SimpleNet(nn.Module):
    """Simple network for demonstration."""
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


def simulate_pruning_results():
    """Simulate pruning results for visualization demo."""
    print("=== Simulating Pruning Results ===\n")
    
    # Simulate results for different strategies
    strategies = ['magnitude_low', 'magnitude_high', 'random']
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = {}
    
    for strategy in strategies:
        results[strategy] = {}
        
        # Base accuracy depends on strategy
        if strategy == 'magnitude_low':
            base_acc = 95.0
            acc_decay = 0.15  # Slow decay
        elif strategy == 'magnitude_high':
            base_acc = 95.0
            acc_decay = 0.5   # Fast decay
        else:  # random
            base_acc = 95.0
            acc_decay = 0.25  # Medium decay
        
        for sparsity in sparsity_levels:
            # Simulate accuracy degradation
            accuracy = base_acc * (1 - sparsity * acc_decay)
            # Add some noise
            accuracy += np.random.normal(0, 1.5)
            
            # Simulate loss increase
            loss = 0.1 + sparsity * 0.8 * (1 + acc_decay)
            loss += np.random.normal(0, 0.05)
            
            results[strategy][sparsity] = {
                'accuracy': max(10, min(100, accuracy)),  # Clip to reasonable range
                'loss': max(0.05, loss)
            }
    
    return results


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("=== Pruning Visualization Demo ===\n")
    
    # Create output directory
    output_dir = Path('results/visualization_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = PruningVisualizer()
    
    # 1. Get simulated results
    results = simulate_pruning_results()
    
    print("Generated results for strategies:")
    for strategy in results:
        print(f"  - {strategy}")
    
    # 2. Create performance comparison plot
    print("\nCreating performance comparison plot...")
    fig1 = visualizer.plot_pruning_performance(
        results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'performance_comparison.png',
        title='Pruning Strategy Performance Comparison',
        show_confidence=False  # No confidence intervals for simulated data
    )
    print(f"Saved to: {output_dir / 'performance_comparison.png'}")
    
    # 3. Create comprehensive comparison grid
    print("\nCreating comprehensive comparison grid...")
    fig2 = visualizer.plot_pruning_comparison_grid(
        results,
        save_path=output_dir / 'comparison_grid.png'
    )
    print(f"Saved to: {output_dir / 'comparison_grid.png'}")
    
    # 4. Simulate multi-seed results
    print("\nSimulating multi-seed results...")
    seed_results = {}
    
    for strategy in ['magnitude_low', 'magnitude_high', 'random']:
        seed_results[strategy] = []
        
        # Simulate 5 different seeds
        for seed in range(5):
            np.random.seed(seed)
            seed_data = {}
            
            for sparsity in [0.3, 0.5, 0.7, 0.9]:
                base_result = results[strategy][sparsity]
                # Add seed-specific variation
                seed_data[sparsity] = {
                    'accuracy': base_result['accuracy'] + np.random.normal(0, 2),
                    'loss': base_result['loss'] + np.random.normal(0, 0.1)
                }
            
            seed_results[strategy].append(seed_data)
    
    # 5. Create multi-seed analysis plot
    print("\nCreating multi-seed analysis plot...")
    fig3 = visualizer.plot_multi_seed_results(
        seed_results,
        metric='accuracy',
        save_path=output_dir / 'multi_seed_accuracy.png'
    )
    print(f"Saved to: {output_dir / 'multi_seed_accuracy.png'}")
    
    print(f"\nAll visualizations saved to: {output_dir}")
    
    return results


def demo_pruning_strategies():
    """Demonstrate different pruning strategies."""
    print("\n\n=== Pruning Strategies Demo ===\n")
    
    # Create a simple model
    model = SimpleNet(hidden_size=128)
    
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    print("Model architecture:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"  {name}: {param.shape}")
    
    # Test different pruning configurations
    sparsity = 0.5
    
    print(f"\nApplying {sparsity*100}% sparsity with different strategies:\n")
    
    # 1. Magnitude pruning (low mode)
    print("1. Magnitude Pruning (low mode - prune small weights):")
    config_low = PruningConfig(amount=sparsity, pruning_mode='low')
    pruner_low = MagnitudePruning(config_low)
    
    # Apply to first layer
    importance_scores = pruner_low.compute_importance_scores(model.fc1)
    mask_low = pruner_low.create_pruning_mask(importance_scores)
    
    print(f"   Kept weights: {mask_low.sum().item()}/{mask_low.numel()}")
    print(f"   Actual sparsity: {(mask_low == 0).float().mean().item():.2%}")
    
    # 2. Magnitude pruning (high mode)
    print("\n2. Magnitude Pruning (high mode - prune large weights):")
    config_high = PruningConfig(amount=sparsity, pruning_mode='high')
    pruner_high = MagnitudePruning(config_high)
    
    mask_high = pruner_high.create_pruning_mask(importance_scores)
    
    print(f"   Kept weights: {mask_high.sum().item()}/{mask_high.numel()}")
    print(f"   Actual sparsity: {(mask_high == 0).float().mean().item():.2%}")
    
    # 3. Random pruning
    print("\n3. Random Pruning:")
    config_random = PruningConfig(amount=sparsity)
    pruner_random = RandomPruning(config_random)
    
    mask_random = pruner_random.create_pruning_mask(
        pruner_random.compute_importance_scores(model.fc1)
    )
    
    print(f"   Kept weights: {mask_random.sum().item()}/{mask_random.numel()}")
    print(f"   Actual sparsity: {(mask_random == 0).float().mean().item():.2%}")
    
    # 4. Compare weight distributions
    print("\n4. Weight distribution analysis:")
    weights = model.fc1.weight.data.flatten()
    
    kept_low = weights[mask_low.flatten().bool()]
    pruned_low = weights[~mask_low.flatten().bool()]
    
    kept_high = weights[mask_high.flatten().bool()]
    pruned_high = weights[~mask_high.flatten().bool()]
    
    print(f"\n   Low pruning (keep large weights):")
    print(f"     Avg magnitude of kept: {kept_low.abs().mean():.4f}")
    print(f"     Avg magnitude of pruned: {pruned_low.abs().mean():.4f}")
    
    print(f"\n   High pruning (keep small weights):")
    print(f"     Avg magnitude of kept: {kept_high.abs().mean():.4f}")
    print(f"     Avg magnitude of pruned: {pruned_high.abs().mean():.4f}")
    
    # 5. Parallel mode pruning
    print("\n5. Parallel Mode Pruning Demo:")
    parallel_strategy = ParallelModePruning(
        modes=['low', 'high', 'random'],
        base_strategy='magnitude'
    )
    
    result = parallel_strategy.prune_parallel(model.fc1, amount=sparsity)
    
    print(f"\n   Sparsity by mode:")
    for mode, sparse in result.sparsities.items():
        print(f"     {mode}: {sparse:.2%}")
    
    # Check overlap
    overlap_low_high = (result.masks['low'] * result.masks['high']).sum().item()
    total = result.masks['low'].numel()
    print(f"\n   Overlap between low and high: {overlap_low_high/total:.2%}")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Simple Pruning Visualization Demo")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. Visualization demo
        results = demo_visualization()
        
        # 2. Pruning strategies demo
        demo_pruning_strategies()
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nCheck the 'results/visualization_demo/' directory for generated plots:")
        print("  - performance_comparison.png: Accuracy and loss curves")
        print("  - comparison_grid.png: Comprehensive strategy comparison")
        print("  - multi_seed_accuracy.png: Statistical analysis across seeds")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 