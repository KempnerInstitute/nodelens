"""
Demo script for parallel pruning strategies.

This script demonstrates how to:
1. Apply low, high, and random pruning simultaneously
2. Use tensorized pruning for efficient computation
3. Visualize pruning patterns across different modes
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from alignment.pruning import PruningConfig
from alignment.pruning.strategies import (
    ParallelModePruning,
    TensorizedPruning,
    MagnitudePruning
)


def create_simple_model():
    """Create a simple model for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def demo_parallel_modes():
    """Demonstrate parallel pruning with different modes."""
    print("=== Parallel Mode Pruning Demo ===\n")
    
    # Create model and config
    model = create_simple_model()
    config = PruningConfig(amount=0.5)
    
    # Initialize parallel pruning strategy
    strategy = ParallelModePruning(
        config=config,
        modes=['low', 'high', 'random'],
        base_strategy='magnitude'
    )
    
    # Apply parallel pruning to first layer
    layer = model[0]
    result = strategy.prune_parallel(layer)
    
    print(f"Original layer shape: {layer.weight.shape}")
    print(f"\nSparsity by mode:")
    for mode, sparsity in result.sparsities.items():
        print(f"  {mode}: {sparsity:.2%}")
    
    # Analyze mask overlap
    low_mask = result.masks['low']
    high_mask = result.masks['high']
    overlap = (low_mask * high_mask).sum().item()
    total = low_mask.numel()
    
    print(f"\nOverlap between low and high pruning: {overlap/total:.2%}")
    
    # Combine masks
    combined_union = strategy.combine_masks(result.masks, method='union')
    combined_intersection = strategy.combine_masks(result.masks, method='intersection')
    
    print(f"\nCombined sparsity (union): {(combined_union == 0).float().mean():.2%}")
    print(f"Combined sparsity (intersection): {(combined_intersection == 0).float().mean():.2%}")
    
    return result


def demo_tensorized_pruning():
    """Demonstrate tensorized pruning computation."""
    print("\n\n=== Tensorized Pruning Demo ===\n")
    
    # Create model
    model = create_simple_model()
    strategy = TensorizedPruning()
    
    # Compute pruning tensor for multiple sparsity levels
    layer = model[0]
    pruning_tensor = strategy.compute_pruning_tensor(
        layer,
        modes=['low', 'high', 'random'],
        amounts=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    
    print(f"Pruning tensor shape: {pruning_tensor.shape}")
    print(f"  [num_modes, num_amounts, *weight_shape]")
    
    # Analyze patterns
    analysis = strategy.analyze_pruning_patterns(pruning_tensor)
    
    print(f"\nSparsity progression:")
    sparsity_prog = analysis['sparsity_progression']
    for i, mode in enumerate(['low', 'high', 'random']):
        print(f"  {mode}: {sparsity_prog[i].tolist()}")
    
    return pruning_tensor, analysis


def demo_comparison_with_config():
    """Compare pruning with different configurations."""
    print("\n\n=== Pruning Mode Comparison ===\n")
    
    # Create model
    model = create_simple_model()
    layer = model[0]
    
    # Test different pruning modes
    modes_to_test = ['low', 'high']
    results = {}
    
    for mode in modes_to_test:
        config = PruningConfig(amount=0.7, pruning_mode=mode)
        strategy = MagnitudePruning(config)
        
        # Get importance scores
        importance = strategy.compute_importance_scores(layer)
        
        # Create mask
        mask = strategy.create_pruning_mask(importance)
        
        # Store results
        results[mode] = {
            'mask': mask,
            'kept_weights': layer.weight[mask.bool()],
            'pruned_weights': layer.weight[~mask.bool()],
            'avg_kept_magnitude': layer.weight[mask.bool()].abs().mean().item(),
            'avg_pruned_magnitude': layer.weight[~mask.bool()].abs().mean().item()
        }
    
    # Compare results
    print("Average magnitude of weights:")
    for mode, res in results.items():
        print(f"\n{mode.capitalize()} pruning:")
        print(f"  Kept weights: {res['avg_kept_magnitude']:.4f}")
        print(f"  Pruned weights: {res['avg_pruned_magnitude']:.4f}")
    
    return results


def visualize_pruning_patterns(pruning_tensor):
    """Visualize pruning patterns as heatmaps."""
    # Select a subset of weights to visualize (first 64x64)
    subset = pruning_tensor[:, :, :64, :64]
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    modes = ['Low', 'High', 'Random']
    amounts = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for i, mode in enumerate(modes):
        for j, amount in enumerate(amounts):
            ax = axes[i, j]
            im = ax.imshow(subset[i, j].cpu().numpy(), cmap='binary', vmin=0, vmax=1)
            ax.set_title(f'{mode} - {amount:.0%}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('pruning_patterns.png', dpi=150, bbox_inches='tight')
    print("\n\nVisualization saved to 'pruning_patterns.png'")


def main():
    """Run all demonstrations."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run demos
    parallel_result = demo_parallel_modes()
    pruning_tensor, analysis = demo_tensorized_pruning()
    comparison_results = demo_comparison_with_config()
    
    # Visualize if matplotlib is available
    try:
        visualize_pruning_patterns(pruning_tensor)
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")
    
    print("\n\n=== Demo Complete ===")


if __name__ == "__main__":
    main() 