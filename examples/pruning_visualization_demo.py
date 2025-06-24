"""
Pruning Visualization Demo

This script demonstrates the visualization capabilities of the pruning framework,
including performance plots, comparison grids, and multi-seed analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from alignment.analysis.visualization import PruningVisualizer
from alignment.pruning import get_pruning_strategy, PruningConfig


def create_pruning_results_from_simulation():
    """Create simulated pruning results for visualization."""
    print("=== Creating Simulated Results ===\n")
    
    # Define strategies and their characteristics
    strategy_configs = {
        'magnitude_low': {'base_acc': 95.0, 'decay': 0.15},  # Best performance
        'magnitude_high': {'base_acc': 95.0, 'decay': 0.5},  # Worst performance
        'random': {'base_acc': 95.0, 'decay': 0.25}          # Medium performance
    }
    
    sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for strategy, config in strategy_configs.items():
        results[strategy] = {}
        
        for sparsity in sparsity_levels:
            # Simulate accuracy degradation
            accuracy = config['base_acc'] * (1 - sparsity * config['decay'])
            accuracy += np.random.normal(0, 1.5)  # Add noise
            
            # Simulate loss increase
            loss = 0.1 + sparsity * 0.8 * (1 + config['decay'])
            loss += np.random.normal(0, 0.05)
            
            results[strategy][sparsity] = {
                'accuracy': max(10, min(100, accuracy)),
                'loss': max(0.05, loss)
            }
    
    return results


def create_multi_seed_results(base_results, num_seeds=5):
    """Generate multi-seed results from base results."""
    seed_results = {}
    
    for strategy, sparsity_data in base_results.items():
        seed_results[strategy] = []
        
        for seed in range(num_seeds):
            np.random.seed(seed)
            seed_data = {}
            
            for sparsity, metrics in sparsity_data.items():
                seed_data[sparsity] = {
                    'accuracy': metrics['accuracy'] + np.random.normal(0, 2),
                    'loss': metrics['loss'] + np.random.normal(0, 0.1)
                }
            
            seed_results[strategy].append(seed_data)
    
    return seed_results


def demonstrate_real_pruning():
    """Demonstrate pruning on a real model to show actual behavior."""
    print("\n=== Real Pruning Demonstration ===\n")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
    
    # Test different pruning modes
    layer = model[0]
    sparsity = 0.5
    
    print(f"Testing pruning on layer with shape: {layer.weight.shape}")
    print(f"Target sparsity: {sparsity*100}%\n")
    
    results = {}
    for mode in ['low', 'high', 'random']:
        # Clone the layer to test each mode independently
        test_layer = nn.Linear(layer.in_features, layer.out_features)
        test_layer.weight.data = layer.weight.data.clone()
        
        config = PruningConfig(amount=sparsity, pruning_mode=mode)
        if mode == 'random':
            strategy = get_pruning_strategy('random', config=config)
        else:
            strategy = get_pruning_strategy('magnitude', config=config)
        
        # Get importance scores and create mask
        importance_scores = strategy.compute_importance_scores(test_layer)
        mask = strategy.create_pruning_mask(importance_scores)
        
        # Analyze results BEFORE applying the mask
        weights = test_layer.weight.data.flatten()
        mask_flat = mask.flatten()
        kept = weights[mask_flat.bool()]
        pruned = weights[~mask_flat.bool()]
        
        results[mode] = {
            'sparsity': (mask == 0).float().mean().item(),
            'kept_mean': kept.abs().mean().item() if kept.numel() > 0 else 0,
            'pruned_mean': pruned.abs().mean().item() if pruned.numel() > 0 else 0,
            'kept_count': kept.numel(),
            'pruned_count': pruned.numel()
        }
        
        print(f"{mode.capitalize()} mode:")
        print(f"  Actual sparsity: {results[mode]['sparsity']:.2%}")
        print(f"  Kept {results[mode]['kept_count']} weights, pruned {results[mode]['pruned_count']}")
        if results[mode]['kept_count'] > 0 and results[mode]['pruned_count'] > 0:
            print(f"  Avg magnitude kept: {results[mode]['kept_mean']:.4f}")
            print(f"  Avg magnitude pruned: {results[mode]['pruned_mean']:.4f}")
    
    return results


def main():
    """Run the visualization demonstration."""
    print("=" * 60)
    print("Pruning Visualization Demo")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path('results/pruning_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create simulated results
    results = create_pruning_results_from_simulation()
    
    # 2. Initialize visualizer
    visualizer = PruningVisualizer()
    
    # 3. Create basic performance plot
    print("\nCreating performance comparison plot...")
    fig1 = visualizer.plot_pruning_performance(
        results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'performance_comparison.png',
        title='Pruning Strategy Performance Comparison',
        show_confidence=False
    )
    print(f"Saved to: {output_dir / 'performance_comparison.png'}")
    
    # 4. Create comprehensive comparison grid
    print("\nCreating comprehensive comparison grid...")
    fig2 = visualizer.plot_pruning_comparison_grid(
        results,
        save_path=output_dir / 'comparison_grid.png'
    )
    print(f"Saved to: {output_dir / 'comparison_grid.png'}")
    
    # 5. Create multi-seed analysis
    print("\nCreating multi-seed analysis...")
    seed_results = create_multi_seed_results(results)
    
    fig3 = visualizer.plot_multi_seed_results(
        seed_results,
        metric='accuracy',
        save_path=output_dir / 'multi_seed_accuracy.png'
    )
    print(f"Saved to: {output_dir / 'multi_seed_accuracy.png'}")
    
    # 6. Demonstrate real pruning
    real_results = demonstrate_real_pruning()
    
    # Summary
    print("\n" + "=" * 60)
    print("Visualization Demo Complete!")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - performance_comparison.png: Accuracy and loss curves")
    print("  - comparison_grid.png: 6-panel comprehensive analysis")
    print("  - multi_seed_accuracy.png: Statistical analysis across seeds")
    
    print("\nKey insights:")
    print("  - Low mode (prune small): Best performance retention")
    print("  - High mode (prune large): Rapid performance degradation")
    print("  - Random mode: Intermediate performance")
    
    print(f"\nReal pruning results:")
    for mode in ['low', 'high', 'random']:
        result = real_results[mode]
        print(f"  - {mode.capitalize()} mode:")
        print(f"      Sparsity: {result['sparsity']:.2%}")
        if result['sparsity'] < 1.0:  # If not all weights were pruned
            print(f"      Kept {result['kept_count']} weights, pruned {result['pruned_count']}")
            if result['kept_count'] > 0 and result['pruned_count'] > 0:
                print(f"      Avg magnitude kept: {result['kept_mean']:.4f}")
                print(f"      Avg magnitude pruned: {result['pruned_mean']:.4f}")


if __name__ == "__main__":
    main() 