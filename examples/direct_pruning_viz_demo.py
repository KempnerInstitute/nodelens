"""
Direct pruning visualization demo - imports only what's needed.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Direct imports to avoid circular dependencies
from alignment.analysis.visualization.pruning_plots import PruningVisualizer

def create_sample_results():
    """Create sample pruning results for visualization."""
    
    # Sample results for different strategies
    results = {
        'magnitude_low': {
            0.1: {'accuracy': 94.5, 'loss': 0.18},
            0.3: {'accuracy': 92.8, 'loss': 0.25},
            0.5: {'accuracy': 89.2, 'loss': 0.38},
            0.7: {'accuracy': 83.5, 'loss': 0.55},
            0.9: {'accuracy': 71.2, 'loss': 0.92}
        },
        'magnitude_high': {
            0.1: {'accuracy': 88.3, 'loss': 0.35},
            0.3: {'accuracy': 75.6, 'loss': 0.68},
            0.5: {'accuracy': 58.2, 'loss': 1.15},
            0.7: {'accuracy': 35.8, 'loss': 1.82},
            0.9: {'accuracy': 15.3, 'loss': 2.45}
        },
        'random': {
            0.1: {'accuracy': 92.1, 'loss': 0.23},
            0.3: {'accuracy': 87.5, 'loss': 0.42},
            0.5: {'accuracy': 79.8, 'loss': 0.68},
            0.7: {'accuracy': 68.2, 'loss': 1.05},
            0.9: {'accuracy': 42.5, 'loss': 1.65}
        }
    }
    
    # Add some statistical variation for multi-seed demo
    results_with_stats = {}
    for strategy, data in results.items():
        results_with_stats[strategy] = {}
        for sparsity, metrics in data.items():
            results_with_stats[strategy][sparsity] = {
                'mean': metrics,
                'std': {
                    'accuracy': np.random.uniform(0.5, 2.0),
                    'loss': np.random.uniform(0.02, 0.08)
                }
            }
    
    return results, results_with_stats


def main():
    """Run visualization demo."""
    print("=" * 60)
    print("Direct Pruning Visualization Demo")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path('results/direct_viz_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = PruningVisualizer()
    
    # Get sample results
    results, results_with_stats = create_sample_results()
    
    print("\nCreating visualizations...")
    
    # 1. Basic performance plot
    print("\n1. Creating performance comparison plot...")
    fig1 = visualizer.plot_pruning_performance(
        results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'performance_basic.png',
        title='Pruning Strategy Comparison',
        show_confidence=False
    )
    print(f"   Saved to: {output_dir / 'performance_basic.png'}")
    
    # 2. Performance plot with confidence intervals
    print("\n2. Creating performance plot with confidence intervals...")
    fig2 = visualizer.plot_pruning_performance(
        results_with_stats,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / 'performance_with_confidence.png',
        title='Pruning Strategy Comparison (with confidence intervals)',
        show_confidence=True
    )
    print(f"   Saved to: {output_dir / 'performance_with_confidence.png'}")
    
    # 3. Comprehensive comparison grid
    print("\n3. Creating comprehensive comparison grid...")
    fig3 = visualizer.plot_pruning_comparison_grid(
        results,
        save_path=output_dir / 'comparison_grid.png'
    )
    print(f"   Saved to: {output_dir / 'comparison_grid.png'}")
    
    # 4. Multi-seed results
    print("\n4. Creating multi-seed analysis...")
    
    # Generate multi-seed data
    seed_results = {}
    for strategy in ['magnitude_low', 'magnitude_high', 'random']:
        seed_results[strategy] = []
        
        base_results = results[strategy]
        for seed in range(5):
            np.random.seed(seed)
            seed_data = {}
            
            for sparsity, metrics in base_results.items():
                seed_data[sparsity] = {
                    'accuracy': metrics['accuracy'] + np.random.normal(0, 1.5),
                    'loss': metrics['loss'] + np.random.normal(0, 0.05)
                }
            
            seed_results[strategy].append(seed_data)
    
    fig4 = visualizer.plot_multi_seed_results(
        seed_results,
        metric='accuracy',
        save_path=output_dir / 'multi_seed_accuracy.png'
    )
    print(f"   Saved to: {output_dir / 'multi_seed_accuracy.png'}")
    
    fig5 = visualizer.plot_multi_seed_results(
        seed_results,
        metric='loss',
        save_path=output_dir / 'multi_seed_loss.png'
    )
    print(f"   Saved to: {output_dir / 'multi_seed_loss.png'}")
    
    print("\n" + "=" * 60)
    print("Visualization Demo Complete!")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  1. performance_basic.png - Simple accuracy/loss curves")
    print("  2. performance_with_confidence.png - With confidence intervals")
    print("  3. comparison_grid.png - 6-panel comprehensive analysis")
    print("  4. multi_seed_accuracy.png - Accuracy across multiple seeds")
    print("  5. multi_seed_loss.png - Loss across multiple seeds")
    
    print("\nKey insights from the visualizations:")
    print("  - magnitude_low: Prunes small weights, maintains accuracy longer")
    print("  - magnitude_high: Prunes large weights, rapid accuracy degradation")
    print("  - random: Intermediate performance, as expected")


if __name__ == "__main__":
    main() 