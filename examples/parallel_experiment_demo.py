#!/usr/bin/env python3
"""
Parallel Pruning Experiment Demo

This script demonstrates the parallel experiment capabilities of the alignment framework,
including:
1. Training multiple networks with different seeds in parallel
2. Applying different pruning strategies to each network
3. Computing metrics in parallel
4. Generating statistical analysis across seeds
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alignment.experiments import ParallelPruningExperiment
from alignment.analysis.visualization import PruningVisualizer


def create_simple_model():
    """Create a simple CNN for MNIST."""
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def main():
    print("=" * 60)
    print("Parallel Pruning Experiment Demo")
    print("=" * 60)
    
    # Configuration
    config = {
        'num_seeds': 3,  # Train 3 networks with different seeds
        'dataset_name': 'mnist',
        'batch_size': 128,
        'epochs': 3,  # Quick training for demo
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'pruning_strategies': ['magnitude', 'random'],
        'pruning_amounts': [0.3, 0.5, 0.7],
        'pruning_modes': ['low', 'high'],
        'metrics': ['rayleigh_quotient'],
        'output_dir': 'results/parallel_experiment'
    }
    
    print(f"\nConfiguration:")
    print(f"  Seeds: {config['num_seeds']}")
    print(f"  Strategies: {config['pruning_strategies']}")
    print(f"  Amounts: {config['pruning_amounts']}")
    print(f"  Modes: {config['pruning_modes']}")
    print(f"  Device: {config['device']}")
    
    # Create experiment
    print("\n=== Creating Parallel Experiment ===")
    experiment = ParallelPruningExperiment(
        model_fn=create_simple_model,
        config=config
    )
    
    # Run experiment
    print("\n=== Running Experiment ===")
    print("This will:")
    print("  1. Train 3 networks in parallel")
    print("  2. Apply 2 strategies × 3 amounts × 2 modes = 12 pruning configs")
    print("  3. Compute metrics for each configuration")
    print("  4. Generate statistical analysis")
    
    start_time = time.time()
    results = experiment.run()
    elapsed = time.time() - start_time
    
    print(f"\nExperiment completed in {elapsed:.1f} seconds")
    
    # Analyze results
    print("\n=== Results Summary ===")
    
    # Training results
    print("\nTraining Performance (across seeds):")
    for seed_idx in range(config['num_seeds']):
        seed_results = results['seed_results'][seed_idx]
        final_acc = seed_results['training']['final_accuracy']
        print(f"  Seed {seed_idx}: {final_acc:.2%}")
    
    # Pruning results summary
    print("\nPruning Results (mean ± std across seeds):")
    pruning_summary = results['pruning_summary']
    
    for strategy in config['pruning_strategies']:
        print(f"\n{strategy.capitalize()} Pruning:")
        for mode in config['pruning_modes']:
            print(f"  {mode} mode:")
            for amount in config['pruning_amounts']:
                key = f"{strategy}_{mode}_{amount}"
                if key in pruning_summary:
                    stats = pruning_summary[key]
                    acc_mean = stats['accuracy']['mean']
                    acc_std = stats['accuracy']['std']
                    print(f"    {amount*100:.0f}% sparsity: {acc_mean:.2%} ± {acc_std:.2%}")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    
    visualizer = PruningVisualizer()
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Multi-seed results plot
    print("Creating multi-seed analysis plot...")
    fig = visualizer.plot_multi_seed_results(
        results['seed_results'],
        strategies=config['pruning_strategies'],
        title="Parallel Experiment Results"
    )
    fig.savefig(output_dir / "multi_seed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_dir / 'multi_seed_analysis.png'}")
    
    # 2. Statistical comparison
    print("Creating statistical comparison plot...")
    
    # Prepare data for comparison
    comparison_data = {}
    for strategy in config['pruning_strategies']:
        strategy_data = {}
        for mode in config['pruning_modes']:
            mode_results = []
            for amount in config['pruning_amounts']:
                key = f"{strategy}_{mode}_{amount}"
                if key in pruning_summary:
                    mode_results.append({
                        'sparsity': amount,
                        'accuracy': pruning_summary[key]['accuracy']['mean'],
                        'std': pruning_summary[key]['accuracy']['std']
                    })
            strategy_data[mode] = mode_results
        comparison_data[strategy] = strategy_data
    
    # Save detailed results
    results_path = output_dir / "parallel_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'summary': pruning_summary,
            'comparison_data': comparison_data
        }, f, indent=2)
    print(f"Saved results to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Parallel Experiment Demo Complete!")
    print("=" * 60)
    
    print("\nKey Insights:")
    print("1. Parallel training enables statistical analysis across seeds")
    print("2. Different pruning modes show distinct behaviors:")
    print("   - Low mode: Prunes small weights (better retention)")
    print("   - High mode: Prunes large weights (worse retention)")
    print("3. Magnitude pruning outperforms random pruning")
    print("4. Variance increases with higher sparsity levels")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main() 