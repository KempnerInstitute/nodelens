#!/usr/bin/env python3
"""Visualize pruning experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_path):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_pruning_comparison(results, output_path=None):
    """Create comparison plots for pruning strategies."""
    pruning_results = results.get('pruning_results', {}).get('strategies', {})
    
    if not pruning_results:
        print("No pruning results found!")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different strategies
    colors = {'magnitude': 'blue', 'alignment': 'red', 'hybrid': 'green'}
    
    # Plot 1: Accuracy vs Sparsity
    for strategy, data in pruning_results.items():
        if 'sparsities' in data and 'accuracies_after_finetune' in data:
            sparsities = [s * 100 for s in data['sparsities']]  # Convert to percentage
            accuracies = data['accuracies_after_finetune']
            
            ax1.plot(sparsities, accuracies, 'o-', label=f'{strategy} (after fine-tune)', 
                    color=colors.get(strategy, 'gray'), linewidth=2, markersize=8)
            
            # Also plot before fine-tuning with dashed line
            if 'accuracies_before_finetune' in data:
                acc_before = data['accuracies_before_finetune']
                ax1.plot(sparsities, acc_before, 'o--', label=f'{strategy} (before fine-tune)', 
                        color=colors.get(strategy, 'gray'), alpha=0.5, linewidth=1.5, markersize=6)
    
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Sparsity for Different Pruning Strategies', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Accuracy retention
    for strategy, data in pruning_results.items():
        if all(k in data for k in ['pruning_amounts', 'accuracies_before_finetune', 'accuracies_after_finetune']):
            amounts = [a * 100 for a in data['pruning_amounts']]
            acc_before = data['accuracies_before_finetune']
            acc_after = data['accuracies_after_finetune']
            
            # Calculate accuracy retention (relative to unpruned model)
            baseline_acc = results.get('test_results', {}).get('final_accuracy', 100)
            retention_before = [a / baseline_acc * 100 for a in acc_before]
            retention_after = [a / baseline_acc * 100 for a in acc_after]
            
            ax2.plot(amounts, retention_after, 'o-', label=f'{strategy}', 
                    color=colors.get(strategy, 'gray'), linewidth=2, markersize=8)
    
    ax2.set_xlabel('Pruning Amount (%)', fontsize=12)
    ax2.set_ylabel('Accuracy Retention (%)', fontsize=12)
    ax2.set_title('Accuracy Retention vs Pruning Amount', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Baseline')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary(results):
    """Print a summary of the results."""
    print("\n=== Experiment Summary ===")
    
    # Model info
    config = results.get('config', {})
    print(f"Model: {config.get('model_name', 'Unknown')}")
    print(f"Dataset: {config.get('dataset_name', 'Unknown')}")
    print(f"Training epochs: {config.get('training_epochs', 'Unknown')}")
    
    # Final performance
    test_results = results.get('test_results', {})
    print(f"\nFinal Performance (before pruning):")
    print(f"  Accuracy: {test_results.get('final_accuracy', 'N/A'):.2f}%")
    print(f"  Loss: {test_results.get('final_loss', 'N/A'):.4f}")
    
    # Pruning results
    pruning_results = results.get('pruning_results', {}).get('strategies', {})
    
    print("\n=== Pruning Results ===")
    for strategy, data in pruning_results.items():
        print(f"\n{strategy.upper()} Pruning:")
        
        if all(k in data for k in ['pruning_amounts', 'sparsities', 
                                   'accuracies_before_finetune', 'accuracies_after_finetune']):
            for i, amount in enumerate(data['pruning_amounts']):
                sparsity = data['sparsities'][i]
                acc_before = data['accuracies_before_finetune'][i]
                acc_after = data['accuracies_after_finetune'][i]
                improvement = acc_after - acc_before
                
                print(f"  {amount*100:.0f}% pruning:")
                print(f"    Actual sparsity: {sparsity*100:.1f}%")
                print(f"    Accuracy: {acc_before:.2f}% → {acc_after:.2f}% (Δ{improvement:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize pruning experiment results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output", help="Output path for plot (if not specified, shows plot)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting, only show summary")
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    
    # Print summary
    print_summary(results)
    
    # Create plots
    if not args.no_plot:
        plot_pruning_comparison(results, args.output)


if __name__ == "__main__":
    main() 