#!/usr/bin/env python3
"""Test script to verify pruning logic fixes."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.data.datasets import MNISTDataset


def create_simple_mlp():
    """Create a simple MLP for testing."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def test_pruning_strategies():
    """Test different pruning strategies with multiple networks."""
    print("Testing pruning fixes...")
    
    # Configuration for quick test
    config = GeneralAlignmentConfig(
        name="test_pruning_fix",
        dataset_name="mnist",
        model_name="mlp",
        model_config={
            "hidden_dims": [256, 128],
            "activation": "relu",
            "dropout_rate": 0.0
        },
        data_path="./data",
        
        # Multi-network settings
        num_networks=3,  # Test with 3 networks for variance
        
        # Quick training
        do_train=True,
        training_epochs=5,  # Quick training
        batch_size=128,
        learning_rate=0.01,
        
        # Pruning settings
        do_pruning_experiments=True,
        pruning_strategies=["magnitude", "alignment", "random"],
        pruning_amounts=[0.1, 0.3, 0.5, 0.7, 0.9],
        pruning_selection_mode=["low", "high", "random"],  # Test all modes
        fine_tune_after_pruning=True,
        fine_tune_epochs=2,  # Quick fine-tuning
        alignment_structured_pruning=True,  # Use structured pruning for alignment
        
        # Disable other analyses for speed
        do_dropout_analysis=False,
        do_eigenfeature_analysis=False,
        
        # Visualization
        generate_plots=True,
        log_dir="./test_pruning_results"
    )
    
    # Run experiment
    experiment = GeneralAlignmentExperiment(config)
    results = experiment.run()
    
    # Analyze results
    print("\n=== Pruning Results Analysis ===")
    pruning_results = results.get("pruning_results", {})
    
    if "strategies" in pruning_results:
        for strategy_key, strategy_data in pruning_results["strategies"].items():
            print(f"\nStrategy: {strategy_key}")
            
            if "accuracies_before_finetune" in strategy_data:
                accs_before = strategy_data["accuracies_before_finetune"]
                accs_after = strategy_data["accuracies_after_finetune"]
                
                # Check if we have standard deviations
                if "accuracies_before_finetune_std" in strategy_data:
                    stds_before = strategy_data["accuracies_before_finetune_std"]
                    stds_after = strategy_data["accuracies_after_finetune_std"]
                    
                    print("Sparsity | Before FT (mean±std) | After FT (mean±std)")
                    print("-" * 60)
                    for i, sparsity in enumerate(strategy_data["sparsities"]):
                        print(f"{sparsity:.1%} | {accs_before[i]:.1f}±{stds_before[i]:.1f}% | "
                              f"{accs_after[i]:.1f}±{stds_after[i]:.1f}%")
                else:
                    print("Sparsity | Before FT | After FT")
                    print("-" * 40)
                    for i, sparsity in enumerate(strategy_data["sparsities"]):
                        print(f"{sparsity:.1%} | {accs_before[i]:.1f}% | {accs_after[i]:.1f}%")
    
    # Check for expected behavior
    print("\n=== Validation Checks ===")
    
    # 1. Random strategy should not be consistently the best
    random_strategies = [k for k in pruning_results["strategies"].keys() if "random" in k]
    if random_strategies:
        print(f"Found {len(random_strategies)} random strategy results")
        
        # Compare random selection mode across different strategies
        for strategy in ["magnitude", "alignment"]:
            low_key = f"{strategy}_low"
            high_key = f"{strategy}_high"
            random_key = f"{strategy}_random"
            
            if all(k in pruning_results["strategies"] for k in [low_key, high_key, random_key]):
                low_accs = pruning_results["strategies"][low_key]["accuracies_after_finetune"]
                high_accs = pruning_results["strategies"][high_key]["accuracies_after_finetune"]
                random_accs = pruning_results["strategies"][random_key]["accuracies_after_finetune"]
                
                # Random should typically be between low and high
                for i in range(len(low_accs)):
                    if low_accs[i] < high_accs[i]:
                        expected_range = (low_accs[i], high_accs[i])
                    else:
                        expected_range = (high_accs[i], low_accs[i])
                    
                    in_range = expected_range[0] <= random_accs[i] <= expected_range[1]
                    print(f"{strategy} @ {pruning_results['strategies'][low_key]['sparsities'][i]:.1%}: "
                          f"Random ({random_accs[i]:.1f}%) {'✓' if in_range else '✗'} in range "
                          f"[{expected_range[0]:.1f}, {expected_range[1]:.1f}]")
    
    # 2. Check that error bars show reasonable variance
    if "accuracies_before_finetune_std" in next(iter(pruning_results["strategies"].values())):
        print("\n✓ Error bars are being computed")
        
        # Check that standard deviations are reasonable (not zero, not too large)
        for strategy_key, strategy_data in pruning_results["strategies"].items():
            stds = strategy_data["accuracies_after_finetune_std"]
            avg_std = np.mean(stds)
            print(f"{strategy_key}: Average std = {avg_std:.2f}%")
    else:
        print("\n✗ No error bars found - check if multiple networks were used")
    
    print("\nTest completed! Check ./test_pruning_results/plots/ for visualizations.")


if __name__ == "__main__":
    test_pruning_strategies() 