#!/usr/bin/env python3
"""Test script to verify parallel pruning evaluation."""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.data.datasets import MNISTDataset

# Create a simple config
config = GeneralAlignmentConfig(
    name="parallel_test",
    dataset_name="mnist",
    model_name="mlp",
    model_config={"hidden_dims": [128, 64], "activation_type": "relu"},
    batch_size=1024,
    num_networks=3,
    training_epochs=0,  # Skip training
    do_train=False,  # Skip training
    do_pruning_experiments=True,
    pruning_strategies=["alignment"],
    pruning_amounts=[0.0, 0.3, 0.6, 0.9],
    pruning_selection_mode=["low", "high"],
    fine_tune_after_pruning=False,
    use_tensorized_pruning=True,
    use_optimized_pruning=True,
    use_ultra_parallel_eval=True,  # Enable ultra-parallel
    eval_batches=5,  # Only evaluate on 5 batches
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create experiment
print("Creating experiment...")
experiment = GeneralAlignmentExperiment(config)

# Initialize networks manually since we're skipping training
print(f"Initializing {config.num_networks} networks...")
for i, net in enumerate(experiment.networks):
    # Initialize with small random weights
    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# Run pruning experiments
print("\nRunning pruning experiments...")
print(f"Configurations: {len(config.pruning_amounts)} sparsity levels × {len(config.pruning_selection_mode)} modes × {config.num_networks} networks")
print(f"Total: {len(config.pruning_amounts) * len(config.pruning_selection_mode) * config.num_networks} configurations")

start_time = time.time()

# Call pruning experiments directly
results = experiment._pruning_experiments()

end_time = time.time()

print(f"\nPruning experiments completed in {end_time - start_time:.2f} seconds")

# Check results
if results and "strategies" in results:
    for strategy, data in results["strategies"].items():
        print(f"\nStrategy: {strategy}")
        if "accuracies_before_finetune" in data:
            accs = data["accuracies_before_finetune"]
            print(f"  Accuracies: {[f'{a:.1f}%' for a in accs]}")
else:
    print("No results returned!")

print("\nTest complete!") 