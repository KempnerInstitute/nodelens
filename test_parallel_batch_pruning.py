#!/usr/bin/env python3
"""Test script to verify parallel batch pruning implementation."""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from alignment.experiments.general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig

# Create a simple config
config = GeneralAlignmentConfig(
    name="parallel_batch_test",
    dataset_name="mnist",
    model_name="mlp",
    model_config={"hidden_dims": [256, 128], "activation_type": "relu"},
    batch_size=1024,
    num_networks=5,
    training_epochs=2,  # Quick training
    do_train=True,
    do_pruning_experiments=True,
    pruning_strategies=["magnitude", "random", "alignment"],
    pruning_amounts=[0.0, 0.2, 0.4, 0.6, 0.8],
    pruning_selection_mode=["low", "high", "random"],
    fine_tune_after_pruning=False,  # No fine-tuning for speed
    eval_batches=10,  # Only evaluate on 10 batches for speed
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Total configurations: 5 networks × 5 sparsity levels × 3 selection modes × 3 strategies = 225 configs

print("Creating experiment...")
experiment = GeneralAlignmentExperiment(config)

print("\nRunning training...")
start_time = time.time()
experiment.run()
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

print("\nRunning parallel batch pruning experiments...")
print(f"Total configurations to evaluate: {config.num_networks} × {len(config.pruning_amounts)} × {len(config.pruning_selection_mode)} × {len(config.pruning_strategies)} = {config.num_networks * len(config.pruning_amounts) * len(config.pruning_selection_mode) * len(config.pruning_strategies)}")

start_time = time.time()
results = experiment._pruning_experiments()
pruning_time = time.time() - start_time

print(f"\nPruning experiments completed in {pruning_time:.2f} seconds")
print(f"Average time per configuration: {pruning_time / (config.num_networks * len(config.pruning_amounts) * len(config.pruning_selection_mode) * len(config.pruning_strategies)):.3f} seconds")

# Print sample results
for strategy_key, strategy_data in results["strategies"].items():
    accs = strategy_data["accuracies_before_finetune"]
    print(f"\n{strategy_key}: {[f'{acc:.1f}%' for acc in accs]}")

print("\nParallel batch pruning test completed successfully!") 