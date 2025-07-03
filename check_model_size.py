#!/usr/bin/env python3
"""Check actual model parameters from checkpoint."""

import torch
import sys
from pathlib import Path

# Find the most recent experiment
results_dir = Path("results")
experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and "mnist_alignment_analysis" in d.name]
latest_exp = sorted(experiment_dirs)[-1] if experiment_dirs else None

if not latest_exp:
    print("No experiment found")
    sys.exit(1)

print(f"Checking experiment: {latest_exp}")

# Find checkpoint
checkpoint_dir = latest_exp / "checkpoints"
checkpoints = list(checkpoint_dir.glob("*.pt"))

if not checkpoints:
    print("No checkpoints found")
    sys.exit(1)

# Load first checkpoint
checkpoint_path = checkpoints[0]
print(f"\nLoading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Analyze model structure
model_state = checkpoint['model_state_dict']
print("\n=== Model Structure ===")
for key, tensor in model_state.items():
    print(f"{key}: {tensor.shape} ({tensor.numel()} parameters)")

# Calculate total parameters
total_params = sum(tensor.numel() for tensor in model_state.values())
print(f"\nTotal parameters: {total_params:,}")

# Try to infer hidden sizes from layer shapes
print("\n=== Inferred Architecture ===")
layer_weights = [(k, v) for k, v in model_state.items() if 'weight' in k and 'bias' not in k]
for i, (name, weight) in enumerate(layer_weights):
    if i == 0:
        print(f"Input layer: {weight.shape[1]} -> {weight.shape[0]}")
    elif i == len(layer_weights) - 1:
        print(f"Output layer: {weight.shape[1]} -> {weight.shape[0]}")
    else:
        print(f"Hidden layer {i}: {weight.shape[1]} -> {weight.shape[0]}") 