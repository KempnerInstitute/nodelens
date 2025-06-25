#!/usr/bin/env python3
"""Simple script to visualize MNIST MLP pruning results."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('logs/mnist_mlp_standard_pruning_results.json', 'r') as f:
    results = json.load(f)

# Extract training data
train_results = results['train_results']
epochs = list(range(1, len(train_results['train_accs']) + 1))

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training curves
ax1.plot(epochs, train_results['train_accs'], 'b-', label='Train Accuracy', linewidth=2)
ax1.plot(epochs, train_results['val_accs'], 'r-', label='Val Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot losses
ax2.plot(epochs, train_results['train_losses'], 'b-', label='Train Loss', linewidth=2)
ax2.plot(epochs, train_results['val_losses'], 'r-', label='Val Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Progress')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('logs/training_curves.png', dpi=150)
print("Saved training curves to logs/training_curves.png")

# Print summary
print("\n=== MNIST MLP Pruning Experiment Summary ===")
print(f"Model: MLP with layers {results['config']['model_config']['mlp_config']['hidden_dims']}")
print(f"Dataset: MNIST")
print(f"\nTraining Results:")
print(f"  - Initial accuracy: {train_results['val_accs'][0]:.2f}%")
print(f"  - Final accuracy: {train_results['val_accs'][-1]:.2f}%")
print(f"\nPruning Results:")
if 'pruning_results' in results and results['pruning_results']:
    pruning = results['pruning_results']['strategies']['magnitude']
    print(f"  - Strategy: Magnitude-based pruning")
    print(f"  - Sparsity: {pruning['sparsities'][0]*100:.1f}%")
    print(f"  - Accuracy after pruning + fine-tuning: {pruning['accuracies'][0]:.2f}%")
    print(f"  - Performance retention: {pruning['accuracies'][0]/train_results['val_accs'][-1]*100:.1f}%")
else:
    print("  - No pruning results found")

plt.close() 