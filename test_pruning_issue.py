"""
Test script to investigate pruning algorithm issues.

This script tests the pruning implementation to identify why networks
break even with small pruning percentages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Import pruning modules
from src.alignment.pruning.base import BasePruningStrategy, PruningConfig
from src.alignment.pruning.strategies.magnitude import MagnitudePruning
from src.alignment.pruning.strategies.random import RandomPruning


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_basic_pruning():
    """Test basic pruning functionality."""
    print("=== Testing Basic Pruning ===")
    
    # Create a simple layer
    layer = nn.Linear(10, 5)
    original_weights = layer.weight.data.clone()
    
    # Test magnitude pruning
    config = PruningConfig(amount=0.2)  # Prune 20%
    strategy = MagnitudePruning(config)
    
    # Apply pruning
    mask = strategy.prune(layer)
    
    # Check results
    print(f"Original weights shape: {original_weights.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Percentage of zeros in mask: {(mask == 0).float().mean().item() * 100:.2f}%")
    print(f"Percentage of zeros in weights: {(layer.weight == 0).float().mean().item() * 100:.2f}%")
    
    # Verify mask was applied correctly
    pruned_weights = layer.weight.data
    mask_applied_correctly = torch.allclose(pruned_weights, original_weights * mask)
    print(f"Mask applied correctly: {mask_applied_correctly}")
    
    # Check if pruning hook is working
    if hasattr(layer, 'weight_mask'):
        print("Pruning mask registered as buffer: ✓")
    else:
        print("WARNING: Pruning mask NOT registered as buffer!")
    
    if hasattr(layer, '_pruning_hook'):
        print("Pruning hook registered: ✓")
    else:
        print("WARNING: Pruning hook NOT registered!")
    
    # Test forward pass
    x = torch.randn(2, 10)
    try:
        output = layer(x)
        print(f"Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
    
    return layer, mask


def test_pruning_persistence():
    """Test if pruning persists through forward passes."""
    print("\n=== Testing Pruning Persistence ===")
    
    layer = nn.Linear(10, 5)
    original_weights = layer.weight.data.clone()
    
    # Apply pruning
    config = PruningConfig(amount=0.5)
    strategy = MagnitudePruning(config)
    mask = strategy.prune(layer)
    
    # Check initial state
    zeros_after_prune = (layer.weight == 0).sum().item()
    print(f"Zeros after pruning: {zeros_after_prune}")
    
    # Perform multiple forward passes
    x = torch.randn(2, 10)
    for i in range(3):
        output = layer(x)
        zeros_after_forward = (layer.weight == 0).sum().item()
        print(f"Zeros after forward pass {i+1}: {zeros_after_forward}")
        
        # Check if weights changed
        if hasattr(layer, '_original_weight'):
            weight_unchanged = torch.allclose(layer._original_weight, original_weights)
            print(f"  Original weights preserved: {weight_unchanged}")
    
    # Simulate gradient update
    print("\nSimulating gradient update...")
    layer.weight.grad = torch.randn_like(layer.weight)
    layer.weight.data -= 0.01 * layer.weight.grad
    
    # Check after update
    zeros_after_update = (layer.weight == 0).sum().item()
    print(f"Zeros after gradient update: {zeros_after_update}")
    
    # Forward pass after update
    output = layer(x)
    zeros_after_forward_update = (layer.weight == 0).sum().item()
    print(f"Zeros after forward pass (post-update): {zeros_after_forward_update}")


def test_network_functionality():
    """Test if pruned network still functions correctly."""
    print("\n=== Testing Network Functionality ===")
    
    # Create network
    model = SimpleNet()
    
    # Get initial predictions
    x = torch.randn(5, 784)
    with torch.no_grad():
        initial_output = model(x)
        initial_predictions = initial_output.argmax(dim=1)
    
    print(f"Initial output norm: {initial_output.norm().item():.4f}")
    print(f"Initial predictions: {initial_predictions.tolist()}")
    
    # Apply pruning to each layer
    config = PruningConfig(amount=0.1)  # Only 10% pruning
    strategy = MagnitudePruning(config)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mask = strategy.prune(module)
            sparsity = (mask == 0).float().mean().item()
            print(f"\nPruned {name}: {sparsity*100:.2f}% sparse")
            
            # Check weight statistics
            weight_stats = {
                'mean': module.weight.data.mean().item(),
                'std': module.weight.data.std().item(),
                'min': module.weight.data.min().item(),
                'max': module.weight.data.max().item(),
                'zeros': (module.weight == 0).sum().item(),
                'total': module.weight.numel()
            }
            print(f"  Weight stats: mean={weight_stats['mean']:.4f}, "
                  f"std={weight_stats['std']:.4f}, "
                  f"zeros={weight_stats['zeros']}/{weight_stats['total']}")
    
    # Test pruned network
    with torch.no_grad():
        pruned_output = model(x)
        pruned_predictions = pruned_output.argmax(dim=1)
    
    print(f"\nPruned output norm: {pruned_output.norm().item():.4f}")
    print(f"Pruned predictions: {pruned_predictions.tolist()}")
    
    # Check if outputs are reasonable
    output_change = (pruned_output - initial_output).norm() / initial_output.norm()
    print(f"\nRelative output change: {output_change.item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = pruned_output.sum()
    loss.backward()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.grad is not None:
                grad_norm = module.weight.grad.norm().item()
                grad_zeros = (module.weight.grad == 0).sum().item()
                print(f"{name}: grad_norm={grad_norm:.4f}, grad_zeros={grad_zeros}")
            else:
                print(f"{name}: No gradients!")


def test_mask_creation():
    """Test mask creation logic in detail."""
    print("\n=== Testing Mask Creation ===")
    
    # Create test tensor
    weights = torch.tensor([
        [0.1, -0.5, 0.3, -0.2],
        [0.4, -0.1, 0.6, -0.3],
        [0.2, -0.4, 0.5, -0.6]
    ])
    
    print("Test weights:")
    print(weights)
    print(f"Absolute values: {weights.abs()}")
    
    # Test unstructured pruning
    config = PruningConfig(amount=0.5, structured=False)
    strategy = MagnitudePruning(config)
    
    # Manually compute expected mask
    importance = weights.abs()
    flat_importance = importance.flatten()
    k = int(0.5 * flat_importance.numel())
    threshold = flat_importance.kthvalue(k).values
    expected_mask = (importance > threshold).float()
    
    print(f"\nFor 50% pruning:")
    print(f"k = {k} elements to prune")
    print(f"threshold = {threshold.item():.4f}")
    print(f"Expected mask:\n{expected_mask}")
    
    # Create actual mask
    actual_mask = strategy.create_pruning_mask(importance, amount=0.5)
    print(f"Actual mask:\n{actual_mask}")
    print(f"Masks match: {torch.allclose(expected_mask, actual_mask)}")


def analyze_pruning_hook():
    """Analyze the pruning hook mechanism in detail."""
    print("\n=== Analyzing Pruning Hook ===")
    
    layer = nn.Linear(5, 3)
    layer.weight.data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ])
    
    print("Original weights:")
    print(layer.weight.data)
    
    # Apply pruning
    config = PruningConfig(amount=0.4)
    strategy = MagnitudePruning(config)
    mask = strategy.prune(layer)
    
    print(f"\nMask:\n{mask}")
    print(f"Weights after pruning:\n{layer.weight.data}")
    
    # Test forward pass behavior
    x = torch.ones(2, 5)
    
    # First forward pass
    print("\nFirst forward pass:")
    output1 = layer(x)
    print(f"Output: {output1}")
    print(f"Weights during forward: {layer.weight.data}")
    
    # Modify weights (simulate optimization step)
    print("\nModifying weights...")
    layer.weight.data += 0.1
    print(f"Weights after modification: {layer.weight.data}")
    
    # Second forward pass
    print("\nSecond forward pass:")
    output2 = layer(x)
    print(f"Output: {output2}")
    print(f"Weights during forward: {layer.weight.data}")
    
    # Check if mask is being applied
    if hasattr(layer, '_original_weight'):
        print(f"\nOriginal weights stored: {layer._original_weight}")
        expected_weights = layer._original_weight * layer.weight_mask
        print(f"Expected weights (original * mask): {expected_weights}")
        print(f"Weights match expected: {torch.allclose(layer.weight.data, expected_weights)}")


def main():
    """Run all tests."""
    print("Investigating Pruning Algorithm Issues\n")
    
    # Run tests
    test_basic_pruning()
    test_pruning_persistence()
    test_mask_creation()
    analyze_pruning_hook()
    test_network_functionality()
    
    print("\n" + "="*50)
    print("Testing complete. Check output for issues.")


if __name__ == "__main__":
    main() 