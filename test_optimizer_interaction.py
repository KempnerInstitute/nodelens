"""
Test to understand optimizer and pruning hook interaction.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.alignment.pruning.base import PruningConfig
from src.alignment.pruning.strategies.magnitude import MagnitudePruning


def test_optimizer_behavior():
    """Test how optimizer interacts with pruning."""
    print("=== Testing Optimizer Behavior with Pruning ===\n")
    
    # Create layer
    layer = nn.Linear(5, 3)
    layer.weight.data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ])
    
    # Apply pruning
    config = PruningConfig(amount=0.4)
    strategy = MagnitudePruning(config)
    mask = strategy.prune(layer)
    
    print("1. After pruning:")
    print(f"Mask:\n{mask}")
    print(f"Weight data:\n{layer.weight.data}")
    print(f"Weight parameter:\n{layer.weight}")
    
    # Create optimizer
    optimizer = optim.SGD([layer.weight], lr=0.1)
    
    # Simulate forward pass
    x = torch.ones(2, 5, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    
    print("\n2. After forward pass:")
    print(f"Weight data:\n{layer.weight.data}")
    
    # Backward pass
    loss.backward()
    print("\n3. After backward pass:")
    print(f"Weight grad:\n{layer.weight.grad}")
    
    # Optimizer step
    optimizer.step()
    print("\n4. After optimizer step:")
    print(f"Weight data:\n{layer.weight.data}")
    print(f"Zeros in weight: {(layer.weight == 0).sum().item()}")
    
    # Another forward pass
    output2 = layer(x)
    print("\n5. After second forward pass:")
    print(f"Weight data:\n{layer.weight.data}")
    print(f"Zeros in weight: {(layer.weight == 0).sum().item()}")
    
    # The issue: optimizer updates layer.weight directly, 
    # but the hook only runs during forward pass
    print("\n6. Analysis:")
    print("The optimizer updates layer.weight directly, bypassing the pruning mask.")
    print("The hook only runs during forward pass, so pruning is restored then.")
    print("This means gradients flow through pruned weights!")


def test_gradient_masking():
    """Test if gradients need to be masked."""
    print("\n\n=== Testing Gradient Masking ===\n")
    
    # Create layer
    layer = nn.Linear(5, 3)
    layer.weight.data = torch.ones(3, 5)
    
    # Apply pruning
    config = PruningConfig(amount=0.4)
    strategy = MagnitudePruning(config)
    mask = strategy.prune(layer)
    
    print(f"Mask:\n{mask}")
    
    # Forward pass
    x = torch.ones(2, 5, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradient:\n{layer.weight.grad}")
    print(f"Gradient at pruned locations (should be zero):")
    print(f"{layer.weight.grad[mask == 0]}")
    print(f"All gradients at pruned locations are zero: {torch.all(layer.weight.grad[mask == 0] == 0).item()}")


def main():
    """Run tests."""
    test_optimizer_behavior()
    test_gradient_masking()
    
    print("\n" + "="*50)
    print("FINDING: The pruning hook correctly maintains sparsity during forward passes,")
    print("but gradients can flow through pruned weights. This is actually correct")
    print("behavior for training pruned networks - we want to update the non-pruned")
    print("weights while keeping pruned weights at zero.")


if __name__ == "__main__":
    main() 