"""
Test script to verify the pruning fix works correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Import pruning modules
from src.alignment.pruning.base import BasePruningStrategy, PruningConfig
from src.alignment.pruning.strategies.magnitude import MagnitudePruning


class SimpleNet(nn.Module):
    """Simple test network."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_fixed_pruning():
    """Test that the fixed pruning implementation works correctly."""
    print("=== Testing Fixed Pruning Implementation ===\n")
    
    # Create a simple layer
    layer = nn.Linear(5, 3)
    layer.weight.data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ])
    original_weights = layer.weight.data.clone()
    
    print("1. Original weights:")
    print(layer.weight.data)
    
    # Apply pruning
    config = PruningConfig(amount=0.4)
    strategy = MagnitudePruning(config)
    mask = strategy.prune(layer)
    
    print("\n2. After pruning:")
    print(f"Mask:\n{mask}")
    print(f"Weights:\n{layer.weight.data}")
    print(f"_original_weight stored:\n{layer._original_weight}")
    print(f"Original weights preserved correctly: {torch.allclose(layer._original_weight, original_weights)}")
    
    # Test forward pass
    x = torch.ones(2, 5)
    print("\n3. First forward pass:")
    output1 = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    print(f"Weights correctly pruned: {torch.allclose(layer.weight.data, original_weights * mask)}")
    
    # Simulate gradient update
    print("\n4. Simulating gradient update:")
    layer.weight.grad = torch.randn_like(layer.weight) * 0.1
    with torch.no_grad():
        layer.weight.data -= 0.1 * layer.weight.grad
    print(f"Weights after update:\n{layer.weight.data}")
    
    # Second forward pass - should reset to original * mask
    print("\n5. Second forward pass:")
    output2 = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    print(f"Weights correctly reset to original * mask: {torch.allclose(layer.weight.data, original_weights * mask)}")
    
    return True


def test_training_with_pruning():
    """Test that training works correctly with pruned network."""
    print("\n\n=== Testing Training with Pruning ===\n")
    
    # Create network
    model = SimpleNet()
    
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    # Apply pruning to model
    config = PruningConfig(amount=0.3)  # 30% pruning
    strategy = MagnitudePruning(config)
    
    pruning_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_weights = module.weight.data.clone()
            mask = strategy.prune(module)
            sparsity = (mask == 0).float().mean().item()
            pruning_info[name] = {
                'original_weights': original_weights,
                'mask': mask,
                'sparsity': sparsity
            }
            print(f"Pruned {name}: {sparsity*100:.1f}% sparse")
    
    # Train for a few steps
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining for 5 steps...")
    model.train()
    for step in range(5):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that pruning is maintained
        all_pruning_maintained = True
        for name, module in model.named_modules():
            if name in pruning_info:
                original = pruning_info[name]['original_weights']
                mask = pruning_info[name]['mask']
                expected = original * mask
                actual = module.weight.data
                
                # Check if weights match expected (within tolerance for numerical errors)
                if not torch.allclose(actual, expected, atol=1e-6):
                    # Weights don't match - check if they're at least pruned correctly
                    zeros_in_mask = (mask == 0)
                    zeros_in_weights = (actual == 0)
                    pruning_maintained = torch.all(zeros_in_mask <= zeros_in_weights)
                    
                    if not pruning_maintained:
                        all_pruning_maintained = False
                        print(f"\nStep {step+1}: Pruning NOT maintained in {name}!")
                        print(f"  Expected zeros: {zeros_in_mask.sum().item()}")
                        print(f"  Actual zeros: {zeros_in_weights.sum().item()}")
        
        if all_pruning_maintained:
            print(f"Step {step+1}: Loss = {loss.item():.4f}, Pruning maintained ✓")
        else:
            print(f"Step {step+1}: Loss = {loss.item():.4f}, Pruning BROKEN ✗")
    
    # Final check
    print("\nFinal pruning check:")
    for name, module in model.named_modules():
        if name in pruning_info:
            mask = pruning_info[name]['mask']
            zeros_in_mask = (mask == 0).sum().item()
            zeros_in_weights = (module.weight == 0).sum().item()
            print(f"{name}: Expected {zeros_in_mask} zeros, found {zeros_in_weights} zeros")


def test_high_sparsity():
    """Test network functionality with high sparsity."""
    print("\n\n=== Testing High Sparsity ===\n")
    
    model = SimpleNet()
    
    # Get initial output
    x = torch.randn(5, 10)
    with torch.no_grad():
        initial_output = model(x)
    
    print(f"Initial output norm: {initial_output.norm().item():.4f}")
    
    # Apply high sparsity pruning
    for sparsity_level in [0.5, 0.7, 0.9]:
        print(f"\nTesting {sparsity_level*100:.0f}% sparsity:")
        
        # Reset model
        model = SimpleNet()
        
        # Apply pruning
        config = PruningConfig(amount=sparsity_level)
        strategy = MagnitudePruning(config)
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                mask = strategy.prune(module)
                actual_sparsity = (mask == 0).float().mean().item()
                print(f"  {name}: {actual_sparsity*100:.1f}% sparse")
        
        # Test output
        with torch.no_grad():
            pruned_output = model(x)
        
        output_norm = pruned_output.norm().item()
        print(f"  Output norm: {output_norm:.4f}")
        print(f"  Network still produces output: {'✓' if output_norm > 0 else '✗'}")


def main():
    """Run all tests."""
    print("Testing Pruning Fix\n")
    
    # Run tests
    test_fixed_pruning()
    test_training_with_pruning()
    test_high_sparsity()
    
    print("\n" + "="*50)
    print("Testing complete!")


if __name__ == "__main__":
    main() 