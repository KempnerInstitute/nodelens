"""
Test script to investigate the specific pruning hook issue.

The previous test revealed that the pruning hook is overwriting the 
original weights with pruned weights, breaking gradient updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import pruning modules
from src.alignment.pruning.base import BasePruningStrategy, PruningConfig
from src.alignment.pruning.strategies.magnitude import MagnitudePruning


def test_hook_issue():
    """Test the specific hook issue."""
    print("=== Testing Pruning Hook Issue ===\n")
    
    # Create layer with known weights
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
    print(f"Has _original_weight: {hasattr(layer, '_original_weight')}")
    if hasattr(layer, '_original_weight'):
        print(f"_original_weight:\n{layer._original_weight}")
    
    # Perform forward pass
    x = torch.ones(2, 5)
    print("\n3. First forward pass:")
    output1 = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    if hasattr(layer, '_original_weight'):
        print(f"_original_weight after forward:\n{layer._original_weight}")
    
    # The issue: _original_weight is being set to the already pruned weights!
    print("\n4. Checking the issue:")
    print(f"Original weights (before pruning):\n{original_weights}")
    print(f"_original_weight (stored by hook):\n{layer._original_weight}")
    print(f"Are they the same? {torch.allclose(original_weights, layer._original_weight)}")
    
    # Simulate gradient update
    print("\n5. Simulating gradient update:")
    layer.weight.data += 0.1
    print(f"Weights after update:\n{layer.weight.data}")
    
    # Second forward pass
    print("\n6. Second forward pass:")
    output2 = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    
    # The problem: weights are reset to _original_weight * mask
    # But _original_weight is already pruned, so we're applying the mask twice!
    print("\n7. Analysis:")
    print(f"Expected (original * mask):\n{original_weights * mask}")
    print(f"Actual (_original_weight * mask):\n{layer._original_weight * mask}")
    print(f"Problem: _original_weight is already pruned!")


def test_correct_implementation():
    """Test what the correct implementation should do."""
    print("\n\n=== Correct Implementation Test ===\n")
    
    # Create layer
    layer = nn.Linear(5, 3)
    layer.weight.data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [10.0, 20.0, 30.0, 40.0, 50.0]
    ])
    true_original_weights = layer.weight.data.clone()
    
    # Apply mask manually (correct way)
    mask = torch.tensor([
        [0., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1.]
    ])
    
    # Store original weights BEFORE applying mask
    layer.register_buffer('_true_original_weight', true_original_weights)
    layer.register_buffer('weight_mask', mask)
    
    # Apply mask
    layer.weight.data *= mask
    
    print("1. After manual pruning:")
    print(f"Weights:\n{layer.weight.data}")
    print(f"True original weights:\n{layer._true_original_weight}")
    
    # Define correct hook
    def correct_hook(mod, inputs):
        # Apply mask to TRUE original weights
        mod.weight.data = mod._true_original_weight * mod.weight_mask
        return inputs
    
    layer.register_forward_pre_hook(correct_hook)
    
    # Test forward pass
    x = torch.ones(2, 5)
    print("\n2. Forward pass:")
    output = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    
    # Simulate optimization (weights get updated)
    print("\n3. Simulating optimization:")
    layer.weight.data += 0.1
    print(f"Weights after update:\n{layer.weight.data}")
    
    # Another forward pass
    print("\n4. Second forward pass:")
    output = layer(x)
    print(f"Weights after forward:\n{layer.weight.data}")
    print(f"Correctly reset to original * mask: {torch.allclose(layer.weight.data, true_original_weights * mask)}")


def main():
    """Run tests."""
    test_hook_issue()
    test_correct_implementation()
    
    print("\n" + "="*50)
    print("DIAGNOSIS: The pruning hook stores _original_weight AFTER the mask")
    print("has already been applied, causing the mask to be applied twice!")
    print("This breaks gradient updates and network functionality.")


if __name__ == "__main__":
    main() 