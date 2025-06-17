#!/usr/bin/env python3
"""
Simple test script for core alignment functionality.
Tests model wrapping and metrics without full experiments.
"""

import torch
import torch.nn as nn
import sys

# Add alignment module to path
sys.path.insert(0, 'src')

from alignment.models import ModelWrapper
from alignment.metrics.rayleigh import RayleighQuotient
from alignment.metrics.information import MutualInformationGaussian


class TestModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def test_model_wrapper():
    """Test ModelWrapper functionality."""
    print("\n=== Testing ModelWrapper ===")
    
    # Create model
    model = TestModel()
    
    # Test auto-discovery
    print("\nTesting auto-discovery:")
    auto_wrapped = ModelWrapper(model)
    print(f"Auto-discovered layers: {auto_wrapped.tracked_layers}")
    
    # Test manual specification
    print("\nTesting manual layer specification:")
    manual_layers = ['conv1', 'conv2', 'fc1', 'fc2']
    wrapped_model = ModelWrapper(model, tracked_layers=manual_layers)
    print(f"Manually specified layers: {wrapped_model.tracked_layers}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    outputs, activations = wrapped_model.forward_with_activations(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Activations collected: {list(activations.keys())}")
    
    # Test weight extraction
    weights = wrapped_model.get_layer_weights()
    print(f"\nWeights extracted:")
    for name, weight in weights.items():
        print(f"  {name}: {weight.shape}")
    
    return wrapped_model, x


def test_rayleigh_quotient(wrapped_model, x):
    """Test Rayleigh Quotient computation."""
    print("\n=== Testing Rayleigh Quotient ===")
    
    # Get activations and weights
    outputs, activations = wrapped_model.forward_with_activations(x)
    weights = wrapped_model.get_layer_weights()
    
    # Create metric
    rq_metric = RayleighQuotient()
    
    # Test on each layer
    for layer_name in wrapped_model.tracked_layers:
        layer_inputs = activations.get(f"{layer_name}_input")
        layer_weights = weights.get(layer_name)
        
        if layer_inputs is not None and layer_weights is not None:
            print(f"\n{layer_name}:")
            print(f"  Input shape: {layer_inputs.shape}")
            print(f"  Weight shape: {layer_weights.shape}")
            
            try:
                rq_scores = rq_metric.compute(layer_inputs, layer_weights)
                print(f"  RQ scores shape: {rq_scores.shape}")
                print(f"  RQ mean: {rq_scores.mean().item():.6f}")
                print(f"  RQ std: {rq_scores.std().item():.6f}")
                print(f"  RQ min: {rq_scores.min().item():.6f}")
                print(f"  RQ max: {rq_scores.max().item():.6f}")
            except Exception as e:
                print(f"  Error: {e}")


def test_dropout_masks(wrapped_model):
    """Test structured dropout functionality."""
    print("\n=== Testing Structured Dropout ===")
    
    # Create dropout masks for each layer
    dropout_masks = {}
    for layer_name in wrapped_model.tracked_layers:
        layer_info = wrapped_model.get_layer_info(layer_name)
        if 'out_features' in layer_info:
            num_units = layer_info['out_features']
        elif 'out_channels' in layer_info:
            num_units = layer_info['out_channels']
        else:
            continue
        
        # Create a mask that drops 30% of units
        mask = torch.rand(num_units) > 0.3
        dropout_masks[layer_name] = mask.float()
        
        print(f"{layer_name}: dropping {(~mask).sum().item()}/{num_units} units")
    
    # Test forward pass with dropout
    x = torch.randn(2, 3, 32, 32)
    
    # Apply dropout
    wrapped_model.apply_structured_dropout(dropout_masks, mode="multiplicative", permanent=False)
    
    # Forward pass
    outputs_with_dropout = wrapped_model.forward(x)
    print(f"\nOutput with dropout: {outputs_with_dropout.shape}")
    
    # Restore weights
    wrapped_model.restore_weights()
    
    # Forward pass without dropout
    outputs_normal = wrapped_model.forward(x)
    print(f"Output after restore: {outputs_normal.shape}")
    
    # Check difference
    diff = (outputs_with_dropout - outputs_normal).abs().mean()
    print(f"Mean absolute difference: {diff.item():.6f}")


def test_layer_info(wrapped_model):
    """Test layer information extraction."""
    print("\n=== Testing Layer Info ===")
    
    for layer_name in wrapped_model.tracked_layers:
        info = wrapped_model.get_layer_info(layer_name)
        print(f"\n{layer_name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")


def test_preprocessing_modes(wrapped_model):
    """Test different preprocessing modes."""
    print("\n=== Testing Preprocessing Modes ===")
    
    x = torch.randn(2, 3, 32, 32)
    
    # Test different modes
    for mode in ["flatten", "unfold", "patchwise"]:
        print(f"\nMode: {mode}")
        wrapped_model.preprocessing_mode = mode
        
        outputs, activations = wrapped_model.forward_with_activations(x)
        
        # Show processed shapes
        for name, act in activations.items():
            if "input" in name:
                print(f"  {name}: {act.shape}")


def main():
    """Run all tests."""
    print("Testing Core Alignment Functionality")
    print("=" * 50)
    
    # Test 1: Model Wrapper
    wrapped_model, x = test_model_wrapper()
    
    # Test 2: Rayleigh Quotient
    test_rayleigh_quotient(wrapped_model, x)
    
    # Test 3: Structured Dropout
    test_dropout_masks(wrapped_model)
    
    # Test 4: Layer Info
    test_layer_info(wrapped_model)
    
    # Test 5: Preprocessing Modes
    test_preprocessing_modes(wrapped_model)
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 