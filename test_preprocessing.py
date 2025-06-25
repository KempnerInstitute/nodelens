"""Test script for the new preprocessing module."""

import torch
import torch.nn as nn
from alignment.preprocessing import (
    CNNPreprocessor, 
    LinearPreprocessor,
    get_preprocessor,
    preprocess_layer_activations
)

def test_cnn_preprocessing():
    """Test CNN preprocessing modes."""
    print("Testing CNN Preprocessing...")
    
    # Create a dummy conv layer and activation
    conv_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    activation = torch.randn(2, 3, 32, 32)  # [batch, channels, height, width]
    
    # Test unfold mode
    preprocessor = CNNPreprocessor(mode="unfold")
    unfolded = preprocessor.preprocess(activation, conv_layer, is_input=True)
    print(f"  Unfold mode: {activation.shape} -> {unfolded.shape}")
    
    # Test patchwise mode
    preprocessor = CNNPreprocessor(mode="patchwise")
    patchwise = preprocessor.preprocess(activation, conv_layer, is_input=True)
    print(f"  Patchwise mode: {activation.shape} -> {patchwise.shape}")
    
    # Test batch_patch_combined mode
    preprocessor = CNNPreprocessor(mode="batch_patch_combined")
    combined = preprocessor.preprocess(activation, conv_layer, is_input=True)
    print(f"  Batch-patch combined mode: {activation.shape} -> {combined.shape}")
    

def test_linear_preprocessing():
    """Test linear layer preprocessing."""
    print("\nTesting Linear Preprocessing...")
    
    # Create dummy linear layer and activation
    linear_layer = nn.Linear(784, 256)
    activation = torch.randn(10, 784)  # [batch, features]
    
    preprocessor = LinearPreprocessor()
    processed = preprocessor.preprocess(activation, linear_layer)
    print(f"  Linear: {activation.shape} -> {processed.shape}")
    
    # Test with higher dimensional input
    activation_3d = torch.randn(10, 28, 28)
    processed_3d = preprocessor.preprocess(activation_3d, linear_layer)
    print(f"  Linear (3D input): {activation_3d.shape} -> {processed_3d.shape}")


def test_get_preprocessor():
    """Test automatic preprocessor selection."""
    print("\nTesting Automatic Preprocessor Selection...")
    
    # Test with different layer types
    conv_layer = nn.Conv2d(3, 64, 3)
    linear_layer = nn.Linear(100, 50)
    
    conv_preprocessor = get_preprocessor(conv_layer, mode="unfold")
    linear_preprocessor = get_preprocessor(linear_layer)
    
    print(f"  Conv2d -> {type(conv_preprocessor).__name__}")
    print(f"  Linear -> {type(linear_preprocessor).__name__}")


def test_batch_preprocessing():
    """Test preprocessing multiple layers at once."""
    print("\nTesting Batch Preprocessing...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 32 * 32, 10)
    )
    
    # Create dummy activations
    activations = {
        "0_input": torch.randn(2, 3, 32, 32),
        "2_input": torch.randn(2, 64, 32, 32),
        "5_input": torch.randn(2, 128 * 32 * 32)
    }
    
    # Create layer modules dict
    layer_modules = {"0": model[0], "2": model[2], "5": model[5]}
    
    # Preprocess all activations
    preprocessed = preprocess_layer_activations(
        activations, 
        layer_modules,
        mode="unfold"
    )
    
    for name, original in activations.items():
        processed = preprocessed[name]
        print(f"  {name}: {original.shape} -> {processed.shape}")


if __name__ == "__main__":
    test_cnn_preprocessing()
    test_linear_preprocessing()
    test_get_preprocessor()
    test_batch_preprocessing()
    print("\nAll tests completed!") 