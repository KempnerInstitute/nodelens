"""
Test script to verify the refactored alignment module is working correctly.
"""

import torch
import torch.nn as nn
from alignment.models import ModelWrapper
from alignment.metrics import METRIC_REGISTRY


def test_metrics():
    """Test that all metrics can be instantiated and run."""
    print("Testing all metrics...")
    
    # Create dummy data
    batch_size = 32
    input_features = 64
    output_features = 32
    
    inputs = torch.randn(batch_size, input_features)
    weights = torch.randn(output_features, input_features)
    outputs = torch.randn(batch_size, output_features)
    
    # Test each metric
    for metric_name, metric_class in METRIC_REGISTRY.items():
        try:
            print(f"\nTesting {metric_name}...")
            
            # Instantiate metric
            if 'pid' in metric_name:
                metric = metric_class(bins=5)  # Use fewer bins for PID
            else:
                metric = metric_class()
            
            # Prepare arguments based on what the metric needs
            kwargs = {}
            if hasattr(metric_class, 'requires_inputs') and metric_class.requires_inputs:
                kwargs['inputs'] = inputs
            if hasattr(metric_class, 'requires_weights') and metric_class.requires_weights:
                kwargs['weights'] = weights
            if hasattr(metric_class, 'requires_outputs') and metric_class.requires_outputs:
                kwargs['outputs'] = outputs
            
            # Compute scores
            scores = metric.compute(**kwargs)
            
            # Verify output
            assert scores is not None
            assert scores.shape[0] > 0
            assert not torch.isnan(scores).any()
            
            print(f"✅ {metric_name}: OK (shape={scores.shape}, mean={scores.mean():.4f})")
            
        except Exception as e:
            print(f"❌ {metric_name}: FAILED - {str(e)}")


def test_model_wrapper():
    """Test ModelWrapper functionality."""
    print("\n\nTesting ModelWrapper...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # Wrap the model
    wrapped_model = ModelWrapper(model)
    
    # Test forward with activations
    batch_size = 16
    inputs = torch.randn(batch_size, 28*28)
    
    outputs, activations = wrapped_model.forward_with_activations(inputs)
    
    print(f"Tracked layers: {wrapped_model.tracked_layers}")
    print(f"Output shape: {outputs.shape}")
    print(f"Activations collected: {list(activations.keys())}")
    
    # Get weights
    weights = wrapped_model.get_layer_weights()
    print(f"Weights collected: {list(weights.keys())}")
    
    print("✅ ModelWrapper: OK")


def test_external_dependencies():
    """Test that external dependencies are properly loaded."""
    print("\n\nTesting external dependencies...")
    
    try:
        from alignment.external import BROJA_2PID
        print("✅ BROJA_2PID module loaded successfully")
    except ImportError as e:
        print(f"❌ BROJA_2PID module failed to load: {e}")


def test_checkpoint_utils():
    """Test checkpoint utilities."""
    print("\n\nTesting checkpoint utilities...")
    
    try:
        from alignment.utils.checkpoint import save_checkpoint, save_model_for_inference
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test saving
        save_checkpoint(model, optimizer, epoch=1, filepath='test_checkpoint.pt')
        print("✅ Checkpoint saving: OK")
        
        # Test inference saving
        save_model_for_inference(model, 'test_inference.pt')
        print("✅ Inference model saving: OK")
        
        # Clean up
        import os
        if os.path.exists('test_checkpoint.pt'):
            os.remove('test_checkpoint.pt')
        if os.path.exists('test_inference.pt'):
            os.remove('test_inference.pt')
            
    except Exception as e:
        print(f"❌ Checkpoint utilities: FAILED - {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("ALIGNMENT MODULE REFACTORING TEST")
    print("=" * 60)
    
    test_metrics()
    test_model_wrapper()
    test_external_dependencies()
    test_checkpoint_utils()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60) 