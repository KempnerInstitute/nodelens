"""
Quick test script to verify the refactored alignment framework works.
"""

import torch
import torch.nn as nn

# Test imports
try:
    from alignment_refactor.core import (
        get_metric, 
        register_metric,
        BaseMetric
    )
    print("✅ Core imports successful")
except Exception as e:
    print(f"❌ Core import error: {e}")

try:
    from alignment_refactor.metrics.rayleigh import RayleighQuotient, DeltaAlignment
    from alignment_refactor.metrics.information import MutualInformationGaussian, AverageRedundancy
    from alignment_refactor.metrics.similarity import WeightCosineSimilarity
    print("✅ Metric imports successful")
except Exception as e:
    print(f"❌ Metric import error: {e}")

try:
    from alignment_refactor.models import ModelWrapper, AlignmentNetwork
    print("✅ Model imports successful")
except Exception as e:
    print(f"❌ Model import error: {e}")


def test_basic_metric():
    """Test basic metric computation."""
    print("\n=== Testing Basic Metric Computation ===")
    
    # Create test data
    batch_size, features = 64, 128
    inputs = torch.randn(batch_size, features)
    weights = torch.randn(32, features)
    
    # Test RQ metric
    rq = RayleighQuotient(relative=True)
    scores = rq.compute(inputs=inputs, weights=weights)
    print(f"RQ scores shape: {scores.shape}, mean: {scores.mean():.4f}")
    
    # Test using registry
    rq_registry = get_metric("rayleigh_quotient")
    scores2 = rq_registry.compute(inputs=inputs, weights=weights)
    print(f"Registry RQ mean: {scores2.mean():.4f}")
    
    return True


def test_model_wrapper():
    """Test model wrapper functionality."""
    print("\n=== Testing Model Wrapper ===")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 10)
    )
    
    # Wrap model
    wrapper = ModelWrapper(model)
    print(f"Auto-discovered layers: {wrapper.tracked_layers}")
    
    # Test forward with activations
    inputs = torch.randn(16, 64)
    outputs, activations = wrapper.forward_with_activations(inputs)
    print(f"Output shape: {outputs.shape}")
    print(f"Activations collected: {list(activations.keys())}")
    
    # Test weight extraction
    weights = wrapper.get_layer_weights()
    print(f"Weights extracted: {[(k, v.shape) for k, v in weights.items()]}")
    
    return True


def test_distributed_metric():
    """Test distributed metric computation."""
    print("\n=== Testing Distributed Computation ===")
    
    # Create metric
    metric = RayleighQuotient()
    
    # Simulate distributed data
    inputs = torch.randn(32, 64)
    weights = torch.randn(16, 64)
    
    # Test distributed computation (simulated)
    scores = metric.compute_distributed(
        inputs=inputs,
        weights=weights,
        world_size=1,  # Single process for testing
        rank=0
    )
    print(f"Distributed scores shape: {scores.shape}, mean: {scores.mean():.4f}")
    
    return True


def test_memory_aware():
    """Test memory-aware computation."""
    print("\n=== Testing Memory-Aware Computation ===")
    
    # Create large tensors
    large_size = 1000
    inputs = torch.randn(100, large_size)
    weights = torch.randn(50, large_size)
    
    # Test with CPU offloading
    metric = RayleighQuotient(
        force_cpu_for_large_ops=True,
        cpu_threshold=1e5  # Lower threshold for testing
    )
    
    scores = metric.compute(inputs=inputs, weights=weights)
    print(f"Large tensor RQ computed: shape={scores.shape}, device={scores.device}")
    
    return True


def test_custom_metric():
    """Test custom metric registration."""
    print("\n=== Testing Custom Metric ===")
    
    @register_metric("test_metric")
    class TestMetric(BaseMetric):
        @property
        def requires_inputs(self) -> bool:
            return True
        
        @property
        def requires_weights(self) -> bool:
            return True
        
        @property
        def requires_outputs(self) -> bool:
            return False
        
        def compute(self, inputs=None, weights=None, outputs=None, **kwargs):
            # Simple test: return sum of weight norms
            return torch.norm(weights, dim=1)
    
    # Test custom metric
    metric = get_metric("test_metric")
    weights = torch.randn(10, 20)
    scores = metric.compute(weights=weights)
    print(f"Custom metric scores: {scores.shape}, mean: {scores.mean():.4f}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Testing Refactored Alignment Framework\n")
    
    tests = [
        ("Basic Metric", test_basic_metric),
        ("Model Wrapper", test_model_wrapper),
        ("Distributed Computation", test_distributed_metric),
        ("Memory-Aware", test_memory_aware),
        ("Custom Metric", test_custom_metric),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} test passed\n")
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}\n")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! The refactored framework is working correctly.")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 