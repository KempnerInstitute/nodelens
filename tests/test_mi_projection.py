"""Tests for the MI Projected vs Mean Input metric."""

import os
import sys
import torch
import numpy as np
import unittest

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alignment.metrics import get_metric, compute_mi_proj_vs_mean_input

class TestMIProjection(unittest.TestCase):
    """Test the MI Projected vs Mean Input metric implementation."""
    
    def setUp(self):
        """Set up test data."""
        # Create random inputs and weights for testing
        self.inputs = torch.randn(100, 10)
        self.weights = torch.randn(5, 10)
        
        # Create inputs with structure for more deterministic MI
        # Set up a case where a neuron's weights will strongly pick up on a specific pattern
        base_signal = torch.randn(100, 1)
        noise = torch.randn(100, 10) * 0.1
        self.structured_inputs = base_signal.repeat(1, 10) + noise
        
        # Create weights that strongly align with the structured pattern
        self.structured_weights = torch.ones(5, 10)
        self.structured_weights[0, :] = 1.0  # First neuron strongly picks up on the pattern
        self.structured_weights[1:, :] = torch.randn(4, 10) * 0.1  # Other neurons weakly respond
        
    def test_mi_projection_random(self):
        """Test MI projection with random inputs and weights."""
        # Compute MI using direct function call
        mi_values = compute_mi_proj_vs_mean_input(
            layer_inputs=self.inputs, 
            layer_weights=self.weights
        )
        
        # Check shape
        self.assertEqual(mi_values.shape, (self.weights.shape[0],))
        
        # Check no NaNs or infs
        self.assertFalse(torch.isnan(mi_values).any())
        self.assertFalse(torch.isinf(mi_values).any())
        
        # For random data, MI should be relatively low but non-zero
        self.assertTrue((mi_values > 0).all())
    
    def test_mi_projection_structured(self):
        """Test MI projection with structured inputs and weights."""
        # Compute MI using direct function call
        mi_values = compute_mi_proj_vs_mean_input(
            layer_inputs=self.structured_inputs, 
            layer_weights=self.structured_weights
        )
        
        # The first neuron should have higher MI than the rest
        self.assertTrue(mi_values[0] > mi_values[1:].mean())
        
    def test_mi_projection_via_registry(self):
        """Test computation via the metrics registry."""
        # Get metric from registry
        metric_instance = get_metric(name="mi_proj_vs_mean_input")
        
        # Compute via API
        mi_values = metric_instance.compute_per_node_scores(
            layer_inputs=self.inputs,
            layer_weights=self.weights
        )
        
        # Check that results match direct computation
        direct_values = compute_mi_proj_vs_mean_input(
            layer_inputs=self.inputs, 
            layer_weights=self.weights
        )
        
        self.assertTrue(torch.allclose(mi_values, direct_values))
        
    def test_bins_parameter(self):
        """Test the effect of the bins parameter."""
        # Compute with different bin sizes
        mi_low_bins = compute_mi_proj_vs_mean_input(
            layer_inputs=self.structured_inputs, 
            layer_weights=self.structured_weights,
            bins=5
        )
        
        mi_high_bins = compute_mi_proj_vs_mean_input(
            layer_inputs=self.structured_inputs, 
            layer_weights=self.structured_weights,
            bins=50
        )
        
        # Different bin counts should produce different results
        # (Results often don't converge until very high bin counts)
        self.assertFalse(torch.allclose(mi_low_bins, mi_high_bins, rtol=1e-2))
        
    def test_force_cpu_parameter(self):
        """Test the force_cpu_for_large_metric_ops parameter."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping force_cpu test")
            
        # Move inputs and weights to CUDA
        inputs_cuda = self.inputs.cuda()
        weights_cuda = self.weights.cuda()
        
        # Compute with and without forcing CPU
        mi_gpu = compute_mi_proj_vs_mean_input(
            layer_inputs=inputs_cuda, 
            layer_weights=weights_cuda,
            force_cpu_for_large_metric_ops=False
        )
        
        mi_cpu = compute_mi_proj_vs_mean_input(
            layer_inputs=inputs_cuda, 
            layer_weights=weights_cuda,
            force_cpu_for_large_metric_ops=True
        )
        
        # Results should be similar regardless of computation device
        self.assertTrue(torch.allclose(mi_gpu, mi_cpu, rtol=1e-4))
        
    def test_input_validation(self):
        """Test robust behavior with invalid inputs."""
        # Test with different dimensions
        inputs_1d = torch.randn(10)
        inputs_3d = torch.randn(10, 5, 5)
        weights_3d = torch.randn(5, 5, 5)
        
        # These should run without errors, though they may return zeros or empty tensors
        metric_instance = get_metric(name="mi_proj_vs_mean_input")
        
        # 1D inputs and 2D weights
        result_1d = metric_instance.compute_per_node_scores(
            layer_inputs=inputs_1d,
            layer_weights=self.weights
        )
        # Empty result is expected due to dimension mismatch
        self.assertEqual(result_1d.numel(), 0)
        
        # 3D inputs and 2D weights - should reshape/flatten inputs
        result_3d_inputs = metric_instance.compute_per_node_scores(
            layer_inputs=inputs_3d,
            layer_weights=self.weights
        )
        # Empty result is expected due to dimension mismatch (after flattening)
        self.assertEqual(result_3d_inputs.numel(), 0)
        
        # 2D inputs and 3D weights - should reshape/flatten weights
        result_3d_weights = metric_instance.compute_per_node_scores(
            layer_inputs=self.inputs,
            layer_weights=weights_3d
        )
        # Empty result is expected due to dimension mismatch (after flattening)
        self.assertEqual(result_3d_weights.numel(), 0)
        
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        # Create mismatched inputs
        mismatched_inputs = torch.randn(100, 8)  # Different feature dimension than weights
        
        # Should return empty tensor due to mismatch
        result = compute_mi_proj_vs_mean_input(
            layer_inputs=mismatched_inputs, 
            layer_weights=self.weights
        )
        
        self.assertEqual(result.numel(), 0)
        
    def test_edge_cases(self):
        """Test edge cases for MI projection."""
        # Single feature and single weight
        inputs_single = torch.randn(100, 1)
        weights_single = torch.randn(1, 1)
        
        # Should compute successfully with single feature
        result_single = compute_mi_proj_vs_mean_input(
            layer_inputs=inputs_single, 
            layer_weights=weights_single
        )
        
        self.assertEqual(result_single.shape, (weights_single.shape[0],))
        
        # Empty input batch
        empty_inputs = torch.zeros(0, 10)  # Empty batch
        
        # Should handle gracefully with empty batch
        result_empty = compute_mi_proj_vs_mean_input(
            layer_inputs=empty_inputs, 
            layer_weights=self.weights
        )
        
        # Should return empty tensor
        self.assertEqual(result_empty.numel(), 0)

if __name__ == "__main__":
    unittest.main() 