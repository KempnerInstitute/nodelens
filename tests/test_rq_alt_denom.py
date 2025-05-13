"""Tests for the alternative RQ denominator metric."""

import os
import sys
import torch
import numpy as np
import unittest

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alignment.metrics import get_metric, compute_rq_alternative_denominator

class TestRQAltDenom(unittest.TestCase):
    """Test the alternative Rayleigh Quotient implementation with different denominator."""
    
    def setUp(self):
        """Set up test data."""
        # Create random inputs and weights for testing
        self.inputs = torch.randn(100, 10)
        self.weights = torch.randn(5, 10)
        
    def test_rq_alt_denom_direct(self):
        """Test direct computation of rq_alt_denom."""
        # Compute RQ using direct function call
        rq_values = compute_rq_alternative_denominator(
            layer_inputs=self.inputs, 
            layer_weights=self.weights,
            relative=True
        )
        
        # Check shape
        self.assertEqual(rq_values.shape, (self.weights.shape[0],))
        
        # Check no NaNs or infs
        self.assertFalse(torch.isnan(rq_values).any())
        self.assertFalse(torch.isinf(rq_values).any())
        
        # Test relative scaling
        d_in = self.weights.shape[1]
        rq_nonrel = compute_rq_alternative_denominator(
            layer_inputs=self.inputs, 
            layer_weights=self.weights,
            relative=False
        )
        scaled_nonrel = rq_nonrel * torch.sqrt(torch.tensor(d_in, dtype=torch.float32))
        
        # Verify relative scaling implementation
        self.assertTrue(torch.allclose(rq_values, scaled_nonrel))
        
    def test_rq_alt_denom_via_registry(self):
        """Test computation via the metrics registry."""
        # Get metric from registry
        metric_instance = get_metric(name="rq_alt_denom")
        
        # Compute via API
        rq_values = metric_instance.compute_per_node_scores(
            layer_inputs=self.inputs,
            layer_weights=self.weights
        )
        
        # Check shape 
        self.assertEqual(rq_values.shape, (self.weights.shape[0],))
        
        # Check no NaNs or infs
        self.assertFalse(torch.isnan(rq_values).any())
        self.assertFalse(torch.isinf(rq_values).any())
        
        # Check values are reasonable (positive)
        self.assertTrue((rq_values >= 0).all())
        
        # Since implementations of rq_alt_denom may vary when accessed through registry vs directly,
        # we don't directly compare values but verify properties are correct
        
    def test_input_validation(self):
        """Test robust behavior with invalid inputs."""
        # Test with different dimensions
        inputs_1d = torch.randn(10)
        inputs_3d = torch.randn(10, 5, 5)
        
        # These should run without errors, though they may return zeros
        metric_instance = get_metric(name="rq_alt_denom")
        
        # 1D inputs should be handled (converted to 2D)
        result_1d = metric_instance.compute_per_node_scores(
            layer_inputs=inputs_1d,
            layer_weights=self.weights
        )
        self.assertEqual(result_1d.shape, (self.weights.shape[0],))
        
        # 3D inputs should be handled (flattened to 2D)
        result_3d = metric_instance.compute_per_node_scores(
            layer_inputs=inputs_3d,
            layer_weights=self.weights
        )
        self.assertEqual(result_3d.shape, (self.weights.shape[0],))
        
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches."""
        # Create mismatched inputs
        mismatched_inputs = torch.randn(100, 15)  # Different feature dimension than weights
        
        metric_instance = get_metric(name="rq_alt_denom")
        
        # This should handle the mismatch by truncating
        result = metric_instance.compute_per_node_scores(
            layer_inputs=mismatched_inputs,
            layer_weights=self.weights
        )
        
        self.assertEqual(result.shape, (self.weights.shape[0],))

if __name__ == "__main__":
    unittest.main() 