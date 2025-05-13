"""Tests for the weight similarity metrics."""

import os
import sys
import torch
import numpy as np
import unittest

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alignment.metrics import (
    get_metric, 
    compute_weight_similarity,
    compute_weight_cosine_similarity,
    compute_weight_dot_similarity,
    compute_weight_euclidean_distance
)

class TestWeightSimilarityMetrics(unittest.TestCase):
    """Test the weight similarity metrics implementation."""
    
    def setUp(self):
        """Set up test data."""
        # Create random weights for testing
        self.random_weights = torch.randn(5, 10)
        
        # Create structured weights with known similarities
        # First two vectors are identical
        self.structured_weights = torch.randn(5, 10)
        self.structured_weights[1] = self.structured_weights[0]  # Make rows 0 and 1 identical
        
        # Make row 2 orthogonal to row 0 (approximately)
        # First normalize row 0
        norm_row0 = self.structured_weights[0] / torch.norm(self.structured_weights[0])
        
        # Create random vector
        v = torch.randn(10)
        
        # Make v orthogonal to row0 using Gram-Schmidt
        projection = torch.dot(v, norm_row0) * norm_row0
        orthogonal_v = v - projection
        
        # Normalize and assign to row 2
        self.structured_weights[2] = orthogonal_v / torch.norm(orthogonal_v)
        
        # Make row 3 opposite to row 0 (cosine similarity = -1)
        self.structured_weights[3] = -self.structured_weights[0]
        
        # Row 4 remains random
        
    def test_cosine_similarity_direct(self):
        """Test direct computation of cosine similarity."""
        # Compute similarity matrix
        cosine_matrix = compute_weight_cosine_similarity(self.structured_weights)
        
        # Check shape
        self.assertEqual(cosine_matrix.shape, (5, 5))
        
        # Diagonal should be all 1.0 (self-similarity)
        diagonal = torch.diag(cosine_matrix)
        self.assertTrue(torch.allclose(diagonal, torch.ones_like(diagonal)))
        
        # Identical vectors should have similarity 1.0
        self.assertAlmostEqual(cosine_matrix[0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(cosine_matrix[1, 0].item(), 1.0, places=5)
        
        # Orthogonal vectors should have similarity near 0
        self.assertAlmostEqual(cosine_matrix[0, 2].item(), 0.0, places=5)
        self.assertAlmostEqual(cosine_matrix[2, 0].item(), 0.0, places=5)
        
        # Opposite vectors should have similarity -1.0
        self.assertAlmostEqual(cosine_matrix[0, 3].item(), -1.0, places=5)
        self.assertAlmostEqual(cosine_matrix[3, 0].item(), -1.0, places=5)
        
    def test_dot_similarity_direct(self):
        """Test direct computation of dot product similarity."""
        # Compute similarity matrix
        dot_matrix = compute_weight_dot_similarity(self.structured_weights)
        
        # Check shape
        self.assertEqual(dot_matrix.shape, (5, 5))
        
        # For identical vectors, dot product = squared norm
        norm_sq_0 = torch.sum(self.structured_weights[0] ** 2).item()
        self.assertAlmostEqual(dot_matrix[0, 1].item(), norm_sq_0, places=5)
        
        # For orthogonal vectors, dot product should be near 0
        self.assertAlmostEqual(dot_matrix[0, 2].item(), 0.0, places=5)
        
        # For opposite vectors, dot product = -squared norm
        self.assertAlmostEqual(dot_matrix[0, 3].item(), -norm_sq_0, places=5)
        
    def test_euclidean_distance_direct(self):
        """Test direct computation of euclidean distance."""
        # Compute distance matrix
        distance_matrix = compute_weight_euclidean_distance(self.structured_weights)
        
        # Check shape
        self.assertEqual(distance_matrix.shape, (5, 5))
        
        # Diagonal should be all 0.0 (self-distance)
        diagonal = torch.diag(distance_matrix)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))
        
        # Identical vectors should have distance 0.0
        self.assertAlmostEqual(distance_matrix[0, 1].item(), 0.0, places=5)
        self.assertAlmostEqual(distance_matrix[1, 0].item(), 0.0, places=5)
        
        # Distance is symmetric
        for i in range(5):
            for j in range(5):
                self.assertAlmostEqual(distance_matrix[i, j].item(), distance_matrix[j, i].item(), places=5)
                
        # For opposite vectors, distance should be 2*norm (as they're unit vectors after normalization)
        norm_0 = torch.norm(self.structured_weights[0]).item()
        self.assertAlmostEqual(distance_matrix[0, 3].item(), 2*norm_0, places=5)
        
    def test_via_registry(self):
        """Test computation via the metrics registry."""
        # Get metrics from registry
        cosine_metric = get_metric(name="weight_cosine_similarity")
        dot_metric = get_metric(name="weight_dot_similarity")
        euclidean_metric = get_metric(name="weight_euclidean_distance")
        
        # Compute via API
        cosine_values = cosine_metric.compute_per_node_scores(layer_weights=self.random_weights)
        dot_values = dot_metric.compute_per_node_scores(layer_weights=self.random_weights)
        euclidean_values = euclidean_metric.compute_per_node_scores(layer_weights=self.random_weights)
        
        # Check that results match direct computation
        direct_cosine = compute_weight_cosine_similarity(self.random_weights)
        direct_dot = compute_weight_dot_similarity(self.random_weights)
        direct_euclidean = compute_weight_euclidean_distance(self.random_weights)
        
        self.assertTrue(torch.allclose(cosine_values, direct_cosine))
        self.assertTrue(torch.allclose(dot_values, direct_dot))
        self.assertTrue(torch.allclose(euclidean_values, direct_euclidean))
        
    def test_generic_compute_weight_similarity(self):
        """Test the generic compute_weight_similarity function with different metric types."""
        # Compute using the generic function with different metric_type
        cosine_matrix = compute_weight_similarity(self.random_weights, metric_type="cosine")
        dot_matrix = compute_weight_similarity(self.random_weights, metric_type="dot")
        euclidean_matrix = compute_weight_similarity(self.random_weights, metric_type="euclidean")
        
        # Compare with specific implementations
        direct_cosine = compute_weight_cosine_similarity(self.random_weights)
        direct_dot = compute_weight_dot_similarity(self.random_weights)
        direct_euclidean = compute_weight_euclidean_distance(self.random_weights)
        
        self.assertTrue(torch.allclose(cosine_matrix, direct_cosine))
        self.assertTrue(torch.allclose(dot_matrix, direct_dot))
        self.assertTrue(torch.allclose(euclidean_matrix, direct_euclidean))
        
    def test_invalid_metric_type(self):
        """Test handling of invalid metric type."""
        # Should raise ValueError for invalid metric_type
        with self.assertRaises(ValueError):
            compute_weight_similarity(self.random_weights, metric_type="invalid_type")
            
    def test_input_validation(self):
        """Test robust behavior with invalid inputs."""
        # Test with different dimensions
        weights_1d = torch.randn(10)
        weights_3d = torch.randn(5, 5, 5)
        
        # These should run without errors, though they may return zeros or empty tensors
        metric_instance = get_metric(name="weight_cosine_similarity")
        
        # 1D weights should be handled by returning empty tensor
        result_1d = metric_instance.compute_per_node_scores(layer_weights=weights_1d)
        self.assertEqual(result_1d.numel(), 0, "Should return empty tensor for 1D weights")
        
        # 3D weights should be handled by returning empty tensor
        result_3d = metric_instance.compute_per_node_scores(layer_weights=weights_3d)
        self.assertEqual(result_3d.numel(), 0, "Should return empty tensor for 3D weights")
    
    def test_edge_cases(self):
        """Test edge cases for weight similarity metrics."""
        # Single weight vector
        single_weight = torch.randn(1, 10)
        
        # Metrics should handle single vector gracefully
        cosine_single = compute_weight_cosine_similarity(single_weight)
        dot_single = compute_weight_dot_similarity(single_weight)
        euclidean_single = compute_weight_euclidean_distance(single_weight)
        
        # Result should be 1x1 matrices
        self.assertEqual(cosine_single.shape, (1, 1))
        self.assertEqual(dot_single.shape, (1, 1))
        self.assertEqual(euclidean_single.shape, (1, 1))
        
        # Cosine similarity with self should be 1.0
        self.assertAlmostEqual(cosine_single.item(), 1.0, places=5)
        
        # Distance with self should be 0.0
        self.assertAlmostEqual(euclidean_single.item(), 0.0, places=5)
        
        # Zero vectors
        zero_weights = torch.zeros(3, 10)
        
        # Metrics should handle zero vectors
        # For zero vectors, cosine is undefined (0/0), so expect NaN or 0
        cosine_zeros = compute_weight_cosine_similarity(zero_weights)
        # For zero vectors, dot product is 0
        dot_zeros = compute_weight_dot_similarity(zero_weights)
        # For zero vectors, distance is 0 between any two zero vectors
        euclidean_zeros = compute_weight_euclidean_distance(zero_weights)
        
        # All dot products should be zero
        self.assertTrue(torch.allclose(dot_zeros, torch.zeros_like(dot_zeros)))
        
        # All distances should be zero
        self.assertTrue(torch.allclose(euclidean_zeros, torch.zeros_like(euclidean_zeros)))

if __name__ == "__main__":
    unittest.main() 