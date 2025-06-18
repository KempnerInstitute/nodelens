"""
Unit tests for Rayleigh quotient metrics.
"""

import pytest
import torch
import numpy as np
from alignment.metrics.rayleigh import (
    RayleighQuotient,
    RayleighQuotientAlternative,
)


class TestRayleighQuotient:
    """Test standard Rayleigh quotient metric."""
    
    def test_basic_computation(self):
        """Test basic RQ computation."""
        metric = RayleighQuotient()
        
        batch_size = 100
        num_features = 10
        num_neurons = 5
        
        # Create inputs with known covariance structure
        inputs = torch.randn(batch_size, num_features)
        weights = torch.randn(num_neurons, num_features)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (num_neurons,)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
    
    def test_identity_covariance(self):
        """Test with identity covariance matrix."""
        metric = RayleighQuotient(relative=False)
        
        # Create uncorrelated inputs (identity covariance)
        batch_size = 1000
        inputs = torch.randn(batch_size, 5)
        
        # Weights aligned with first dimension
        weights = torch.zeros(3, 5)
        weights[0, 0] = 1.0  # First weight vector aligned with first dimension
        weights[1, 1] = 1.0  # Second with second dimension
        weights[2] = torch.ones(5) / np.sqrt(5)  # Equal across all dimensions
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        # With identity covariance, RQ should approximately equal weight norm squared
        assert abs(scores[0] - 1.0) < 0.2  # Approximately 1
        assert abs(scores[1] - 1.0) < 0.2  # Approximately 1
        assert abs(scores[2] - 1.0) < 0.2  # Approximately 1
    
    def test_relative_scaling(self):
        """Test relative scaling option."""
        batch_size = 100
        inputs = torch.randn(batch_size, 8)
        weights = torch.randn(4, 8)
        
        # Test with relative=True
        metric_rel = RayleighQuotient(relative=True)
        scores_rel = metric_rel.compute(inputs=inputs, weights=weights)
        
        # Test with relative=False
        metric_abs = RayleighQuotient(relative=False)
        scores_abs = metric_abs.compute(inputs=inputs, weights=weights)
        
        # Relative scores should be normalized
        assert scores_rel.max() <= 1.0 or abs(scores_rel.max() - 1.0) < 0.1
        # Absolute scores can be larger
        assert (scores_abs >= scores_rel).all() or torch.allclose(scores_abs, scores_rel)
    
    def test_zero_weights(self):
        """Test with zero weight vectors."""
        metric = RayleighQuotient()
        
        inputs = torch.randn(50, 6)
        weights = torch.zeros(3, 6)
        weights[1] = torch.randn(6)  # Only middle weight is non-zero
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores[0] == 0  # Zero weight should give zero score
        assert scores[2] == 0  # Zero weight should give zero score
        assert scores[1] >= 0  # Non-zero weight should give non-negative score
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatch."""
        metric = RayleighQuotient()
        
        inputs = torch.randn(50, 10)
        weights = torch.randn(3, 8)  # Mismatch: 10 vs 8
        
        # Should handle gracefully by truncating
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()


class TestRayleighQuotientAlternative:
    """Test alternative Rayleigh quotient metric."""
    
    def test_basic_computation(self):
        """Test basic alternative RQ computation."""
        metric = RayleighQuotientAlternative()
        
        batch_size = 100
        inputs = torch.randn(batch_size, 7)
        weights = torch.randn(4, 7)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (4,)
        assert not torch.isnan(scores).any()
        assert (scores >= 0).all()  # Should be non-negative
    
    def test_alternative_denominator(self):
        """Test that alternative denominator uses trace(C)."""
        metric = RayleighQuotientAlternative(relative=True)
        
        # Create inputs with known covariance
        batch_size = 200
        dim = 3
        inputs = torch.randn(batch_size, dim)
        
        # Scale inputs to have specific variance
        inputs = inputs * torch.tensor([1.0, 2.0, 3.0])
        
        # Weight aligned with highest variance dimension
        weights = torch.zeros(1, dim)
        weights[0, 2] = 1.0  # Aligned with dimension of variance 9
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        # Score should be approximately var(dim2) / trace(C) / num_features
        # trace(C) ≈ 1 + 4 + 9 = 14
        expected = 9.0 / 14.0 / dim
        assert abs(scores[0] - expected) < 0.1
    
    def test_no_relative_scaling(self):
        """Test without relative scaling."""
        metric = RayleighQuotientAlternative(relative=False)
        
        inputs = torch.randn(80, 5)
        weights = torch.eye(5)  # Identity matrix
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (5,)
        assert (scores > 0).all()  # Should be positive
    
    def test_epsilon_parameter(self):
        """Test epsilon parameter for numerical stability."""
        metric = RayleighQuotientAlternative(epsilon=1e-6)
        
        # Create inputs with very small covariance
        inputs = torch.randn(50, 4) * 1e-8
        weights = torch.randn(2, 4)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


class TestComparisonBetweenRQVersions:
    """Test comparison between standard and alternative RQ."""
    
    def test_similar_rankings(self):
        """Test that both versions produce similar rankings."""
        batch_size = 150
        inputs = torch.randn(batch_size, 6)
        weights = torch.randn(10, 6)
        
        metric_std = RayleighQuotient()
        metric_alt = RayleighQuotientAlternative()
        
        scores_std = metric_std.compute(inputs=inputs, weights=weights)
        scores_alt = metric_alt.compute(inputs=inputs, weights=weights)
        
        # Rankings should be somewhat correlated
        rank_std = torch.argsort(scores_std)
        rank_alt = torch.argsort(scores_alt)
        
        # At least half should have same relative order
        same_order = 0
        for i in range(len(rank_std)):
            for j in range(i+1, len(rank_std)):
                if (rank_std[i] < rank_std[j]) == (rank_alt[i] < rank_alt[j]):
                    same_order += 1
        
        total_pairs = len(rank_std) * (len(rank_std) - 1) / 2
        assert same_order / total_pairs > 0.5  # At least 50% agreement


class TestEdgeCases:
    """Test edge cases for Rayleigh quotient metrics."""
    
    def test_insufficient_samples(self):
        """Test with too few samples for covariance."""
        metric = RayleighQuotient()
        
        inputs = torch.randn(1, 5)  # Only 1 sample
        weights = torch.randn(3, 5)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (3,)
        assert (scores == 0).all()  # Should return zeros
    
    def test_single_feature(self):
        """Test with single feature."""
        metric = RayleighQuotient()
        
        inputs = torch.randn(50, 1)
        weights = torch.randn(2, 1)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (2,)
        assert not torch.isnan(scores).any()
    
    def test_high_dimensional(self):
        """Test with high-dimensional inputs."""
        metric = RayleighQuotient()
        
        inputs = torch.randn(30, 100)  # More features than samples
        weights = torch.randn(5, 100)
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert scores.shape == (5,)
        assert not torch.isnan(scores).any()
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        metric = RayleighQuotient()
        
        # Test with very large values
        inputs = torch.randn(50, 5) * 1e6
        weights = torch.randn(3, 5) * 1e6
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        
        # Test with very small values
        inputs = torch.randn(50, 5) * 1e-6
        weights = torch.randn(3, 5) * 1e-6
        
        scores = metric.compute(inputs=inputs, weights=weights)
        
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 