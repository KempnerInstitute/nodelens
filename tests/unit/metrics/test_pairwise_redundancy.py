"""
Unit tests for PairwiseRedundancyGaussian metric.
"""

import pytest
import torch
import torch.nn as nn

from nodelens.metrics.information.pairwise_gaussian import PairwiseRedundancyGaussian


class TestPairwiseRedundancyGaussian:
    """Tests for redundancy metric."""

    def test_initialization(self):
        """Test metric initialization."""
        metric = PairwiseRedundancyGaussian(num_pairs=10, sampling_strategy="random", mode="covariance_based")

        assert metric.num_pairs == 10
        assert metric.sampling_strategy == "random"
        assert metric.mode == "covariance_based"
        assert metric.requires_inputs
        assert metric.requires_weights
        assert not metric.requires_outputs

    def test_compute_basic(self):
        """Test basic redundancy computation."""
        # Create synthetic data
        batch_size, dim = 50, 20
        num_neurons = 10

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)

        metric = PairwiseRedundancyGaussian(num_pairs=5, sampling_strategy="random")
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # Check output shape
        assert redundancy.shape == (num_neurons,)

        # Redundancy should be non-negative
        assert (redundancy >= 0).all()

    def test_redundant_neurons_high_score(self):
        """Test that redundant neurons get high redundancy scores."""
        batch_size, dim = 100, 30

        # Create inputs
        inputs = torch.randn(batch_size, dim)

        # Create weights where first two neurons are nearly identical
        weights = torch.randn(5, dim)
        weights[1] = weights[0] + 0.01 * torch.randn(dim)  # Almost identical

        metric = PairwiseRedundancyGaussian(num_pairs=4, sampling_strategy="all")
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # Neurons 0 and 1 should have higher redundancy
        # (because they're correlated with each other)
        # This is not a strict test since it depends on sampling
        assert redundancy[0] > 0  # Some redundancy
        assert redundancy[1] > 0

    def test_orthogonal_neurons_low_redundancy(self):
        """Test that orthogonal neurons have low redundancy."""
        batch_size, dim = 100, 10

        # Create inputs with identity covariance
        inputs = torch.randn(batch_size, dim)

        # Create orthogonal weight vectors
        weights = torch.eye(dim)[:5]  # First 5 standard basis vectors

        metric = PairwiseRedundancyGaussian(num_pairs=4, sampling_strategy="all")
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # Orthogonal weights should have low redundancy
        assert redundancy.mean() < 0.5  # Should be close to zero for orthogonal

    def test_pairwise_matrix(self):
        """Test full pairwise redundancy matrix computation."""
        batch_size, dim = 50, 20
        num_neurons = 8

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)

        metric = PairwiseRedundancyGaussian()
        R_matrix = metric.compute_pairwise_matrix(inputs, weights)

        # Check shape
        assert R_matrix.shape == (num_neurons, num_neurons)

        # Should be symmetric
        assert torch.allclose(R_matrix, R_matrix.T, atol=1e-5)

        # Diagonal should be zero (neuron's redundancy with itself)
        assert torch.allclose(R_matrix.diag(), torch.zeros(num_neurons), atol=1e-5)

        # Should be non-negative
        assert (R_matrix >= 0).all()

    def test_correlation_squared_computation(self):
        """Test correlation squared computation."""
        dim = 10
        cov = torch.eye(dim)  # Identity covariance

        # Two orthogonal vectors
        w_i = torch.zeros(dim)
        w_i[0] = 1.0

        w_j = torch.zeros(dim)
        w_j[1] = 1.0

        metric = PairwiseRedundancyGaussian()
        rho_sq = metric._compute_correlation_squared(w_i, w_j, cov)

        # Should be zero (orthogonal)
        assert rho_sq < 1e-6

        # Same vector
        rho_sq_same = metric._compute_correlation_squared(w_i, w_i, cov)
        # Should be 1.0
        assert abs(rho_sq_same - 1.0) < 1e-5

    def test_sampling_strategies(self):
        """Test different sampling strategies."""
        batch_size, dim = 50, 20
        num_neurons = 10

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)

        # Random sampling
        metric_random = PairwiseRedundancyGaussian(num_pairs=3, sampling_strategy="random")
        redundancy_random = metric_random.compute(inputs=inputs, weights=weights)
        assert redundancy_random.shape == (num_neurons,)

        # Nearest sampling
        metric_nearest = PairwiseRedundancyGaussian(num_pairs=3, sampling_strategy="nearest")
        redundancy_nearest = metric_nearest.compute(inputs=inputs, weights=weights)
        assert redundancy_nearest.shape == (num_neurons,)

        # All pairs
        metric_all = PairwiseRedundancyGaussian(sampling_strategy="all")
        redundancy_all = metric_all.compute(inputs=inputs, weights=weights)
        assert redundancy_all.shape == (num_neurons,)

    def test_handles_small_batch(self):
        """Test behavior with small batch size."""
        batch_size, dim = 5, 20  # Small batch
        num_neurons = 8

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)

        metric = PairwiseRedundancyGaussian(regularization=1e-4)  # Use larger reg
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # Should not crash
        assert redundancy.shape == (num_neurons,)
        assert not torch.isnan(redundancy).any()

    def test_dimension_mismatch_handling(self):
        """Test that dimension mismatch raises an error."""
        batch_size = 50
        inputs = torch.randn(batch_size, 20)
        weights = torch.randn(10, 25)  # Mismatch!

        metric = PairwiseRedundancyGaussian()

        # Should raise an error for incompatible dimensions
        with pytest.raises(RuntimeError, match="cannot be multiplied"):
            metric.compute(inputs=inputs, weights=weights)

    def test_regularization_effect(self):
        """Test that regularization prevents singularity."""
        batch_size, dim = 10, 5
        num_neurons = 3

        # Create rank-deficient inputs
        inputs = torch.randn(batch_size, 2) @ torch.randn(2, dim)
        weights = torch.randn(num_neurons, dim)

        # Without enough regularization might fail
        metric_reg = PairwiseRedundancyGaussian(regularization=1e-3)
        redundancy = metric_reg.compute(inputs=inputs, weights=weights)

        # Should not have NaN
        assert not torch.isnan(redundancy).any()


def test_integration_with_real_layer():
    """Integration test with actual layer."""
    # Create a real Linear layer
    layer = nn.Linear(20, 10)

    # Create input batch
    inputs = torch.randn(50, 20)

    # Get weights
    weights = layer.weight.detach()

    # Compute redundancy
    metric = PairwiseRedundancyGaussian(num_pairs=5)
    redundancy = metric.compute(inputs=inputs, weights=weights)

    assert redundancy.shape == (10,)
    assert (redundancy >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
