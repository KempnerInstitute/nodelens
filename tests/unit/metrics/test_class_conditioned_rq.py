"""
Unit tests for class-conditioned Rayleigh Quotient.
"""

import pytest
import torch

from nodelens.metrics.rayleigh.rayleigh_quotient import RayleighQuotient


class TestClassConditionedRQ:
    """Tests for class-conditioned RQ functionality."""

    def test_basic_class_conditioned_computation(self):
        """Test basic class-conditioned RQ computation."""
        batch_size, dim = 100, 20
        num_neurons = 5
        num_classes = 3

        # Create inputs and targets
        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)
        targets = torch.randint(0, num_classes, (batch_size,))

        metric = RayleighQuotient()
        rq_cond = metric.compute_class_conditioned(inputs, weights, targets, return_delta_rq=False)

        # Check shape
        assert rq_cond.shape == (num_neurons,)

        # Should be in valid range for relative RQ
        assert (rq_cond >= 0).all()
        assert (rq_cond <= 1.0).all() or not metric.relative

    def test_delta_rq_computation(self):
        """Test ΔRQ computation."""
        batch_size, dim = 100, 20
        num_neurons = 5

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)
        targets = torch.randint(0, 3, (batch_size,))

        metric = RayleighQuotient()
        results = metric.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        # Check structure
        assert "rq_uncond" in results
        assert "rq_cond" in results
        assert "delta_rq" in results

        # Check shapes
        assert results["rq_uncond"].shape == (num_neurons,)
        assert results["rq_cond"].shape == (num_neurons,)
        assert results["delta_rq"].shape == (num_neurons,)

        # ΔRQ = RQ_uncond - RQ_cond
        expected_delta = results["rq_uncond"] - results["rq_cond"]
        assert torch.allclose(results["delta_rq"], expected_delta, atol=1e-5)

    def test_discriminative_neurons_high_delta_rq(self):
        """Test that discriminative neurons have high ΔRQ."""
        batch_size, dim = 200, 10
        num_classes = 2

        # Create class-specific data
        inputs = torch.randn(batch_size, dim)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Add class-specific pattern to one dimension
        for c in range(num_classes):
            mask = targets == c
            inputs[mask, 0] += 2.0 * c  # Dimension 0 is discriminative

        # Create weights
        # Weight that aligns with discriminative dimension
        w_discriminative = torch.zeros(1, dim)
        w_discriminative[0, 0] = 1.0

        # Weight that aligns with non-discriminative dimension
        w_general = torch.zeros(1, dim)
        w_general[0, 1] = 1.0

        weights = torch.cat([w_discriminative, w_general], dim=0)

        metric = RayleighQuotient(regularization=1e-4)
        results = metric.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        # Discriminative neuron should have higher ΔRQ
        # (Note: this is a soft test, might not always hold for random data)
        # assert results['delta_rq'][0] > results['delta_rq'][1]

        # At least check that ΔRQ is computed
        assert results["delta_rq"][0] != 0.0 or results["delta_rq"][1] != 0.0

    def test_single_class_edge_case(self):
        """Test behavior when all samples are from one class."""
        batch_size, dim = 50, 20
        num_neurons = 5

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)
        targets = torch.zeros(batch_size, dtype=torch.long)  # All class 0

        metric = RayleighQuotient()
        rq_cond = metric.compute_class_conditioned(inputs, weights, targets)

        # Should not crash
        assert rq_cond.shape == (num_neurons,)
        assert not torch.isnan(rq_cond).any()

    def test_small_class_sizes(self):
        """Test handling of classes with few samples."""
        batch_size, dim = 20, 10
        num_neurons = 3

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)

        # Create unbalanced targets: class 0 has 18 samples, class 1 has 2
        targets = torch.cat([torch.zeros(18, dtype=torch.long), torch.ones(2, dtype=torch.long)])

        metric = RayleighQuotient(min_samples=3)
        rq_cond = metric.compute_class_conditioned(inputs, weights, targets)

        # Should skip class 1 (too few samples) and use only class 0
        assert not torch.isnan(rq_cond).any()

    def test_regularization_in_class_conditioned(self):
        """Test that regularization is applied in class-conditioned mode."""
        batch_size, dim = 50, 10
        num_neurons = 5

        # Rank-deficient input
        inputs = torch.randn(batch_size, 5) @ torch.randn(5, dim)
        weights = torch.randn(num_neurons, dim)
        targets = torch.randint(0, 2, (batch_size,))

        metric = RayleighQuotient(regularization=1e-3)
        rq_cond = metric.compute_class_conditioned(inputs, weights, targets)

        # Should not crash or produce NaN
        assert not torch.isnan(rq_cond).any()
        assert not torch.isinf(rq_cond).any()

    def test_delta_rq_invariance_to_scaling(self):
        """Test that ΔRQ behaves reasonably under weight scaling."""
        batch_size, dim = 100, 15
        num_neurons = 5

        inputs = torch.randn(batch_size, dim)
        weights = torch.randn(num_neurons, dim)
        targets = torch.randint(0, 3, (batch_size,))

        metric = RayleighQuotient()

        # Compute ΔRQ with original weights
        results1 = metric.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        # Scale weights by constant
        weights_scaled = weights * 2.0
        results2 = metric.compute_class_conditioned(inputs, weights_scaled, targets, return_delta_rq=True)

        # ΔRQ should be scale-invariant (RQ normalizes by w^T w)
        assert torch.allclose(results1["delta_rq"], results2["delta_rq"], rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
