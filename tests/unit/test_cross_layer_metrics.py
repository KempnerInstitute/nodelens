"""
Unit tests for cross-layer activation mixing metrics.

Tests validate:
- compute_downstream_importance: shape, non-negativity, high-corr -> high MI
- compute_within_layer_redundancy: correlated pair -> high redundancy
"""

import pytest
import torch

from alignment.metrics.cross_layer import (
    compute_downstream_importance,
    compute_within_layer_redundancy,
)


# ---------------------------------------------------------------------------
# Tests: compute_downstream_importance
# ---------------------------------------------------------------------------

class TestDownstreamImportance:

    def test_output_shape(self):
        curr = torch.randn(50, 8)
        nxt = torch.randn(50, 12)
        di = compute_downstream_importance(curr, nxt)
        assert di.shape == (8,)

    def test_non_negative(self):
        curr = torch.randn(50, 8)
        nxt = torch.randn(50, 12)
        di = compute_downstream_importance(curr, nxt)
        assert torch.all(di >= -1e-6), f"Expected non-negative, min={di.min()}"

    def test_high_correlation_gives_high_mi(self):
        """A current neuron copied to next layer should have higher DI than random."""
        torch.manual_seed(42)
        curr = torch.randn(100, 4)
        # Next layer: first neuron = copy of curr[:,0] + noise
        nxt = torch.randn(100, 4)
        nxt[:, 0] = curr[:, 0] + 0.01 * torch.randn(100)

        di = compute_downstream_importance(curr, nxt)
        # Neuron 0 should have highest DI (it's copied forward)
        assert di[0] > di[1:].mean(), (
            f"Copied neuron DI={di[0]:.4f} should exceed mean of others={di[1:].mean():.4f}"
        )

    def test_uncorrelated_gives_low_mi(self):
        """Completely independent layers should have near-zero MI."""
        torch.manual_seed(42)
        curr = torch.randn(200, 4)
        nxt = torch.randn(200, 4)
        di = compute_downstream_importance(curr, nxt)
        # With independent data, MI should be very small (near 0)
        assert di.mean() < 0.1, f"Expected low DI for independent data, got {di.mean():.4f}"

    def test_max_refs_subsampling(self):
        """When next layer has more neurons than max_refs, should subsample."""
        torch.manual_seed(42)
        curr = torch.randn(50, 4)
        nxt = torch.randn(50, 100)
        di = compute_downstream_importance(curr, nxt, max_refs=10)
        assert di.shape == (4,)
        assert torch.all(torch.isfinite(di))


# ---------------------------------------------------------------------------
# Tests: compute_within_layer_redundancy
# ---------------------------------------------------------------------------

class TestWithinLayerRedundancy:

    def test_output_shape(self):
        acts = torch.randn(50, 8)
        red = compute_within_layer_redundancy(acts)
        assert red.shape == (8,)

    def test_non_negative(self):
        acts = torch.randn(50, 8)
        red = compute_within_layer_redundancy(acts)
        assert torch.all(red >= -1e-6)

    def test_correlated_pair_high_redundancy(self):
        """Two correlated neurons should have higher redundancy than independent ones."""
        torch.manual_seed(42)
        base = torch.randn(100, 1)
        acts = torch.randn(100, 4)
        # Make neurons 0 and 1 highly correlated
        acts[:, 0] = base.squeeze() + 0.01 * torch.randn(100)
        acts[:, 1] = base.squeeze() + 0.01 * torch.randn(100)

        red = compute_within_layer_redundancy(acts)
        # Neurons 0,1 should have higher redundancy than 2,3
        corr_red = (red[0] + red[1]) / 2
        indep_red = (red[2] + red[3]) / 2
        assert corr_red > indep_red, (
            f"Correlated pair redundancy={corr_red:.4f} should exceed "
            f"independent pair={indep_red:.4f}"
        )

    def test_max_refs_subsampling(self):
        torch.manual_seed(42)
        acts = torch.randn(50, 100)
        red = compute_within_layer_redundancy(acts, max_refs=10)
        assert red.shape == (100,)
        assert torch.all(torch.isfinite(red))

    def test_constant_neuron_low_redundancy(self):
        """A constant neuron (zero variance) should have low redundancy."""
        torch.manual_seed(42)
        acts = torch.randn(50, 4)
        acts[:, 0] = 5.0  # constant

        red = compute_within_layer_redundancy(acts)
        # Constant neuron correlation with others should be ~0 -> low MI
        assert red[0] < red[1:].mean() + 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
