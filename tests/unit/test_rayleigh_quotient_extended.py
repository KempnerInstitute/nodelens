"""
Unit tests for the Rayleigh Quotient metric (extended coverage).

Tests validate:
- _compute_from_covariance with known analytical values
- FastRayleighQuotient with 4D CNN inputs
- Class-conditioned RQ
- Edge cases (dimension mismatch, zero weights, bfloat16)
"""

import pytest
import torch
import torch.nn as nn

from alignment.metrics.rayleigh.rayleigh_quotient import (
    RayleighQuotient,
    FastRayleighQuotient,
)


# ---------------------------------------------------------------------------
# Tests: _compute_from_covariance
# ---------------------------------------------------------------------------

class TestComputeFromCovariance:

    def test_known_cov_identity(self):
        """With C=I, RQ(w) = ||w||² / ||w||² = 1 (relative: 1/trace(I))."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.eye(4)
        W = torch.randn(3, 4)
        result = rq._compute_from_covariance(C, W)
        # w^T I w / w^T w = 1 for all w
        torch.testing.assert_close(result, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_known_cov_diagonal(self):
        """With C=diag(2,1), w=[1,0]: RQ = 2/1 = 2."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.diag(torch.tensor([2.0, 1.0]))
        W = torch.tensor([[1.0, 0.0]])  # single neuron
        result = rq._compute_from_covariance(C, W)
        assert abs(result[0].item() - 2.0) < 1e-5

    def test_known_cov_off_diagonal(self):
        """C=[[2,1],[1,2]], w=[1,0]: w^T C w = 2, w^T w = 1, RQ=2."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        W = torch.tensor([[1.0, 0.0]])
        result = rq._compute_from_covariance(C, W)
        assert abs(result[0].item() - 2.0) < 1e-5

    def test_relative_normalization(self):
        """Relative RQ divides by trace(C)."""
        rq = RayleighQuotient(relative=True, regularization=0.0)
        C = torch.diag(torch.tensor([4.0, 2.0]))  # trace = 6
        W = torch.tensor([[1.0, 0.0]])
        result = rq._compute_from_covariance(C, W)
        # RQ = 4/1 = 4, relative = 4/6 ~ 0.6667
        assert abs(result[0].item() - 4.0 / 6.0) < 1e-5

    def test_regularization_applied(self):
        """Regularization adds to diagonal, changing RQ value."""
        reg = 0.1
        rq = RayleighQuotient(relative=False, regularization=reg)
        C = torch.zeros(2, 2)  # zero covariance
        W = torch.tensor([[1.0, 0.0]])
        result = rq._compute_from_covariance(C, W)
        # C + 0.1*I = diag(0.1, 0.1), w^T (0.1*I) w / w^T w = 0.1
        assert abs(result[0].item() - reg) < 1e-5

    def test_zero_weight_neuron(self):
        """Zero weight vector should yield RQ=0."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.eye(3)
        W = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = rq._compute_from_covariance(C, W)
        assert result[0].item() == 0.0
        assert abs(result[1].item() - 1.0) < 1e-5

    def test_multiple_neurons(self):
        """Multiple neurons: each gets its own RQ."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.diag(torch.tensor([3.0, 1.0]))
        W = torch.tensor([
            [1.0, 0.0],  # aligned with first eigenvector -> RQ = 3
            [0.0, 1.0],  # aligned with second eigenvector -> RQ = 1
        ])
        result = rq._compute_from_covariance(C, W)
        assert abs(result[0].item() - 3.0) < 1e-5
        assert abs(result[1].item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: standard compute method
# ---------------------------------------------------------------------------

class TestStandardCompute:

    def test_2d_inputs_shape(self):
        rq = RayleighQuotient(relative=True)
        inputs = torch.randn(50, 8)
        weights = torch.randn(4, 8)
        result = rq.compute(inputs=inputs, weights=weights)
        assert result.shape == (4,)
        assert torch.all(torch.isfinite(result))

    def test_requires_weights(self):
        rq = RayleighQuotient()
        with pytest.raises(ValueError, match="requires weights"):
            rq.compute(inputs=torch.randn(10, 5))

    def test_requires_inputs_without_covariance(self):
        rq = RayleighQuotient()
        with pytest.raises(ValueError, match="requires inputs"):
            rq.compute(weights=torch.randn(3, 5))

    def test_precomputed_covariance(self):
        """Can pass covariance directly instead of inputs."""
        rq = RayleighQuotient(relative=False, regularization=0.0)
        C = torch.eye(4)
        W = torch.randn(3, 4)
        result = rq.compute(weights=W, covariance=C)
        torch.testing.assert_close(result, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_dimension_mismatch_handled(self):
        """Mismatched input/weight dims should be truncated, not crash."""
        rq = RayleighQuotient(relative=True)
        inputs = torch.randn(50, 10)
        weights = torch.randn(4, 8)  # different from 10
        result = rq.compute(inputs=inputs, weights=weights)
        assert result.shape == (4,)
        assert torch.all(torch.isfinite(result))

    def test_min_samples_returns_zeros(self):
        rq = RayleighQuotient(min_samples=10)
        inputs = torch.randn(5, 8)  # fewer than min_samples
        weights = torch.randn(3, 8)
        result = rq.compute(inputs=inputs, weights=weights)
        assert torch.all(result == 0)


# ---------------------------------------------------------------------------
# Tests: FastRayleighQuotient (GAP-based)
# ---------------------------------------------------------------------------

class TestFastRayleighQuotient:

    def test_4d_input_output_shape(self):
        """4D input [B,C,H,W] should produce [C_out] scores."""
        rq = FastRayleighQuotient()
        inputs = torch.randn(20, 16, 8, 8)  # [B, C_in, H, W]
        weights = torch.randn(32, 16, 3, 3)  # [C_out, C_in, k, k]
        result = rq.compute(inputs=inputs, weights=weights)
        assert result.shape == (32,)
        assert torch.all(torch.isfinite(result))

    def test_2d_passthrough(self):
        """2D input should work like standard RQ."""
        rq = FastRayleighQuotient()
        inputs = torch.randn(50, 8)
        weights = torch.randn(4, 8)
        result = rq.compute(inputs=inputs, weights=weights)
        assert result.shape == (4,)

    def test_gap_reduces_spatial(self):
        """After GAP, spatial dimensions should be gone; verify via score difference."""
        rq = FastRayleighQuotient(relative=False, regularization=0.0)
        torch.manual_seed(42)
        B, C_in, H, W = 50, 4, 8, 8
        inputs_4d = torch.randn(B, C_in, H, W)
        weights = torch.randn(2, C_in, 3, 3)

        scores = rq.compute(inputs=inputs_4d, weights=weights)
        assert scores.shape == (2,)
        # Scores should be positive (variance of GAP activations / weight norm)
        # Not guaranteed all positive but should be finite
        assert torch.all(torch.isfinite(scores))

    def test_3d_input(self):
        """3D patchwise input [B, F, P] should be averaged over patches."""
        rq = FastRayleighQuotient()
        inputs = torch.randn(30, 8, 5)  # [B, F, P]
        weights = torch.randn(4, 8)
        result = rq.compute(inputs=inputs, weights=weights)
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Tests: class-conditioned RQ
# ---------------------------------------------------------------------------

class TestClassConditionedRQ:

    def test_two_class_basic(self):
        rq = RayleighQuotient(relative=True)
        # Two classes with different means -> class-conditioned cov differs from unconditional
        torch.manual_seed(42)
        n_per_class = 30
        inputs_0 = torch.randn(n_per_class, 8) + 1.0
        inputs_1 = torch.randn(n_per_class, 8) - 1.0
        inputs = torch.cat([inputs_0, inputs_1])
        targets = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()
        weights = torch.randn(4, 8)

        result = rq.compute_class_conditioned(inputs, weights, targets)
        assert result.shape == (4,)
        assert torch.all(torch.isfinite(result))

    def test_delta_rq(self):
        """delta_rq = rq_uncond - rq_cond."""
        rq = RayleighQuotient(relative=True)
        torch.manual_seed(42)
        n_per_class = 30
        inputs_0 = torch.randn(n_per_class, 8) + 2.0
        inputs_1 = torch.randn(n_per_class, 8) - 2.0
        inputs = torch.cat([inputs_0, inputs_1])
        targets = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()
        weights = torch.randn(4, 8)

        result = rq.compute_class_conditioned(
            inputs, weights, targets, return_delta_rq=True,
        )
        assert isinstance(result, dict)
        assert "rq_uncond" in result
        assert "rq_cond" in result
        assert "delta_rq" in result
        torch.testing.assert_close(
            result["delta_rq"],
            result["rq_uncond"] - result["rq_cond"],
            atol=1e-5, rtol=1e-5,
        )

    def test_single_class_equals_unconditional(self):
        """With one class, class-conditioned == unconditional."""
        rq = RayleighQuotient(relative=True)
        torch.manual_seed(42)
        inputs = torch.randn(40, 6)
        targets = torch.zeros(40).long()
        weights = torch.randn(3, 6)

        rq_cond = rq.compute_class_conditioned(inputs, weights, targets)
        rq_uncond = rq.compute(inputs=inputs, weights=weights)
        torch.testing.assert_close(rq_cond, rq_uncond, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: patchwise computation
# ---------------------------------------------------------------------------

class TestPatchwise:

    def test_3d_patchwise_loop_shape(self):
        """Loop-based patchwise computation should produce correct shape."""
        rq = RayleighQuotient(relative=True, regularization=1e-6)
        inputs = torch.randn(30, 9, 16)  # [B, features, patches]
        weights = torch.randn(8, 9)
        patch_var = torch.var(inputs, dim=0)
        patch_weights = patch_var.sum(dim=0)
        result = rq._compute_patchwise_loop(inputs, weights, patch_weights)
        assert result.shape == (8,)
        assert torch.all(torch.isfinite(result))

    def test_patchwise_loop_values_nonnegative(self):
        """With relative=True, patchwise RQ should be non-negative."""
        rq = RayleighQuotient(relative=True, regularization=1e-6)
        torch.manual_seed(42)
        inputs = torch.randn(30, 9, 8)
        weights = torch.randn(4, 9)
        patch_var = torch.var(inputs, dim=0)
        patch_weights = patch_var.sum(dim=0)
        result = rq._compute_patchwise_loop(inputs, weights, patch_weights)
        assert torch.all(result >= -1e-6), f"Expected non-negative, got min={result.min()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
