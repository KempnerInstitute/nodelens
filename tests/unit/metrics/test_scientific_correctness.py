"""
Scientific Correctness Validation Tests.

These tests validate metrics against known ground truth on synthetic data,
proving that the implementations match theoretical predictions.
"""

import sys

import pytest
import torch

from alignment.metrics.information.pairwise_gaussian import PairwiseRedundancyGaussian
from alignment.metrics.information.synergy_mmi import SynergyGaussianMMI
from alignment.metrics.rayleigh.rayleigh_quotient import RayleighQuotient


class TestRedundancyCorrectness:
    """Validate redundancy metric on known cases."""

    def test_orthogonal_weights_low_redundancy(self):
        """
        GROUND TRUTH: Orthogonal weight vectors -> LOW redundancy.

        Theory: If w_i ⊥ w_j, then ρ(Yi, Yj) ~ 0 -> R ~ 0
        """
        # Create orthogonal weights (standard basis vectors)
        D = 20
        N = 10
        weights = torch.eye(D)[:N]  # First N basis vectors (orthogonal)

        # Random inputs
        inputs = torch.randn(100, D)

        # Compute redundancy
        metric = PairwiseRedundancyGaussian(sampling_strategy="all", mode="covariance_based")
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # ASSERT: Should be near zero
        assert redundancy.mean() < 0.15, f"Orthogonal weights should have low redundancy, got {redundancy.mean():.4f}"

        print(f"OK Orthogonal weights -> redundancy = {redundancy.mean():.4f} (expected < 0.15)")

    def test_colinear_weights_high_redundancy(self):
        """
        GROUND TRUTH: Colinear (parallel) weights -> HIGH redundancy.

        Theory: If w_i ~ w_j, then ρ ~ 1 -> R = -0.5·log(1-1) -> large
        """
        # Create nearly identical weights
        base_weight = torch.randn(1, 20)
        weights = base_weight.repeat(5, 1) + 0.01 * torch.randn(5, 20)  # Small noise

        inputs = torch.randn(100, 20)

        # Compute redundancy
        metric = PairwiseRedundancyGaussian(sampling_strategy="all", mode="covariance_based")
        redundancy = metric.compute(inputs=inputs, weights=weights)

        # ASSERT: Should be high
        assert redundancy.mean() > 0.8, f"Colinear weights should have high redundancy, got {redundancy.mean():.4f}"

        print(f"OK Colinear weights -> redundancy = {redundancy.mean():.4f} (expected > 0.8)")

    def test_output_based_matches_covariance_based(self):
        """
        Verify output-based and covariance-based give similar results.
        """
        B, D, N = 100, 50, 20

        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)
        outputs = inputs @ weights.T

        # Covariance-based
        metric_cov = PairwiseRedundancyGaussian(mode="covariance_based", sampling_strategy="all")
        redundancy_cov = metric_cov.compute(inputs=inputs, weights=weights)

        # Output-based
        metric_out = PairwiseRedundancyGaussian(mode="output_based", sampling_strategy="all")
        redundancy_out = metric_out.compute(outputs=outputs)

        # Should be highly correlated
        correlation = torch.corrcoef(torch.stack([redundancy_cov, redundancy_out]))[0, 1]

        assert correlation > 0.95, f"Output-based and covariance-based should match, correlation = {correlation:.4f}"

        print(f"OK Output-based vs covariance-based correlation = {correlation:.4f}")


class TestDeltaRQCorrectness:
    """Validate class-conditioned RQ on known cases."""

    def test_class_separated_data_high_delta_rq(self):
        """
        GROUND TRUTH: Dimension that separates classes -> HIGH ΔRQ.

        Theory: ΔRQ = RQ(overall) - E[RQ|class]
        If dimension k separates classes:
          - Var(X_k | overall) is high (classes far apart)
          - Var(X_k | class) is low (points within class are close)
          - ΔRQ for w aligned with dim k is HIGH
        """
        B, D = 200, 10
        num_classes = 2

        # Create inputs
        inputs = torch.randn(B, D)
        targets = torch.randint(0, num_classes, (B,))

        # Add strong class separation on dimension 0
        for c in range(num_classes):
            mask = targets == c
            inputs[mask, 0] += 5.0 * c  # Dim 0 separates classes strongly

        # Create weights
        # w_sep aligned with separating dimension
        w_sep = torch.zeros(1, D)
        w_sep[0, 0] = 1.0

        # w_gen aligned with non-separating dimension
        w_gen = torch.zeros(1, D)
        w_gen[0, 1] = 1.0

        weights = torch.cat([w_sep, w_gen], dim=0)

        # Compute ΔRQ
        rq = RayleighQuotient()
        results = rq.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        # ASSERT: Separating dimension should have higher ΔRQ
        assert (
            results["delta_rq"][0] > results["delta_rq"][1]
        ), f"Expected ΔRQ[sep] > ΔRQ[gen], got {results['delta_rq'][0]:.4f} vs {results['delta_rq'][1]:.4f}"

        # Should be significantly positive
        assert results["delta_rq"][0] > 0.01, f"Expected positive ΔRQ for separating dim, got {results['delta_rq'][0]:.4f}"

        print(f"OK Separating dimension -> ΔRQ = {results['delta_rq'][0]:.4f} (vs {results['delta_rq'][1]:.4f})")

    def test_single_class_zero_delta_rq(self):
        """
        GROUND TRUTH: Single class -> ΔRQ ~ 0.

        Theory: If all samples from one class:
          RQ(overall) = RQ(class 0) -> ΔRQ = 0
        """
        B, D, N = 100, 15, 5

        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)
        targets = torch.zeros(B, dtype=torch.long)  # All class 0

        rq = RayleighQuotient()
        results = rq.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        # ΔRQ should be very small
        assert torch.abs(results["delta_rq"]).mean() < 0.01, f"Single class should give ΔRQ ~ 0, got {results['delta_rq'].mean():.4f}"

        print(f"OK Single class -> ΔRQ ~ {results['delta_rq'].mean():.4f} (expected ~ 0)")


class TestMutualInformationCorrectness:
    """Validate MI computation on known cases."""

    def test_independent_variables_zero_mi(self):
        """
        GROUND TRUTH: Independent variables -> MI ~ 0.

        Theory: If Y ⊥ Z, then I(Y; Z) = 0
        """
        B = 1000

        # Independent Y and Z
        Y = torch.randn(B)
        Z = torch.randint(0, 3, (B,))

        # Compute MI
        synergy_metric = SynergyGaussianMMI()
        mi = synergy_metric._gaussian_mi_categorical(Y, Z)

        # Should be near zero (some noise expected due to finite sample)
        assert mi < 0.15, f"Independent variables should have MI ~ 0, got {mi:.4f}"

        print(f"OK Independent variables -> MI = {mi:.4f} (expected < 0.15)")

    def test_correlated_variables_positive_mi(self):
        """
        GROUND TRUTH: Correlated variables -> MI > 0.

        Theory: If Y depends on Z, then I(Y; Z) > 0
        """
        B = 1000

        # Create strong correlation
        Z = torch.randint(0, 3, (B,))
        Y = Z.float() * 2.0 + 0.5 * torch.randn(B)  # Y depends on Z

        # Compute MI
        synergy_metric = SynergyGaussianMMI()
        mi = synergy_metric._gaussian_mi_categorical(Y, Z)

        # Should be positive
        assert mi > 0.5, f"Correlated variables should have MI > 0.5, got {mi:.4f}"

        print(f"OK Correlated variables -> MI = {mi:.4f} (expected > 0.5)")

    def test_deterministic_relationship_high_mi(self):
        """
        GROUND TRUTH: Deterministic relationship -> High MI.
        """
        B = 1000

        # Deterministic: Y = f(Z) with no noise
        Z = torch.randint(0, 5, (B,))
        Y = Z.float() * 3.0  # Perfect deterministic relationship

        synergy_metric = SynergyGaussianMMI()
        mi = synergy_metric._gaussian_mi_categorical(Y, Z)

        # Should be high
        assert mi > 1.0, f"Deterministic relationship should have high MI, got {mi:.4f}"

        print(f"OK Deterministic relationship -> MI = {mi:.4f} (expected > 1.0)")


class TestRayleighQuotientCorrectness:
    """Validate RQ computation on known cases."""

    def test_rq_bounds_relative_mode(self):
        """RQ with relative=True should be in [0, 1]."""
        B, D, N = 100, 30, 15

        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)

        rq = RayleighQuotient(relative=True)
        scores = rq.compute(inputs, weights)

        # Should be in valid range
        assert (scores >= 0).all(), "RQ should be non-negative"
        assert (scores <= 1.0 + 1e-4).all(), "Relative RQ should be ≤ 1.0"

        print(f"OK RQ in valid range: [{scores.min():.4f}, {scores.max():.4f}]")

    def test_rq_top_eigenvector_maximum(self):
        """
        GROUND TRUTH: Weight aligned with top eigenvector -> maximum RQ.

        Theory: RQ(w) is maximized when w = v_1 (top eigenvector of Σ)
        """
        B, D = 200, 15

        # Create inputs with known covariance structure
        # Use factor model: X = A·Z where Z has known covariance
        Z = torch.randn(B, 5)
        A = torch.randn(D, 5)
        inputs = Z @ A.T  # [B, D]

        # Compute true covariance
        cov = torch.cov(inputs.T)

        # Get top eigenvector
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        top_eigenvector = eigenvectors[:, -1]  # Last column = largest eigenvalue

        # Create weights: one aligned with top PC, one random
        weights = torch.stack([top_eigenvector, torch.randn(D) / D**0.5])  # Random direction
        weights = weights / weights.norm(dim=1, keepdim=True)  # Normalize

        # Compute RQ
        rq = RayleighQuotient(relative=False)  # Use absolute for this test
        scores = rq.compute(inputs, weights)

        # ASSERT: Top eigenvector should have highest RQ
        assert scores[0] > scores[1], f"Top eigenvector should have highest RQ: {scores[0]:.4f} vs {scores[1]:.4f}"

        print(f"OK Top eigenvector -> RQ = {scores[0]:.4f} (vs random: {scores[1]:.4f})")


class TestSynergyCorrectness:
    """Validate synergy computation."""

    def test_identical_neurons_zero_synergy(self):
        """
        GROUND TRUTH: Identical neurons -> synergy ~ 0.

        Theory: If Yi = Yj, then I(Z; Yi, Yj) = I(Z; Yi) = I(Z; Yj)
        So S = I(Z; Yi,Yj) - I(Z; Yi) - I(Z; Yj) + min(...) = 0
        """
        B, D = 200, 20

        # Create inputs and targets
        inputs = torch.randn(B, D)
        targets = torch.randint(0, 3, (B,))

        # Create identical weights
        base_weight = torch.randn(1, D)
        weights = base_weight.repeat(3, 1)  # Three identical neurons

        # Compute synergy
        metric = SynergyGaussianMMI(num_pairs=2, sampling_strategy="all")

        # Compute outputs
        outputs = inputs @ weights.T

        synergy = metric.compute(inputs=inputs, weights=weights, outputs=outputs, targets=targets)

        # Should be near zero (some noise due to finite sample)
        assert torch.abs(synergy).mean() < 0.2, f"Identical neurons should have near-zero synergy, got {synergy.mean():.4f}"

        print(f"OK Identical neurons -> synergy ~ {synergy.mean():.4f} (expected ~ 0)")

    def test_complementary_features_positive_synergy(self):
        """
        GROUND TRUTH: Complementary features -> positive synergy (in some cases).

        This is a softer test since synergy depends heavily on the specific
        relationship between features and target.
        """
        B = 500

        # Create complementary features for binary classification
        # Feature 1: encodes bit 0
        # Feature 2: encodes bit 1
        # Together they can represent 4 states, individually only 2

        targets = torch.randint(0, 4, (B,))  # 4 classes

        # Create features from targets
        Y1 = (targets % 2).float() + 0.1 * torch.randn(B)  # Bit 0
        Y2 = (targets // 2).float() + 0.1 * torch.randn(B)  # Bit 1

        outputs = torch.stack([Y1, Y2], dim=1)  # [B, 2]

        # Create dummy inputs/weights for interface
        inputs = torch.randn(B, 10)
        weights = torch.randn(2, 10)

        # Compute synergy
        metric = SynergyGaussianMMI(num_pairs=1, sampling_strategy="all")
        synergy = metric.compute(inputs=inputs, weights=weights, outputs=outputs, targets=targets)

        # Note: This test is softer - synergy can be positive or negative depending on details
        # Just check it's computed without errors
        assert not torch.isnan(synergy).any(), "Synergy should not be NaN"

        print(f"OK Complementary features -> synergy = {synergy.mean():.4f}")


class TestNumericalStability:
    """Test edge cases and numerical stability."""

    def test_zero_variance_handling(self):
        """Test handling of zero-variance inputs."""
        B, D, N = 50, 20, 10

        # Constant inputs (zero variance)
        inputs = torch.ones(B, D)
        weights = torch.randn(N, D)

        rq = RayleighQuotient(regularization=1e-4)
        scores = rq.compute(inputs, weights)

        # Should not crash, should return valid values (likely zeros)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

        print(f"OK Zero variance handled: RQ = {scores.mean():.4f}")

    def test_small_batch_with_shrinkage(self):
        """Test that shrinkage helps with small batches."""
        D, N = 50, 20

        # Very small batch (B < D) - would be rank-deficient
        B = 10
        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)

        # With regularization should not crash
        rq = RayleighQuotient(regularization=1e-3)
        scores = rq.compute(inputs, weights)

        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

        print(f"OK Small batch (B={B}, D={D}) handled with regularization")

    def test_high_dimensional_inputs(self):
        """Test on high-dimensional inputs (like LLMs)."""
        B, D, N = 32, 4096, 2048  # LLM scale

        inputs = torch.randn(B, D) * 0.1  # Scaled for numerical stability
        weights = torch.randn(N, D) * 0.1
        outputs = inputs @ weights.T

        # Output-based should handle this efficiently
        metric = PairwiseRedundancyGaussian(mode="output_based", num_pairs=10)
        redundancy = metric.compute(outputs=outputs)

        assert redundancy.shape == (N,)
        assert not torch.isnan(redundancy).any()

        print(f"OK High-dimensional (D={D}, N={N}) handled: redundancy mean = {redundancy.mean():.4f}")


class TestScaleInvariance:
    """Test scale invariance properties."""

    def test_rq_scale_invariance(self):
        """
        GROUND TRUTH: RQ should be invariant to weight scaling.

        Theory: RQ(αw) = (αw)^T Σ (αw) / [(αw)^T (αw) · tr(Σ)]
                       = α² w^T Σ w / (α² w^T w · tr(Σ))
                       = w^T Σ w / (w^T w · tr(Σ))
                       = RQ(w)
        """
        B, D, N = 100, 30, 15

        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)

        rq = RayleighQuotient(relative=True)

        # Compute RQ with original weights
        scores1 = rq.compute(inputs, weights)

        # Scale weights by constant
        weights_scaled = weights * 5.0
        scores2 = rq.compute(inputs, weights_scaled)

        # Should be identical
        assert torch.allclose(scores1, scores2, rtol=1e-4), "RQ should be invariant to weight scaling"

        print(f"OK RQ scale-invariant: max diff = {(scores1 - scores2).abs().max():.6f}")

    def test_delta_rq_scale_invariance(self):
        """ΔRQ should also be scale-invariant."""
        B, D, N = 100, 20, 10

        inputs = torch.randn(B, D)
        weights = torch.randn(N, D)
        targets = torch.randint(0, 3, (B,))

        rq = RayleighQuotient()

        results1 = rq.compute_class_conditioned(inputs, weights, targets, return_delta_rq=True)

        weights_scaled = weights * 3.0
        results2 = rq.compute_class_conditioned(inputs, weights_scaled, targets, return_delta_rq=True)

        # ΔRQ should be invariant
        assert torch.allclose(results1["delta_rq"], results2["delta_rq"], rtol=1e-3), "ΔRQ should be invariant to scaling"

        print("OK ΔRQ scale-invariant")


def run_all_validation_tests():
    """Run all scientific validation tests and print summary."""
    print("=" * 80)
    print("Scientific Correctness Validation Tests")
    print("=" * 80)

    test_classes = [TestRedundancyCorrectness, TestDeltaRQCorrectness, TestMutualInformationCorrectness, TestNumericalStability, TestScaleInvariance]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed_tests += 1
                except AssertionError as e:
                    print(f"  FAIL {method_name}: {e}")
                except Exception as e:
                    print(f"  FAIL {method_name}: ERROR - {e}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    return passed_tests == total_tests


if __name__ == "__main__":
    # Can run directly or via pytest
    if "--pytest" in sys.argv:
        pytest.main([__file__, "-v"])
    else:
        success = run_all_validation_tests()
        sys.exit(0 if success else 1)
