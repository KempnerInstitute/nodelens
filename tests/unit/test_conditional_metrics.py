"""
Tests for metrics/conditional_metrics.py: class-conditioned metrics.
"""

import pytest
import torch

from alignment.metrics.conditional_metrics import (
    ConditionalRayleighQuotient,
    MIAboutClass,
    ConditionalActivationNorm,
    DeltaRQ,
    ConditionalMIGaussian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class_data(n_per_class=20, n_features=8, n_classes=3, seed=42):
    """Create synthetic class-separated data."""
    rng = torch.Generator().manual_seed(seed)
    inputs_list, targets_list = [], []
    for c in range(n_classes):
        # Each class centered at different location
        center = torch.zeros(n_features)
        center[c % n_features] = 3.0
        x = center + 0.5 * torch.randn(n_per_class, n_features, generator=rng)
        inputs_list.append(x)
        targets_list.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(inputs_list), torch.cat(targets_list)


# =========================================================================
# ConditionalRayleighQuotient
# =========================================================================


class TestConditionalRayleighQuotient:
    def test_basic_output_shape(self):
        inputs, targets = _make_class_data(n_per_class=15, n_features=8)
        weights = torch.randn(4, 8)
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert scores.shape == (4,)

    def test_scores_finite(self):
        inputs, targets = _make_class_data()
        weights = torch.randn(6, 8)
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert torch.isfinite(scores).all()

    def test_no_targets_falls_back(self):
        inputs = torch.randn(30, 8)
        weights = torch.randn(4, 8)
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=None)
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_requires_inputs_and_weights(self):
        metric = ConditionalRayleighQuotient()
        with pytest.raises(ValueError, match="requires both"):
            metric.compute(inputs=None, weights=torch.randn(4, 8))
        with pytest.raises(ValueError, match="requires both"):
            metric.compute(inputs=torch.randn(10, 8), weights=None)

    def test_relative_mode(self):
        inputs, targets = _make_class_data()
        weights = torch.randn(4, 8)
        metric_rel = ConditionalRayleighQuotient(relative=True)
        metric_abs = ConditionalRayleighQuotient(relative=False)
        scores_rel = metric_rel.compute(inputs=inputs, weights=weights, targets=targets)
        scores_abs = metric_abs.compute(inputs=inputs, weights=weights, targets=targets)
        # Relative scores should generally be smaller
        assert not torch.allclose(scores_rel, scores_abs)

    def test_return_delta(self):
        inputs, targets = _make_class_data()
        weights = torch.randn(4, 8)
        metric = ConditionalRayleighQuotient(return_delta=True)
        delta = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert delta.shape == (4,)
        assert torch.isfinite(delta).all()

    def test_dim_mismatch_handled(self):
        inputs = torch.randn(30, 10)
        weights = torch.randn(4, 8)  # Different input dim
        targets = torch.randint(0, 3, (30,))
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert scores.shape == (4,)

    def test_high_dim_inputs_flattened(self):
        inputs = torch.randn(30, 2, 4)  # 3D -> flatten to (30, 8)
        weights = torch.randn(4, 8)
        targets = torch.randint(0, 2, (30,))
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert scores.shape == (4,)

    def test_batch_mismatch_with_patches(self):
        # Simulate unfolded CNN: 30 samples * 4 patches = 120 rows
        inputs = torch.randn(120, 8)
        weights = torch.randn(4, 8)
        targets = torch.randint(0, 3, (30,))  # Original batch size
        metric = ConditionalRayleighQuotient()
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert scores.shape == (4,)

    def test_min_samples_filtering(self):
        inputs = torch.randn(10, 8)
        weights = torch.randn(4, 8)
        # Only 1 sample per class (below min_samples=2)
        targets = torch.arange(10)
        metric = ConditionalRayleighQuotient(min_samples=2)
        scores = metric.compute(inputs=inputs, weights=weights, targets=targets)
        # Should return zeros since no class has enough samples
        assert scores.shape == (4,)

    def test_properties(self):
        metric = ConditionalRayleighQuotient()
        assert metric.requires_inputs is True
        assert metric.requires_weights is True
        assert metric.requires_outputs is False


# =========================================================================
# MIAboutClass
# =========================================================================


class TestMIAboutClass:
    def test_gaussian_basic(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = MIAboutClass(method="gaussian")
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()
        assert (scores >= 0).all()

    def test_binning_basic(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = MIAboutClass(method="binning", bins=5)
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_no_targets_returns_zeros(self):
        outputs = torch.randn(30, 8)
        metric = MIAboutClass()
        scores = metric.compute(outputs=outputs, targets=None)
        assert (scores == 0).all()

    def test_requires_outputs(self):
        metric = MIAboutClass()
        with pytest.raises(ValueError, match="requires outputs"):
            metric.compute(outputs=None)

    def test_high_dim_outputs_flattened(self):
        outputs = torch.randn(30, 2, 4)  # 3D -> flatten to (30, 8)
        targets = torch.randint(0, 2, (30,))
        metric = MIAboutClass(method="gaussian")
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)

    def test_batch_mismatch_with_patches(self):
        outputs = torch.randn(120, 8)
        targets = torch.randint(0, 3, (30,))
        metric = MIAboutClass()
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)

    def test_class_informative_neuron(self):
        """A neuron perfectly correlated with class should have high MI."""
        torch.manual_seed(42)
        n = 100
        targets = torch.randint(0, 2, (n,))
        outputs = torch.randn(n, 4)
        # Make neuron 0 perfectly class-dependent
        outputs[:, 0] = targets.float() * 5.0 + 0.1 * torch.randn(n)
        metric = MIAboutClass(method="gaussian", min_samples_per_class=5)
        scores = metric.compute(outputs=outputs, targets=targets)
        # Neuron 0 should have highest MI
        assert scores[0] > scores[1:]  .mean()

    def test_properties(self):
        metric = MIAboutClass()
        assert metric.requires_inputs is False
        assert metric.requires_weights is False
        assert metric.requires_outputs is True


# =========================================================================
# ConditionalActivationNorm
# =========================================================================


class TestConditionalActivationNorm:
    def test_mean_aggregation(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = ConditionalActivationNorm(aggregation="mean")
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()

    def test_max_aggregation(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = ConditionalActivationNorm(aggregation="max")
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)

    def test_variance_aggregation(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = ConditionalActivationNorm(aggregation="variance")
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_ratio_aggregation(self):
        outputs = torch.abs(torch.randn(60, 8)) + 0.1  # Positive activations
        targets = torch.randint(0, 3, (60,))
        metric = ConditionalActivationNorm(aggregation="ratio", normalize=False)
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert (scores >= 1.0 - 1e-6).all()  # ratio >= 1 (without normalization)

    def test_unknown_aggregation_raises(self):
        metric = ConditionalActivationNorm(aggregation="bogus")
        with pytest.raises(ValueError, match="Unknown aggregation"):
            metric.compute(outputs=torch.randn(10, 4), targets=torch.zeros(10, dtype=torch.long))

    def test_no_targets_fallback(self):
        outputs = torch.randn(30, 8)
        metric = ConditionalActivationNorm()
        scores = metric.compute(outputs=outputs, targets=None)
        assert scores.shape == (8,)

    def test_requires_outputs(self):
        metric = ConditionalActivationNorm()
        with pytest.raises(ValueError, match="requires outputs"):
            metric.compute(outputs=None)

    def test_cnn_4d_outputs(self):
        outputs = torch.randn(10, 8, 4, 4)  # [B, C, H, W]
        targets = torch.randint(0, 3, (10,))
        metric = ConditionalActivationNorm()
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)

    def test_normalize_flag(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric_norm = ConditionalActivationNorm(normalize=True)
        metric_raw = ConditionalActivationNorm(normalize=False)
        s_norm = metric_norm.compute(outputs=outputs, targets=targets)
        s_raw = metric_raw.compute(outputs=outputs, targets=targets)
        assert not torch.allclose(s_norm, s_raw)

    def test_properties(self):
        metric = ConditionalActivationNorm()
        assert metric.requires_inputs is False
        assert metric.requires_weights is False
        assert metric.requires_outputs is True


# =========================================================================
# DeltaRQ
# =========================================================================


class TestDeltaRQ:
    def test_basic(self):
        inputs, targets = _make_class_data()
        weights = torch.randn(4, 8)
        metric = DeltaRQ()
        delta = metric.compute(inputs=inputs, weights=weights, targets=targets)
        assert delta.shape == (4,)
        assert torch.isfinite(delta).all()

    def test_properties(self):
        metric = DeltaRQ()
        assert metric.requires_inputs is True
        assert metric.requires_weights is True
        assert metric.requires_outputs is False


# =========================================================================
# ConditionalMIGaussian
# =========================================================================


class TestConditionalMIGaussian:
    def test_basic(self):
        outputs = torch.randn(60, 8)
        targets = torch.randint(0, 3, (60,))
        metric = ConditionalMIGaussian()
        scores = metric.compute(outputs=outputs, targets=targets)
        assert scores.shape == (8,)
        assert torch.isfinite(scores).all()

    def test_no_targets(self):
        outputs = torch.randn(30, 8)
        metric = ConditionalMIGaussian()
        scores = metric.compute(outputs=outputs, targets=None)
        assert scores.shape == (8,)

    def test_requires_outputs(self):
        metric = ConditionalMIGaussian()
        with pytest.raises(ValueError, match="requires outputs"):
            metric.compute(outputs=None)

    def test_with_target_outputs(self):
        outputs = torch.randn(30, 8)
        targets = torch.randint(0, 2, (30,))
        target_outputs = torch.randn(30, 1)
        metric = ConditionalMIGaussian(use_pc_reference=False)
        scores = metric.compute(
            outputs=outputs, targets=targets, target_outputs=target_outputs
        )
        assert scores.shape == (8,)

    def test_properties(self):
        metric = ConditionalMIGaussian(use_pc_reference=True)
        assert metric.requires_inputs is True
        assert metric.requires_outputs is True
