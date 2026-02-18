"""
Tests for metrics/gradient_based.py: gradient-based metrics and utilities.
"""

import pytest
import torch

from alignment.metrics.gradient_based import (
    GradientAlignment,
    GradientStatisticsTracker,
    LocalLearningRuleSearch,
    TaylorSaliency,
    compute_direction_consistency,
)

# =========================================================================
# TaylorSaliency
# =========================================================================


class TestTaylorSaliency:
    def test_abs_mean_2d(self):
        outputs = torch.randn(16, 8)
        gradients = torch.randn(16, 8)
        metric = TaylorSaliency(mode="abs_mean")
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_mean_abs_2d(self):
        outputs = torch.randn(16, 8)
        gradients = torch.randn(16, 8)
        metric = TaylorSaliency(mode="mean_abs")
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_sq_mean_2d(self):
        outputs = torch.randn(16, 8)
        gradients = torch.randn(16, 8)
        metric = TaylorSaliency(mode="sq_mean")
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores.shape == (8,)
        assert (scores >= 0).all()

    def test_4d_cnn_outputs(self):
        outputs = torch.randn(4, 8, 4, 4)  # [B, C, H, W]
        gradients = torch.randn(4, 8, 4, 4)
        metric = TaylorSaliency(mode="abs_mean")
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores.shape == (8,)  # Per-channel scores

    def test_4d_all_modes(self):
        outputs = torch.randn(4, 8, 4, 4)
        gradients = torch.randn(4, 8, 4, 4)
        for mode in ["abs_mean", "mean_abs", "sq_mean"]:
            metric = TaylorSaliency(mode=mode)
            scores = metric.compute(outputs=outputs, gradients=gradients)
            assert scores.shape == (8,)

    def test_3d_inputs_flattened(self):
        outputs = torch.randn(4, 3, 8)  # [B, T, N]
        gradients = torch.randn(4, 3, 8)
        metric = TaylorSaliency()
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores.shape == (8,)

    def test_requires_both(self):
        metric = TaylorSaliency()
        with pytest.raises(ValueError, match="requires both"):
            metric.compute(outputs=None, gradients=torch.randn(4, 8))
        with pytest.raises(ValueError, match="requires both"):
            metric.compute(outputs=torch.randn(4, 8), gradients=None)

    def test_shape_mismatch_raises(self):
        metric = TaylorSaliency()
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.compute(outputs=torch.randn(4, 8), gradients=torch.randn(4, 6))

    def test_unknown_mode_raises(self):
        metric = TaylorSaliency(mode="bogus")
        with pytest.raises(ValueError, match="Unknown"):
            metric.compute(outputs=torch.randn(4, 8), gradients=torch.randn(4, 8))

    def test_unknown_mode_4d_raises(self):
        metric = TaylorSaliency(mode="bogus")
        with pytest.raises(ValueError, match="Unknown"):
            metric.compute(
                outputs=torch.randn(4, 8, 2, 2),
                gradients=torch.randn(4, 8, 2, 2),
            )

    def test_properties(self):
        metric = TaylorSaliency()
        assert metric.requires_inputs is False
        assert metric.requires_weights is False
        assert metric.requires_outputs is True
        assert metric.requires_gradients is True

    def test_high_activation_high_gradient(self):
        """Neurons with large activation*gradient should score high."""
        outputs = torch.zeros(10, 4)
        gradients = torch.zeros(10, 4)
        # Neuron 0: large product
        outputs[:, 0] = 10.0
        gradients[:, 0] = 10.0
        # Neuron 1: small product
        outputs[:, 1] = 0.1
        gradients[:, 1] = 0.1
        metric = TaylorSaliency(mode="abs_mean")
        scores = metric.compute(outputs=outputs, gradients=gradients)
        assert scores[0] > scores[1]


# =========================================================================
# GradientAlignment
# =========================================================================


class TestGradientAlignment:
    def test_hebbian_basic(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)  # [N, D_in]
        metric = GradientAlignment(local_signal="hebbian")
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert scores.shape == (4,)
        assert (scores >= 0).all()

    def test_anti_hebbian(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        targets = torch.randint(0, 4, (16,))
        metric = GradientAlignment(local_signal="anti_hebbian")
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients, targets=targets)
        assert scores.shape == (4,)

    def test_anti_hebbian_no_targets(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = GradientAlignment(local_signal="anti_hebbian")
        # Falls back to hebbian
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert scores.shape == (4,)

    def test_oja_signal(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = GradientAlignment(local_signal="oja")
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert scores.shape == (4,)

    def test_output_signal(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = GradientAlignment(local_signal="output")
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert scores.shape == (4,)

    def test_input_signal(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = GradientAlignment(local_signal="input")
        scores = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert scores.shape == (4,)

    def test_unknown_signal_raises(self):
        metric = GradientAlignment(local_signal="bogus")
        with pytest.raises(ValueError, match="Unknown local signal"):
            metric.compute(
                inputs=torch.randn(16, 8),
                outputs=torch.randn(16, 4),
                gradients=torch.randn(4, 8),
            )

    def test_no_gradients_returns_zeros(self):
        metric = GradientAlignment()
        scores = metric.compute(
            inputs=torch.randn(16, 8),
            outputs=torch.randn(16, 4),
            gradients=None,
        )
        assert scores.shape == (4,)
        assert (scores == 0).all()

    def test_requires_inputs_and_outputs(self):
        metric = GradientAlignment()
        with pytest.raises(ValueError, match="requires inputs"):
            metric.compute(inputs=None, outputs=torch.randn(4, 8))

    def test_normalize_flag(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        m_norm = GradientAlignment(normalize=True)
        m_raw = GradientAlignment(normalize=False)
        s_norm = m_norm.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        s_raw = m_raw.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert s_norm.shape == s_raw.shape

    def test_properties(self):
        metric = GradientAlignment()
        assert metric.requires_inputs is True
        assert metric.requires_outputs is True


# =========================================================================
# LocalLearningRuleSearch
# =========================================================================


class TestLocalLearningRuleSearch:
    def test_basic(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = LocalLearningRuleSearch()
        indices = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert indices.shape == (4,)
        assert all(0 <= idx < 5 for idx in indices.tolist())

    def test_return_correlations(self):
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        metric = LocalLearningRuleSearch()
        corr_matrix = metric.compute(
            inputs=inputs,
            outputs=outputs,
            gradients=gradients,
            return_correlations=True,
        )
        assert corr_matrix.shape == (4, 5)  # 4 neurons, 5 rules

    def test_requires_gradients(self):
        metric = LocalLearningRuleSearch()
        with pytest.raises(ValueError, match="requires gradients"):
            metric.compute(
                inputs=torch.randn(8, 4),
                outputs=torch.randn(8, 4),
                gradients=None,
            )

    def test_custom_rules(self):
        metric = LocalLearningRuleSearch(candidate_rules=["hebbian", "oja"])
        inputs = torch.randn(16, 8)
        outputs = torch.randn(16, 4)
        gradients = torch.randn(4, 8)
        indices = metric.compute(inputs=inputs, outputs=outputs, gradients=gradients)
        assert all(0 <= idx < 2 for idx in indices.tolist())

    def test_get_learning_rule_for_neuron(self):
        metric = LocalLearningRuleSearch()
        assert metric.get_learning_rule_for_neuron(0, 0) == "hebbian"
        assert metric.get_learning_rule_for_neuron(0, 1) == "anti_hebbian"


# =========================================================================
# GradientStatisticsTracker
# =========================================================================


class TestGradientStatisticsTracker:
    def test_register_layer(self):
        tracker = GradientStatisticsTracker()
        tracker.register_layer("conv1")
        assert "conv1" in tracker.gradient_history
        assert "conv1" in tracker.signal_history
        assert "conv1" in tracker.correlation_history

    def test_update(self):
        tracker = GradientStatisticsTracker()
        grad = torch.randn(4, 8)
        signal = torch.randn(4, 8)
        tracker.update("conv1", grad, signal)
        assert len(tracker.gradient_history["conv1"]) == 1
        assert len(tracker.signal_history["conv1"]) == 1
        assert len(tracker.correlation_history["conv1"]) == 1

    def test_average_correlation(self):
        tracker = GradientStatisticsTracker()
        for _ in range(5):
            g = torch.randn(4, 8)
            tracker.update("fc1", g, g)  # Perfect correlation
        avg = tracker.get_average_correlation("fc1")
        assert abs(avg - 1.0) < 0.01  # Should be ~1.0

    def test_average_correlation_unknown_layer(self):
        tracker = GradientStatisticsTracker()
        assert tracker.get_average_correlation("nonexistent") == 0.0


# =========================================================================
# compute_direction_consistency
# =========================================================================


class TestDirectionConsistency:
    def test_identical_gradients(self):
        g = torch.randn(16)
        history = [g.clone() for _ in range(5)]
        assert compute_direction_consistency(history) == pytest.approx(1.0, abs=0.01)

    def test_single_gradient(self):
        assert compute_direction_consistency([torch.randn(8)]) == 1.0

    def test_empty_list(self):
        assert compute_direction_consistency([]) == 1.0

    def test_random_gradients(self):
        history = [torch.randn(16) for _ in range(10)]
        consistency = compute_direction_consistency(history)
        assert 0.0 <= consistency <= 1.0
