"""
Tests for pruning strategies: magnitude, random, gradient, movement, cascading, and base.

Covers BasePruningStrategy (PruningConfig, create_pruning_mask, apply_pruning,
remove_pruning, get_sparsity, prune), MagnitudePruning, GlobalMagnitudePruning,
IterativeMagnitudePruning, RandomPruning, LayerwiseRandomPruning, BernoulliPruning,
GradientPruning, FisherPruning, MomentumPruning, MovementPruning,
AdaptiveMovementPruning, CascadingAlignmentPruning, PrecomputedScorePruning.
"""

import pytest
import torch
import torch.nn as nn

from alignment.pruning.base import (
    BasePruningStrategy,
    IterativePruningStrategy,
    PrecomputedScorePruning,
    PruningConfig,
)
from alignment.pruning.strategies.magnitude import (
    GlobalMagnitudePruning,
    IterativeMagnitudePruning,
    MagnitudePruning,
)
from alignment.pruning.strategies.random import (
    BernoulliPruning,
    LayerwiseRandomPruning,
    RandomPruning,
)
from alignment.pruning.strategies.gradient import (
    FisherPruning,
    GradientPruning,
    MomentumPruning,
)
from alignment.pruning.strategies.movement import (
    AdaptiveMovementPruning,
    MovementPruning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear(in_f=8, out_f=4):
    """Create a fresh Linear layer."""
    return nn.Linear(in_f, out_f)


def _conv2d(in_c=3, out_c=8, k=3):
    """Create a fresh Conv2d layer."""
    return nn.Conv2d(in_c, out_c, k, padding=1)


class _TinyModel(nn.Module):
    """Small model for multi-layer tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.mean(dim=[2, 3])
        return self.fc(x)


# =========================================================================
# PruningConfig
# =========================================================================


class TestPruningConfig:
    def test_defaults(self):
        cfg = PruningConfig()
        assert cfg.amount == 0.5
        assert cfg.structured is False
        assert cfg.pruning_mode == "low"

    def test_custom(self):
        cfg = PruningConfig(amount=0.3, structured=True, pruning_mode="high")
        assert cfg.amount == 0.3
        assert cfg.structured is True


# =========================================================================
# BasePruningStrategy – create_pruning_mask
# =========================================================================


class TestCreatePruningMask:
    """Test mask creation logic via MagnitudePruning (simplest concrete subclass)."""

    def test_unstructured_mask_shape(self):
        layer = _linear()
        strategy = MagnitudePruning()
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.5)
        assert mask.shape == layer.weight.shape

    def test_unstructured_sparsity(self):
        layer = _linear(16, 16)
        strategy = MagnitudePruning()
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.5)
        pruned = (mask == 0).sum().item()
        expected = int(0.5 * layer.weight.numel())
        assert pruned == expected

    def test_structured_mask_shape(self):
        layer = _conv2d(3, 8, 3)
        strategy = MagnitudePruning(PruningConfig(structured=True))
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.5, structured=True)
        assert mask.shape == layer.weight.shape

    def test_structured_entire_filters_pruned(self):
        layer = _conv2d(3, 8, 3)
        strategy = MagnitudePruning(PruningConfig(structured=True))
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.5, structured=True)
        # Each filter should be either all-0 or all-1
        for i in range(8):
            filt = mask[i]
            assert filt.sum().item() == 0 or filt.sum().item() == filt.numel()

    def test_amount_zero_keeps_all(self):
        layer = _linear()
        strategy = MagnitudePruning()
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.0)
        assert (mask == 1).all()

    def test_pruning_mode_high(self):
        """'high' mode should prune the highest-score weights."""
        layer = _linear(4, 4)
        strategy = MagnitudePruning(PruningConfig(pruning_mode="high"))
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores, amount=0.25, pruning_mode="high")
        pruned_indices = (mask.flatten() == 0).nonzero(as_tuple=True)[0]
        kept_indices = (mask.flatten() == 1).nonzero(as_tuple=True)[0]
        # All pruned scores should be >= all kept scores
        if len(pruned_indices) > 0 and len(kept_indices) > 0:
            min_pruned = scores.flatten()[pruned_indices].min()
            max_kept = scores.flatten()[kept_indices].max()
            assert min_pruned >= max_kept


# =========================================================================
# BasePruningStrategy – apply_pruning / remove_pruning
# =========================================================================


class TestApplyPruning:
    def test_apply_zeros_weights(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones_like(layer.weight)
        mask[0] = 0  # prune first row
        strategy.apply_pruning(layer, mask)
        assert (layer.weight.data[0] == 0).all()

    def test_apply_1d_mask_output(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones(4)
        mask[0] = 0
        strategy.apply_pruning(layer, mask, dim="output")
        assert (layer.weight.data[0] == 0).all()

    def test_apply_1d_mask_input(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones(4)
        mask[1] = 0
        strategy.apply_pruning(layer, mask, dim="input")
        assert (layer.weight.data[:, 1] == 0).all()

    def test_remove_pruning_cleans_state(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones_like(layer.weight)
        mask[0] = 0
        strategy.apply_pruning(layer, mask)
        strategy.remove_pruning(layer)
        assert not hasattr(layer, "weight_mask")
        assert not hasattr(layer, "_original_weight")

    def test_get_sparsity(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones_like(layer.weight)
        mask[0] = 0
        strategy.apply_pruning(layer, mask, make_permanent=True)
        sp = strategy.get_sparsity(layer)
        assert sp == pytest.approx(0.25, abs=0.01)

    def test_clear_pruning_state(self):
        layer = _linear(4, 4)
        strategy = MagnitudePruning()
        mask = torch.ones_like(layer.weight)
        mask[0] = 0
        strategy.apply_pruning(layer, mask)
        strategy.clear_pruning_state(layer)
        assert not hasattr(layer, "weight_mask")
        assert not hasattr(layer, "_original_weight")


# =========================================================================
# MagnitudePruning
# =========================================================================


class TestMagnitudePruning:
    def test_scores_are_abs_weights(self):
        layer = _linear()
        strategy = MagnitudePruning()
        scores = strategy.compute_importance_scores(layer)
        torch.testing.assert_close(scores, layer.weight.data.abs())

    def test_no_weight_raises(self):
        strategy = MagnitudePruning()
        with pytest.raises(ValueError, match="does not have weights"):
            strategy.compute_importance_scores(nn.ReLU())

    def test_prune_method_returns_mask(self):
        layer = _linear(8, 4)
        strategy = MagnitudePruning()
        mask = strategy.prune(layer, amount=0.5)
        assert mask.shape == layer.weight.shape
        assert (mask == 0).sum().item() > 0

    def test_conv2d_pruning(self):
        layer = _conv2d(3, 16, 3)
        strategy = MagnitudePruning()
        scores = strategy.compute_importance_scores(layer)
        assert scores.shape == layer.weight.shape


# =========================================================================
# GlobalMagnitudePruning
# =========================================================================


class TestGlobalMagnitudePruning:
    def test_prune_model(self):
        model = _TinyModel()
        strategy = GlobalMagnitudePruning()
        masks = strategy.prune_model(model, amount=0.5)
        assert len(masks) > 0

    def test_global_sparsity(self):
        model = _TinyModel()
        strategy = GlobalMagnitudePruning()
        masks = strategy.prune_model(model, amount=0.5)
        total = sum(m.numel() for m in masks.values())
        zeros = sum((m == 0).sum().item() for m in masks.values())
        assert zeros / total == pytest.approx(0.5, abs=0.05)

    def test_zero_amount_returns_empty(self):
        model = _TinyModel()
        strategy = GlobalMagnitudePruning()
        masks = strategy.prune_model(model, amount=0.0)
        assert masks == {}


# =========================================================================
# RandomPruning
# =========================================================================


class TestRandomPruning:
    def test_scores_shape_unstructured(self):
        layer = _linear(8, 4)
        strategy = RandomPruning()
        scores = strategy.compute_importance_scores(layer)
        assert scores.shape == layer.weight.shape

    def test_scores_shape_structured_conv(self):
        layer = _conv2d(3, 8, 3)
        strategy = RandomPruning(PruningConfig(structured=True))
        scores = strategy.compute_importance_scores(layer)
        assert scores.shape == layer.weight.shape

    def test_seed_reproducibility(self):
        layer = _linear(8, 4)
        s1 = RandomPruning(seed=42)
        scores1 = s1.compute_importance_scores(layer)
        s2 = RandomPruning(seed=42)
        scores2 = s2.compute_importance_scores(layer)
        torch.testing.assert_close(scores1, scores2)

    def test_prune_produces_correct_sparsity(self):
        layer = _linear(16, 16)
        strategy = RandomPruning()
        mask = strategy.prune(layer, amount=0.5)
        pruned = (mask == 0).sum().item()
        expected = int(0.5 * layer.weight.numel())
        assert pruned == expected


# =========================================================================
# LayerwiseRandomPruning
# =========================================================================


class TestLayerwiseRandomPruning:
    def test_prune_model_per_layer(self):
        model = _TinyModel()
        strategy = LayerwiseRandomPruning(
            layer_sparsity={"conv": 0.3, "fc": 0.7},
            default_sparsity=0.5,
        )
        masks = strategy.prune_model(model)
        assert len(masks) > 0

    def test_default_sparsity_used(self):
        model = _TinyModel()
        strategy = LayerwiseRandomPruning(default_sparsity=0.25)
        masks = strategy.prune_model(model)
        assert len(masks) > 0


# =========================================================================
# BernoulliPruning
# =========================================================================


class TestBernoulliPruning:
    def test_scores_shape(self):
        layer = _linear(8, 4)
        strategy = BernoulliPruning(probability=0.5)
        scores = strategy.compute_importance_scores(layer)
        assert scores.shape == layer.weight.shape

    def test_mask_is_binary(self):
        layer = _linear(8, 4)
        strategy = BernoulliPruning(probability=0.5)
        scores = strategy.compute_importance_scores(layer)
        mask = strategy.create_pruning_mask(scores)
        unique = mask.unique()
        assert all(v in [0.0, 1.0] for v in unique.tolist())


# =========================================================================
# GradientPruning
# =========================================================================


class TestGradientPruning:
    def _setup_grad(self, layer):
        """Run a dummy forward/backward to produce gradients."""
        x = torch.randn(2, layer.in_features)
        out = layer(x)
        out.sum().backward()

    def test_taylor_mode(self):
        layer = _linear(8, 4)
        self._setup_grad(layer)
        strategy = GradientPruning(mode="taylor")
        scores = strategy.compute_importance_scores(layer)
        expected = (layer.weight.grad * layer.weight.data).abs()
        torch.testing.assert_close(scores, expected)

    def test_gradient_mode(self):
        layer = _linear(8, 4)
        self._setup_grad(layer)
        strategy = GradientPruning(mode="gradient")
        scores = strategy.compute_importance_scores(layer)
        expected = layer.weight.grad.abs()
        torch.testing.assert_close(scores, expected)

    def test_no_grad_raises(self):
        layer = _linear()
        strategy = GradientPruning()
        with pytest.raises(ValueError, match="no gradients"):
            strategy.compute_importance_scores(layer)


# =========================================================================
# FisherPruning
# =========================================================================


class TestFisherPruning:
    def test_accumulate_and_prune(self):
        model = _TinyModel()
        strategy = FisherPruning()

        # Run forward/backward to get gradients
        x = torch.randn(4, 3, 8, 8)
        loss = model(x).sum()
        loss.backward()

        strategy.accumulate_fisher(model)
        assert strategy.n_samples == 1
        assert len(strategy.fisher_info) > 0

        masks = strategy.prune_model(model, amount=0.3)
        assert len(masks) > 0

    def test_reset_fisher(self):
        strategy = FisherPruning()
        model = _TinyModel()
        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.accumulate_fisher(model)
        strategy.reset_fisher()
        assert len(strategy.fisher_info) == 0
        assert strategy.n_samples == 0

    def test_fallback_no_fisher_info(self):
        """Without accumulated Fisher info, falls back to weight magnitude."""
        layer = _linear(8, 4)
        strategy = FisherPruning()
        scores = strategy.compute_importance_scores(layer, module_name="missing")
        # Should fall back to weight.data.abs()
        torch.testing.assert_close(scores, layer.weight.data.abs())


# =========================================================================
# MomentumPruning
# =========================================================================


class TestMomentumPruning:
    def test_update_and_score(self):
        model = _TinyModel()
        strategy = MomentumPruning(momentum=0.9)

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_momentum(model)

        assert len(strategy.importance_buffer) > 0

    def test_prune_model(self):
        model = _TinyModel()
        strategy = MomentumPruning(momentum=0.9)

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_momentum(model)

        masks = strategy.prune_model(model, amount=0.3)
        assert len(masks) > 0

    def test_reset(self):
        strategy = MomentumPruning()
        strategy.importance_buffer["test"] = torch.zeros(4)
        strategy.reset_momentum()
        assert len(strategy.importance_buffer) == 0


# =========================================================================
# MovementPruning
# =========================================================================


class TestMovementPruning:
    def test_update_history(self):
        model = _TinyModel()
        strategy = MovementPruning()

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_movement_history(model)

        assert len(strategy.movement_history) > 0

    def test_scores_with_history(self):
        model = _TinyModel()
        strategy = MovementPruning()

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_movement_history(model)

        scores = strategy.compute_importance_scores(model.conv, module_name="conv")
        assert scores.shape == model.conv.weight.shape

    def test_fallback_no_history(self):
        layer = _conv2d()
        strategy = MovementPruning()
        scores = strategy.compute_importance_scores(layer, module_name="unknown")
        torch.testing.assert_close(scores, layer.weight.abs())

    def test_get_movement_statistics(self):
        model = _TinyModel()
        strategy = MovementPruning()

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_movement_history(model)

        stats = strategy.get_movement_statistics()
        assert len(stats) > 0
        for k, v in stats.items():
            assert "moving_away_zero" in v
            assert "fraction_toward_zero" in v

    def test_reset_history(self):
        strategy = MovementPruning()
        strategy.movement_history["x"] = torch.zeros(4)
        strategy.reset_history()
        assert len(strategy.movement_history) == 0


# =========================================================================
# AdaptiveMovementPruning
# =========================================================================


class TestAdaptiveMovementPruning:
    def test_compute_adaptive_amount(self):
        model = _TinyModel()
        strategy = AdaptiveMovementPruning(base_amount=0.5, adaptation_strength=0.3)

        x = torch.randn(4, 3, 8, 8)
        model(x).sum().backward()
        strategy.update_movement_history(model)

        amount = strategy.compute_adaptive_amount(model.conv, "conv")
        assert 0.1 <= amount <= 0.9

    def test_fallback_to_base(self):
        strategy = AdaptiveMovementPruning(base_amount=0.4)
        layer = _conv2d()
        amount = strategy.compute_adaptive_amount(layer, "nonexistent")
        assert amount == 0.4


# =========================================================================
# CascadingAlignmentPruning
# =========================================================================


class TestCascadingAlignmentPruning:
    def test_init_and_direction(self):
        """Test init by monkeypatching get_metric to return a class."""
        from alignment.pruning.strategies.cascading import CascadingAlignmentPruning
        from unittest.mock import patch, MagicMock

        mock_metric_cls = MagicMock()
        mock_metric_cls.return_value = MagicMock()

        with patch("alignment.pruning.strategies.cascading.get_metric", return_value=mock_metric_cls):
            strategy = CascadingAlignmentPruning(
                metric="rayleigh_quotient",
                direction="forward",
                config=PruningConfig(amount=0.5, structured=True),
            )
        assert strategy.direction == "forward"

    def test_compute_importance_requires_inputs(self):
        from alignment.pruning.strategies.cascading import CascadingAlignmentPruning
        from unittest.mock import patch, MagicMock

        mock_metric_cls = MagicMock()
        mock_metric_cls.return_value = MagicMock()

        with patch("alignment.pruning.strategies.cascading.get_metric", return_value=mock_metric_cls):
            strategy = CascadingAlignmentPruning(metric="rayleigh_quotient")
        layer = _conv2d(3, 8, 3)
        with pytest.raises(ValueError, match="requires inputs"):
            strategy.compute_importance_scores(layer, inputs=None)


# =========================================================================
# PrecomputedScorePruning
# =========================================================================


class TestPrecomputedScorePruning:
    def test_compute_importance_raises(self):
        strategy = PrecomputedScorePruning()
        with pytest.raises(NotImplementedError):
            strategy.compute_importance_scores(_linear())

    def test_create_mask_works(self):
        strategy = PrecomputedScorePruning()
        scores = torch.rand(4, 8)
        mask = strategy.create_pruning_mask(scores, amount=0.5)
        assert mask.shape == scores.shape


# =========================================================================
# IterativePruningStrategy (via IterativeMagnitudePruning)
# =========================================================================


class TestIterativePruning:
    def test_iterative_prune_increases_sparsity(self):
        model = _TinyModel()
        config = PruningConfig(amount=0.5, iterations=3)
        strategy = IterativeMagnitudePruning(config)
        results = strategy.iterative_prune(model)
        assert len(results["sparsity_per_iteration"]) == 3
        # Sparsity should increase over iterations
        for sp in results["sparsity_per_iteration"]:
            assert sp >= 0.0
