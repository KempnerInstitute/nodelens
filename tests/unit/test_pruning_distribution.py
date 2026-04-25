"""
Tests for pruning/distribution.py: PruningDistributionManager and DistributionStrategy.
"""

import pytest
import torch
import torch.nn as nn

from nodelens.pruning.distribution import DistributionStrategy, PruningDistributionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def _make_scores(layer_names, n_per_layer=8):
    """Create random scores for each layer."""
    return {name: torch.rand(n_per_layer) for name in layer_names}


# =========================================================================
# DistributionStrategy enum
# =========================================================================


class TestDistributionStrategy:
    def test_uniform_value(self):
        assert DistributionStrategy.UNIFORM.value == "uniform"

    def test_global_threshold_value(self):
        assert DistributionStrategy.GLOBAL_THRESHOLD.value == "global_threshold"


# =========================================================================
# PruningDistributionManager
# =========================================================================


class TestPruningDistributionManager:
    def test_uniform_distribution(self):
        mgr = PruningDistributionManager(strategy="uniform", target_sparsity=0.5)
        amounts = mgr.compute_distribution(
            _TinyModel(),
            ["layer1", "layer2"],
        )
        assert amounts["layer1"] == pytest.approx(0.5)
        assert amounts["layer2"] == pytest.approx(0.5)

    def test_uniform_clamps_to_max(self):
        mgr = PruningDistributionManager(strategy="uniform", target_sparsity=0.99, max_amount=0.8)
        amounts = mgr.compute_distribution(_TinyModel(), ["layer1"])
        assert amounts["layer1"] == pytest.approx(0.8)

    def test_uniform_clamps_to_min(self):
        mgr = PruningDistributionManager(strategy="uniform", target_sparsity=0.01, min_amount=0.1)
        amounts = mgr.compute_distribution(_TinyModel(), ["layer1"])
        assert amounts["layer1"] == pytest.approx(0.1)

    def test_global_threshold_distribution(self):
        mgr = PruningDistributionManager(strategy="global_threshold", target_sparsity=0.5)
        model = _TinyModel()
        scores = _make_scores(["layer1", "layer2"], n_per_layer=8)
        amounts = mgr.compute_distribution(model, ["layer1", "layer2"], layer_scores=scores)
        assert "layer1" in amounts
        assert "layer2" in amounts
        for v in amounts.values():
            assert 0.0 <= v <= 1.0

    def test_global_threshold_no_scores_raises(self):
        mgr = PruningDistributionManager(strategy="global_threshold", target_sparsity=0.5)
        with pytest.raises(ValueError, match="layer_scores"):
            mgr.compute_distribution(_TinyModel(), ["layer1"])

    def test_size_proportional_distribution(self):
        mgr = PruningDistributionManager(strategy="size_proportional", target_sparsity=0.5)
        model = _TinyModel()
        amounts = mgr.compute_distribution(model, ["layer1", "layer2"])
        assert "layer1" in amounts
        assert "layer2" in amounts
        # Larger layer should potentially get different amount
        for v in amounts.values():
            assert 0.0 <= v <= 1.0

    def test_importance_weighted_requires_scores(self):
        mgr = PruningDistributionManager(strategy="importance_weighted", target_sparsity=0.5)
        with pytest.raises(ValueError, match="layer_scores"):
            mgr.compute_distribution(_TinyModel(), ["layer1"])

    def test_importance_weighted_distribution(self):
        mgr = PruningDistributionManager(strategy="importance_weighted", target_sparsity=0.5)
        model = _TinyModel()
        scores = _make_scores(["layer1", "layer2"])
        amounts = mgr.compute_distribution(model, ["layer1", "layer2"], layer_scores=scores)
        for v in amounts.values():
            assert 0.0 <= v <= 1.0

    def test_unknown_strategy_falls_back(self):
        """Unknown strategy string should fall back to UNIFORM."""
        mgr = PruningDistributionManager(strategy="totally_bogus", target_sparsity=0.3)
        assert mgr.strategy == DistributionStrategy.UNIFORM
        amounts = mgr.compute_distribution(_TinyModel(), ["layer1"])
        assert amounts["layer1"] == pytest.approx(0.3)

    def test_global_knapsack_distribution(self):
        mgr = PruningDistributionManager(strategy="global_knapsack", target_sparsity=0.5)
        model = _TinyModel()
        scores = _make_scores(["layer1", "layer2"])
        amounts = mgr.compute_distribution(model, ["layer1", "layer2"], layer_scores=scores)
        for v in amounts.values():
            assert 0.0 <= v <= 1.0
