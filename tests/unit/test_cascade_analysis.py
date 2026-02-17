"""
Unit tests for cascade (ablation) analysis.

Tests validate:
- baseline: accuracy in [0,1], loss >= 0
- ablate: weights restored after ablation (no side effects)
- by_cluster: returns dict keyed by cluster type
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from alignment.analysis.cascade_analysis import CascadeAnalysis, CascadeResult


# ---------------------------------------------------------------------------
# Tiny model + dataset
# ---------------------------------------------------------------------------

class _TinyCNN(nn.Module):
    """Minimal 2-layer CNN for testing cascade analysis."""

    def __init__(self, n_classes: int = 5, n_channels: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, n_channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _synthetic_loader(n_samples: int = 40, n_classes: int = 5):
    """Create a simple list-of-tuples dataloader."""
    images = torch.randn(n_samples, 3, 8, 8)
    labels = torch.randint(0, n_classes, (n_samples,))
    # Return batches of 10
    batches = []
    bs = 10
    for i in range(0, n_samples, bs):
        batches.append((images[i:i+bs], labels[i:i+bs]))
    return batches


# ---------------------------------------------------------------------------
# Tests: baseline
# ---------------------------------------------------------------------------

class TestBaseline:

    def test_acc_in_unit_interval(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        result = ca.baseline()
        assert 0.0 <= result["acc"] <= 1.0

    def test_loss_nonnegative(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        result = ca.baseline()
        assert result["loss"] >= 0.0

    def test_baseline_cached(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        r1 = ca.baseline()
        r2 = ca.baseline()  # should use cached
        assert r1["acc"] == r2["acc"]


# ---------------------------------------------------------------------------
# Tests: ablate
# ---------------------------------------------------------------------------

class TestAblate:

    def test_returns_cascade_result(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        result = ca.ablate("conv1", [0, 1])
        assert isinstance(result, CascadeResult)
        assert result.n_removed == 2

    def test_weights_restored_after_ablation(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        # Store weights before
        w_before = model.conv1.weight.data.clone()
        ca.ablate("conv1", [0, 1, 2])
        # Weights should be identical after ablation
        torch.testing.assert_close(model.conv1.weight.data, w_before)

    def test_ablating_all_drops_accuracy(self):
        model = _TinyCNN(n_channels=8)
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        ca.baseline()
        result = ca.ablate("conv1", list(range(8)))  # zero all channels
        # Loss should increase (or acc should drop)
        assert result.loss_increase >= 0 or result.accuracy_drop >= 0

    def test_invalid_layer_returns_zero_damage(self):
        model = _TinyCNN()
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        result = ca.ablate("nonexistent_layer", [0])
        assert result.accuracy_drop == 0.0
        assert result.loss_increase == 0.0


# ---------------------------------------------------------------------------
# Tests: by_cluster
# ---------------------------------------------------------------------------

class TestByCluster:

    def test_returns_dict_keyed_by_type(self):
        model = _TinyCNN(n_channels=8)
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        types = {0: "critical", 1: "redundant", 2: "synergistic", 3: "background"}
        results = ca.by_cluster("conv1", labels, types, n_rm=2)
        assert isinstance(results, dict)
        for ctype in types.values():
            assert ctype in results
            assert isinstance(results[ctype], CascadeResult)
            assert results[ctype].cluster_type == ctype

    def test_n_removed_respects_limit(self):
        model = _TinyCNN(n_channels=8)
        loader = _synthetic_loader()
        ca = CascadeAnalysis(model, loader, device="cpu")
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        types = {0: "A", 1: "B"}
        results = ca.by_cluster("conv1", labels, types, n_rm=2)
        for r in results.values():
            assert r.n_removed <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
