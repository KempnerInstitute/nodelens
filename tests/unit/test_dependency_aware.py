"""
Tests for pruning/dependency_aware.py: DependencyAwarePruning.
"""

import torch
import torch.nn as nn

from nodelens.pruning.dependency_aware import DependencyAwarePruning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleNet(nn.Module):
    """Conv -> Conv -> FC with known shapes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =========================================================================
# DependencyAwarePruning
# =========================================================================


class TestDependencyAwarePruning:
    def test_init_builds_graph(self):
        model = _SimpleNet()
        dap = DependencyAwarePruning(model)
        assert hasattr(dap, "model")

    def test_prune_returns_masks(self):
        model = _SimpleNet()
        dap = DependencyAwarePruning(model)
        scores = {
            "conv1": torch.rand(8),
            "conv2": torch.rand(16),
        }
        result = dap.prune(scores, amount=0.5, mode="low")
        assert "masks" in result
        assert len(result["masks"]) > 0

    def test_prune_per_layer_amounts(self):
        model = _SimpleNet()
        dap = DependencyAwarePruning(model)
        scores = {
            "conv1": torch.rand(8),
            "conv2": torch.rand(16),
        }
        per_layer = {"conv1": 0.25, "conv2": 0.5}
        result = dap.prune(scores, amount=0.5, mode="low", per_layer_amounts=per_layer)
        assert "masks" in result

    def test_prune_high_mode(self):
        model = _SimpleNet()
        dap = DependencyAwarePruning(model)
        scores = {"conv1": torch.rand(8)}
        result = dap.prune(scores, amount=0.5, mode="high")
        assert "masks" in result

    def test_empty_scores(self):
        model = _SimpleNet()
        dap = DependencyAwarePruning(model)
        result = dap.prune({}, amount=0.5)
        assert "masks" in result
