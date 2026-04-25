"""
Unit tests for dependency-aware structured pruning.
"""

import torch
import torch.nn as nn

from nodelens.pruning.dependency_aware import DependencyAwarePruning, DependencyGraph


class _TinyCNN(nn.Module):
    """
    Tiny CNN with simple conv dependencies:
    conv1 -> conv2
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(4, 5, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class _ResidualLikeBlock(nn.Module):
    """Small module with residual-compatible convolutions for graph metadata tests."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class TestDependencyAwarePruning:
    """Tests for DependencyAwarePruning."""

    def test_propagates_channel_masks_between_convs(self):
        """
        Pruning conv1 output channels should propagate to conv2 input channels,
        and the resulting weight masks should be shape-compatible.
        """
        model = _TinyCNN()
        pruner = DependencyAwarePruning(model)

        # Create simple importance scores: keep first two channels in conv1, all in conv2
        layer_scores = {
            "conv1": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "conv2": torch.ones(5),
        }

        result = pruner.prune(layer_scores, amount=0.5, dry_run=True)

        masks = result["masks"]
        assert "conv1" in masks
        assert "conv2" in masks

        conv1_masks = masks["conv1"]
        conv2_masks = masks["conv2"]

        # Output mask of conv1 should be boolean of length 4
        assert conv1_masks["output_mask"].dtype == torch.bool
        assert conv1_masks["output_mask"].shape[0] == model.conv1.out_channels

        # Input mask of conv2 should match conv1's output mask
        assert torch.equal(conv2_masks["input_mask"], conv1_masks["output_mask"])

        # Weight masks should have same shape as weights and be boolean
        assert conv1_masks["weight_mask"].shape == model.conv1.weight.shape
        assert conv1_masks["weight_mask"].dtype == torch.bool
        assert conv2_masks["weight_mask"].shape == model.conv2.weight.shape
        assert conv2_masks["weight_mask"].dtype == torch.bool

    def test_invalid_plan_raises_error_when_validation_fails(self, monkeypatch):
        """If validation reports invalid plan, prune should raise ValueError."""
        model = _TinyCNN()
        pruner = DependencyAwarePruning(model)

        # Create dummy scores; we'll force validation to fail
        layer_scores = {"conv1": torch.ones(model.conv1.out_channels)}

        def fake_validate(self, masks):
            return {"valid": False, "errors": ["test error"]}

        monkeypatch.setattr(type(pruner), "_validate_pruning_plan", fake_validate)

        try:
            _ = pruner.prune(layer_scores, amount=0.5, dry_run=True)
        except ValueError as e:
            assert "Invalid pruning plan" in str(e)
        else:
            assert False, "Expected ValueError due to invalid pruning plan"

    def test_marks_residual_compatible_conv_candidates(self):
        """DependencyGraph records same-parent, channel-preserving conv candidates."""
        graph = DependencyGraph(_ResidualLikeBlock())

        first = graph.graph["block.0"]
        second = graph.graph["block.2"]

        assert first.skip_connection_with == ["block.2"]
        assert second.skip_connection_with == ["block.0"]
