"""
Unit tests for MaskOperations.
"""

import pytest
import torch

from alignment.services.mask_ops import MaskOperations


class TestMaskOperations:
    """Tests for MaskOperations class."""

    def test_create_structured_mask_low(self):
        """Test structured mask creation with low mode."""
        scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])
        mask = MaskOperations.create_structured_mask(scores, amount=0.4, mode="low")

        # Should prune 2 out of 5 (40%)
        assert mask.sum() == 3

        # Should keep highest scores: [0.5, 0.9, 0.7]
        # Should prune: [0.1, 0.3]
        assert not mask[0]  # 0.1
        assert mask[1]  # 0.5
        assert not mask[2]  # 0.3
        assert mask[3]  # 0.9
        assert mask[4]  # 0.7

    def test_create_structured_mask_high(self):
        """Test structured mask creation with high mode."""
        scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])
        mask = MaskOperations.create_structured_mask(scores, amount=0.4, mode="high")

        # Should prune highest 40% (2 neurons)
        assert mask.sum() == 3

        # Should keep lowest scores
        assert not mask[3]  # 0.9 (highest)
        assert not mask[4]  # 0.7

    def test_create_structured_mask_random(self):
        """Test random structured mask."""
        scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.7])

        # Set seed for reproducibility
        torch.manual_seed(42)
        mask = MaskOperations.create_structured_mask(scores, amount=0.4, mode="random")

        # Should keep 60%
        assert mask.sum() == 3

    def test_create_structured_mask_min_keep(self):
        """Test min_keep parameter."""
        scores = torch.tensor([0.1, 0.5, 0.3])

        # Try to prune 100%, but min_keep=1
        mask = MaskOperations.create_structured_mask(scores, amount=1.0, mode="low", min_keep=1)

        assert mask.sum() >= 1

    def test_create_unstructured_mask(self):
        """Test unstructured (weight-level) mask creation."""
        scores = torch.randn(10, 20)  # 200 weights
        mask = MaskOperations.create_unstructured_mask(scores, amount=0.5, mode="low")

        # Should keep 50%
        assert mask.sum() == 100

        # Should have same shape
        assert mask.shape == scores.shape

    def test_expand_neuron_mask_to_weights_linear(self):
        """Test expanding neuron mask to Linear layer weights."""
        neuron_mask = torch.tensor([True, False, True, False, True])  # 3 kept
        weight_shape = (5, 10)  # [out_neurons, in_features]

        expanded = MaskOperations.expand_neuron_mask_to_weights(neuron_mask, weight_shape, dim=0)

        assert expanded.shape == weight_shape

        # Row 1 should be all True (neuron kept)
        assert expanded[0].all()
        # Row 1 should be all False (neuron pruned)
        assert not expanded[1].any()
        assert expanded[2].all()
        assert not expanded[3].any()
        assert expanded[4].all()

    def test_expand_neuron_mask_to_weights_conv(self):
        """Test expanding neuron mask to Conv2d weights."""
        neuron_mask = torch.tensor([True, False, True])  # 2 kept
        weight_shape = (3, 16, 3, 3)  # [out_channels, in_channels, k, k]

        expanded = MaskOperations.expand_neuron_mask_to_weights(neuron_mask, weight_shape, dim=0)

        assert expanded.shape == weight_shape

        # Channel 0 should be all True
        assert expanded[0].all()
        # Channel 1 should be all False
        assert not expanded[1].any()
        # Channel 2 should be all True
        assert expanded[2].all()

    def test_apply_mask_multiply(self):
        """Test applying mask via multiplication."""
        weights = torch.randn(5, 10)
        mask = torch.tensor([True, False, True, False, True]).unsqueeze(1).expand(5, 10)

        masked_weights = MaskOperations.apply_mask_to_weights(weights, mask, mode="multiply")

        # Row 1 and 3 should be zero
        assert torch.allclose(masked_weights[1], torch.zeros(10))
        assert torch.allclose(masked_weights[3], torch.zeros(10))

        # Row 0, 2, 4 should be unchanged
        assert torch.allclose(masked_weights[0], weights[0])

    def test_apply_mask_zero(self):
        """Test applying mask via zeroing."""
        weights = torch.randn(5, 10)
        mask = torch.ones_like(weights, dtype=torch.bool)
        mask[1, :] = False

        masked_weights = MaskOperations.apply_mask_to_weights(weights, mask, mode="zero")

        # Row 1 should be zero
        assert torch.allclose(masked_weights[1], torch.zeros(10))

    def test_get_mask_statistics(self):
        """Test mask statistics computation."""
        mask = torch.tensor([True, False, True, False, True, True, False])

        stats = MaskOperations.get_mask_statistics(mask)

        assert stats["total_elements"] == 7
        assert stats["kept_elements"] == 4
        assert stats["pruned_elements"] == 3
        assert abs(stats["sparsity"] - 3 / 7) < 1e-6
        assert abs(stats["density"] - 4 / 7) < 1e-6

    def test_combine_masks_and(self):
        """Test combining masks with AND operation."""
        mask1 = torch.tensor([True, True, False, True])
        mask2 = torch.tensor([True, False, True, True])

        combined = MaskOperations.combine_masks([mask1, mask2], operation="and")

        # Only elements True in both
        expected = torch.tensor([True, False, False, True])
        assert torch.equal(combined, expected)

    def test_combine_masks_or(self):
        """Test combining masks with OR operation."""
        mask1 = torch.tensor([True, True, False, True])
        mask2 = torch.tensor([True, False, True, True])

        combined = MaskOperations.combine_masks([mask1, mask2], operation="or")

        # Elements True in either
        expected = torch.tensor([True, True, True, True])
        assert torch.equal(combined, expected)

    def test_global_threshold_mask(self):
        """Test global threshold masking across layers."""
        layer_scores = {"layer1": torch.tensor([0.1, 0.5, 0.3]), "layer2": torch.tensor([0.9, 0.2, 0.8, 0.4])}

        # Prune 50% globally (total 7 neurons, keep 3-4)
        masks = MaskOperations.global_threshold_mask(layer_scores, global_amount=0.5, mode="low")

        # Check we have masks for both layers
        assert "layer1" in masks
        assert "layer2" in masks

        # Total kept should be ~3-4
        total_kept = masks["layer1"].sum() + masks["layer2"].sum()
        assert total_kept in [3, 4]

    def test_empty_mask_edge_case(self):
        """Test edge case with zero pruning amount."""
        scores = torch.randn(10)
        mask = MaskOperations.create_structured_mask(scores, amount=0.0, mode="low")

        # Should keep all
        assert mask.all()

    def test_full_prune_with_min_keep(self):
        """Test that min_keep is respected."""
        scores = torch.randn(10)
        mask = MaskOperations.create_structured_mask(scores, amount=1.0, mode="low", min_keep=2)  # Try to prune all

        # Should keep at least 2
        assert mask.sum() >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
