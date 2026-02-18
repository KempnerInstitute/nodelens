"""
Tests for services/mask_ops.py: MaskOperations static methods.
"""

import pytest
import torch

from alignment.services.mask_ops import MaskOperations


class TestCreateStructuredMask:
    def test_low_mode(self):
        scores = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        mask = MaskOperations.create_structured_mask(scores, amount=0.4, mode="low")
        # 40% of 5 = 2 pruned -> 3 kept
        assert mask.sum().item() == 3

    def test_high_mode(self):
        scores = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        mask = MaskOperations.create_structured_mask(scores, amount=0.4, mode="high")
        assert mask.sum().item() == 3

    def test_random_mode(self):
        scores = torch.rand(10)
        mask = MaskOperations.create_structured_mask(scores, amount=0.5, mode="random")
        assert mask.sum().item() == 5

    def test_min_keep(self):
        scores = torch.rand(4)
        mask = MaskOperations.create_structured_mask(scores, amount=0.99, min_keep=2)
        assert mask.sum().item() >= 2

    def test_zero_amount(self):
        scores = torch.rand(5)
        mask = MaskOperations.create_structured_mask(scores, amount=0.0)
        assert mask.all()

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            MaskOperations.create_structured_mask(torch.rand(4), amount=0.5, mode="bogus")


class TestCreateUnstructuredMask:
    def test_correct_sparsity(self):
        scores = torch.rand(4, 4)
        mask = MaskOperations.create_unstructured_mask(scores, amount=0.5)
        expected = int(0.5 * 16)
        assert (mask == 0).sum().item() == expected

    def test_zero_amount(self):
        scores = torch.rand(3, 3)
        mask = MaskOperations.create_unstructured_mask(scores, amount=0.0)
        assert mask.all()

    def test_shape_preserved(self):
        scores = torch.rand(2, 3, 4)
        mask = MaskOperations.create_unstructured_mask(scores, amount=0.5)
        assert mask.shape == scores.shape


class TestExpandNeuronMaskToWeights:
    def test_linear_dim0(self):
        neuron_mask = torch.tensor([True, False, True, True])
        expanded = MaskOperations.expand_neuron_mask_to_weights(neuron_mask, (4, 8), dim=0)
        assert expanded.shape == (4, 8)
        assert expanded[1].sum().item() == 0  # row 1 pruned
        assert expanded[0].all()  # row 0 kept

    def test_conv2d_dim0(self):
        neuron_mask = torch.tensor([True, False, True])
        expanded = MaskOperations.expand_neuron_mask_to_weights(neuron_mask, (3, 4, 3, 3), dim=0)
        assert expanded.shape == (3, 4, 3, 3)
        assert expanded[1].sum().item() == 0

    def test_linear_dim1(self):
        neuron_mask = torch.tensor([True, False, True])
        expanded = MaskOperations.expand_neuron_mask_to_weights(neuron_mask, (4, 3), dim=1)
        assert expanded[:, 1].sum().item() == 0


class TestApplyMaskToWeights:
    def test_multiply_mode(self):
        w = torch.ones(3, 3)
        m = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=torch.bool)
        result = MaskOperations.apply_mask_to_weights(w, m, mode="multiply")
        assert result[0, 1].item() == 0.0

    def test_zero_mode(self):
        w = torch.randn(3, 3)
        m = torch.ones(3, 3, dtype=torch.bool)
        m[0, 0] = False
        result = MaskOperations.apply_mask_to_weights(w, m, mode="zero")
        assert result[0, 0].item() == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            MaskOperations.apply_mask_to_weights(torch.zeros(3, 3), torch.zeros(2, 2, dtype=torch.bool))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            MaskOperations.apply_mask_to_weights(torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.bool), mode="bogus")


class TestGetMaskStatistics:
    def test_basic_stats(self):
        mask = torch.tensor([1, 0, 1, 0, 1])
        stats = MaskOperations.get_mask_statistics(mask)
        assert stats["total_elements"] == 5
        assert stats["kept_elements"] == 3
        assert stats["pruned_elements"] == 2
        assert stats["sparsity"] == pytest.approx(0.4)
        assert stats["density"] == pytest.approx(0.6)


class TestCombineMasks:
    def test_and(self):
        m1 = torch.tensor([True, True, False, False])
        m2 = torch.tensor([True, False, True, False])
        result = MaskOperations.combine_masks([m1, m2], operation="and")
        assert result.tolist() == [True, False, False, False]

    def test_or(self):
        m1 = torch.tensor([True, True, False, False])
        m2 = torch.tensor([True, False, True, False])
        result = MaskOperations.combine_masks([m1, m2], operation="or")
        assert result.tolist() == [True, True, True, False]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No masks"):
            MaskOperations.combine_masks([])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            MaskOperations.combine_masks([torch.ones(3, dtype=torch.bool), torch.ones(4, dtype=torch.bool)])


class TestGlobalThresholdMask:
    def test_low_mode(self):
        scores = {
            "layer1": torch.tensor([1.0, 5.0, 3.0]),
            "layer2": torch.tensor([2.0, 4.0]),
        }
        masks = MaskOperations.global_threshold_mask(scores, global_amount=0.4, mode="low")
        assert "layer1" in masks
        assert "layer2" in masks
        # 40% of 5 total = 2 pruned
        total_pruned = sum((~m).sum().item() for m in masks.values())
        assert total_pruned == 2

    def test_high_mode(self):
        scores = {
            "layer1": torch.tensor([1.0, 5.0]),
            "layer2": torch.tensor([3.0]),
        }
        masks = MaskOperations.global_threshold_mask(scores, global_amount=0.33, mode="high")
        total_kept = sum(m.sum().item() for m in masks.values())
        assert total_kept == 2  # keep 2 of 3

    def test_random_mode(self):
        scores = {"a": torch.rand(10)}
        masks = MaskOperations.global_threshold_mask(scores, global_amount=0.5, mode="random")
        assert masks["a"].sum().item() == 5
