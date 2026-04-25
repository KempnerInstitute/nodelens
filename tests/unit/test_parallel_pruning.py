"""
Tests for pruning/strategies/parallel.py: parallel pruning strategies.
"""

import pytest
import torch
import torch.nn as nn

from nodelens.pruning.strategies.parallel import AsyncParallelPruning, ParallelModePruning, ParallelPruningResult, TensorizedPruning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# =========================================================================
# ParallelPruningResult
# =========================================================================


class TestParallelPruningResult:
    def test_basic(self):
        result = ParallelPruningResult(
            masks={"low": torch.ones(4, 8)},
            sparsities={"low": 0.0},
        )
        assert "low" in result.masks
        assert result.combined_mask is None


# =========================================================================
# ParallelModePruning
# =========================================================================


class TestParallelModePruning:
    def test_init_default(self):
        pmp = ParallelModePruning()
        assert pmp.modes == ["low", "high", "random"]
        assert pmp.base_strategy == "magnitude"

    def test_init_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown base strategy"):
            ParallelModePruning(base_strategy="bogus")

    def test_prune_parallel(self):
        model = _TinyModel()
        pmp = ParallelModePruning(modes=["low", "high"])
        result = pmp.prune_parallel(model.fc1, amount=0.5)
        assert isinstance(result, ParallelPruningResult)
        assert "low" in result.masks
        assert "high" in result.masks
        assert result.importance_scores is not None

    def test_prune_parallel_with_random(self):
        model = _TinyModel()
        pmp = ParallelModePruning(modes=["low", "random"])
        result = pmp.prune_parallel(model.fc1, amount=0.5)
        assert "low" in result.masks
        assert "random" in result.masks

    def test_sparsities_computed(self):
        model = _TinyModel()
        pmp = ParallelModePruning(modes=["low"])
        result = pmp.prune_parallel(model.fc1, amount=0.5)
        assert "low" in result.sparsities
        assert 0.0 <= result.sparsities["low"] <= 1.0

    def test_combine_masks_intersection(self):
        pmp = ParallelModePruning()
        masks = {
            "a": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "b": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        combined = pmp.combine_masks(masks, method="intersection")
        assert combined.tolist() == [1.0, 0.0, 0.0, 0.0]

    def test_combine_masks_union(self):
        pmp = ParallelModePruning()
        masks = {
            "a": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "b": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
        combined = pmp.combine_masks(masks, method="union")
        assert combined.tolist() == [1.0, 1.0, 1.0, 0.0]

    def test_combine_masks_majority(self):
        pmp = ParallelModePruning()
        masks = {
            "a": torch.tensor([1.0, 1.0, 0.0]),
            "b": torch.tensor([1.0, 0.0, 0.0]),
            "c": torch.tensor([0.0, 1.0, 0.0]),
        }
        combined = pmp.combine_masks(masks, method="majority")
        assert combined[0] == 1.0  # 2/3 keep
        assert combined[2] == 0.0  # 0/3 keep

    def test_combine_masks_unknown_raises(self):
        pmp = ParallelModePruning()
        with pytest.raises(ValueError, match="Unknown combination"):
            pmp.combine_masks({"a": torch.ones(4)}, method="bogus")


# =========================================================================
# TensorizedPruning
# =========================================================================


class TestTensorizedPruning:
    def test_compute_importance_scores(self):
        model = _TinyModel()
        tp = TensorizedPruning()
        scores = tp.compute_importance_scores(model.fc1)
        assert scores.shape == model.fc1.weight.shape

    def test_no_weights_raises(self):
        tp = TensorizedPruning()
        with pytest.raises(ValueError, match="does not have weights"):
            tp.compute_importance_scores(nn.ReLU())

    def test_compute_pruning_tensor_shape(self):
        model = _TinyModel()
        tp = TensorizedPruning()
        tensor = tp.compute_pruning_tensor(model.fc1, modes=["low", "high"], amounts=[0.3, 0.5])
        assert tensor.shape == (2, 2, 16, 8)  # [modes, amounts, out, in]

    def test_pruning_tensor_values(self):
        model = _TinyModel()
        tp = TensorizedPruning()
        tensor = tp.compute_pruning_tensor(model.fc1, modes=["low"], amounts=[0.0, 0.5])
        # amount=0.0 -> all ones
        assert tensor[0, 0].sum() == 16 * 8
        # amount=0.5 -> about half pruned
        half = int(0.5 * 16 * 8)
        pruned = (tensor[0, 1] == 0).sum().item()
        assert pruned == half

    def test_pruning_tensor_random(self):
        model = _TinyModel()
        tp = TensorizedPruning()
        tensor = tp.compute_pruning_tensor(model.fc1, modes=["random"], amounts=[0.5])
        assert tensor.shape[0] == 1

    def test_analyze_pruning_patterns(self):
        model = _TinyModel()
        tp = TensorizedPruning()
        tensor = tp.compute_pruning_tensor(model.fc1, modes=["low", "high"], amounts=[0.3, 0.5, 0.7])
        analysis = tp.analyze_pruning_patterns(tensor)
        assert "sparsity_progression" in analysis
        assert "mode_overlap" in analysis
        assert "pruning_variance" in analysis


# =========================================================================
# AsyncParallelPruning
# =========================================================================


class TestAsyncParallelPruning:
    def test_prune_modules_parallel(self):
        model = _TinyModel()
        app = AsyncParallelPruning()
        results = app.prune_modules_parallel(
            [model.fc1, model.fc2],
            amounts=0.5,
            modes=["low"],
        )
        assert len(results) == 2
        for result in results:
            assert "low" in result

    def test_prune_modules_per_layer_amounts(self):
        model = _TinyModel()
        app = AsyncParallelPruning()
        results = app.prune_modules_parallel(
            [model.fc1, model.fc2],
            amounts=[0.3, 0.7],
            modes=["low"],
        )
        assert len(results) == 2

    def test_prune_modules_with_random(self):
        model = _TinyModel()
        app = AsyncParallelPruning()
        results = app.prune_modules_parallel(
            [model.fc1],
            amounts=0.5,
            modes=["low", "random"],
        )
        assert "low" in results[0]
        assert "random" in results[0]

    def test_no_weights_raises(self):
        app = AsyncParallelPruning()
        with pytest.raises(ValueError, match="does not have weights"):
            app.compute_importance_scores(nn.ReLU())
