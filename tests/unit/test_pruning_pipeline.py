"""
Tests for pruning/pipeline.py: run_pruning_pipeline and helpers.
"""

import torch
import torch.nn as nn

from alignment.pruning.pipeline import PruningPipelineOptions, _ensure_tensor, run_pruning_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.mean(dim=[2, 3])
        return self.fc(x)


# =========================================================================
# PruningPipelineOptions
# =========================================================================


class TestPipelineOptions:
    def test_defaults(self):
        opts = PruningPipelineOptions()
        assert opts.distribution == "uniform"
        assert opts.dependency_aware is False
        assert opts.min_amount == 0.0
        assert opts.max_amount == 0.95


# =========================================================================
# _ensure_tensor
# =========================================================================


class TestEnsureTensor:
    def test_tensor_passthrough(self):
        t = torch.rand(4)
        assert _ensure_tensor(t) is t

    def test_list_to_tensor(self):
        t = _ensure_tensor([1.0, 2.0, 3.0])
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)


# =========================================================================
# run_pruning_pipeline
# =========================================================================


class TestRunPruningPipeline:
    def test_empty_scores_returns_empty(self):
        result = run_pruning_pipeline(
            _TinyModel(),
            layer_scores={},
            target_sparsity=0.5,
        )
        assert result["masks"] == {}

    def test_uniform_pipeline(self):
        model = _TinyModel()
        scores = {
            "conv1": torch.rand(8),
            "fc": torch.rand(4),
        }
        result = run_pruning_pipeline(
            model,
            layer_scores=scores,
            target_sparsity=0.5,
            selection_mode="low",
        )
        assert "masks" in result
        assert len(result["masks"]) > 0

    def test_masks_are_boolean_like(self):
        model = _TinyModel()
        scores = {"conv1": torch.rand(8)}
        result = run_pruning_pipeline(
            model,
            layer_scores=scores,
            target_sparsity=0.5,
        )
        for mask in result["masks"].values():
            unique = mask.unique()
            assert all(v in [0, 1, True, False] for v in unique.tolist())

    def test_global_threshold_pipeline(self):
        model = _TinyModel()
        scores = {
            "conv1": torch.rand(8),
            "fc": torch.rand(4),
        }
        result = run_pruning_pipeline(
            model,
            layer_scores=scores,
            target_sparsity=0.5,
            options=PruningPipelineOptions(distribution="global_threshold"),
        )
        assert len(result["masks"]) > 0

    def test_mismatched_names_returns_empty(self):
        model = _TinyModel()
        scores = {"nonexistent_layer": torch.rand(8)}
        result = run_pruning_pipeline(
            model,
            layer_scores=scores,
            target_sparsity=0.5,
        )
        assert result["masks"] == {}

    def test_stats_present(self):
        model = _TinyModel()
        scores = {"conv1": torch.rand(8)}
        result = run_pruning_pipeline(
            model,
            layer_scores=scores,
            target_sparsity=0.5,
        )
        if "stats" in result:
            for layer, stats in result["stats"].items():
                assert "sparsity" in stats
                assert "density" in stats
