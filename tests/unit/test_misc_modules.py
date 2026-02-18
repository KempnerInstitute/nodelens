"""
Tests for miscellaneous modules: dynamic_scoring, reporters, delta_alignment, pairwise_base.
"""

import json
import pytest
import torch
import pandas as pd

from alignment.analysis.dynamic_scoring import (
    DynamicScoreAggregator,
    compute_dynamic_importance,
)
from alignment.analysis.reporting.json_reporter import JSONReporter
from alignment.analysis.reporting.markdown import MarkdownReporter
from alignment.metrics.rayleigh.delta_alignment import (
    DeltaAlignment,
    NormalizedDeltaAlignment,
)
from alignment.metrics.pairwise_base import PairwiseMetric


# =========================================================================
# DynamicScoreAggregator
# =========================================================================


class TestDynamicScoreAggregator:
    def test_init_weights_normalized(self):
        agg = DynamicScoreAggregator(weight_final=0.4, weight_trend=0.2,
                                      weight_loss_corr=0.3, weight_stability=0.1)
        total = agg.weight_final + agg.weight_trend + agg.weight_loss_corr + agg.weight_stability
        assert total == pytest.approx(1.0)

    def test_compute_trend(self):
        agg = DynamicScoreAggregator()
        # Increasing scores over 5 steps for 4 neurons
        scores = torch.stack([torch.arange(4).float() * (t + 1) for t in range(5)])
        trends = agg.compute_trend(scores)
        assert trends.shape == (4,)
        # All should have positive trend (increasing over time)
        assert (trends[1:] > 0).all()

    def test_compute_stability(self):
        agg = DynamicScoreAggregator()
        # Constant scores = high stability
        scores = torch.ones(5, 4)
        stability = agg.compute_stability(scores)
        assert stability.shape == (4,)

    def test_compute_stability_variable(self):
        agg = DynamicScoreAggregator()
        scores = torch.randn(10, 4)
        # Make neuron 0 constant, neuron 3 variable
        scores[:, 0] = 1.0
        scores[:, 3] = torch.randn(10) * 10.0
        stability = agg.compute_stability(scores)
        # Neuron 0 (constant) should be most stable
        assert stability[0] >= stability[3]

    def test_compute_loss_correlation(self):
        agg = DynamicScoreAggregator()
        scores = torch.randn(10, 4)
        loss = list(range(10, 0, -1))  # Decreasing loss
        corr = agg.compute_loss_correlation(scores, loss)
        assert corr.shape == (4,)
        assert (corr >= 0).all()  # Absolute correlations

    def test_aggregate_full(self):
        agg = DynamicScoreAggregator()
        scores = torch.randn(10, 4).abs()
        loss = [1.0 - 0.1 * i for i in range(10)]
        result = agg.aggregate_full(scores, loss)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_aggregate_with_score_history(self):
        agg = DynamicScoreAggregator()
        # Build score_history structure with tensor_history
        score_history = {
            "history": {"conv1": {"rq": [0.5, 0.6, 0.7]}},
            "tensor_history": {
                "conv1": {"rq": [torch.rand(8) for _ in range(5)]},
            },
        }
        loss_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        result = agg.aggregate(score_history, loss_history, "conv1", "rq")
        assert result.shape == (8,)

    def test_aggregate_scalar_fallback(self):
        agg = DynamicScoreAggregator()
        score_history = {
            "history": {"conv1": {"rq": [0.5, 0.6, 0.7]}},
        }
        loss_history = [1.0, 0.8, 0.6]
        result = agg.aggregate(score_history, loss_history, "conv1", "rq")
        assert result.item() == pytest.approx(0.7)

    def test_aggregate_missing_layer_raises(self):
        agg = DynamicScoreAggregator()
        with pytest.raises(ValueError, match="No history"):
            agg.aggregate({"history": {}}, [], "conv1")

    def test_aggregate_missing_metric_raises(self):
        agg = DynamicScoreAggregator()
        with pytest.raises(ValueError, match="No rq"):
            agg.aggregate({"history": {"conv1": {}}}, [], "conv1", "rq")

    def test_align_loss_history_same_length(self):
        result = DynamicScoreAggregator._align_loss_history({}, [1.0, 2.0, 3.0], 3)
        assert result == [1.0, 2.0, 3.0]

    def test_align_loss_history_with_steps(self):
        result = DynamicScoreAggregator._align_loss_history(
            {"steps": [0, 5, 10]}, list(range(20)), 3
        )
        assert len(result) == 3

    def test_align_loss_history_fallback_uniform(self):
        result = DynamicScoreAggregator._align_loss_history({}, list(range(100)), 5)
        assert len(result) == 5

    def test_align_loss_history_empty_loss(self):
        result = DynamicScoreAggregator._align_loss_history({}, [], 3)
        assert result == [0.0, 0.0, 0.0]

    def test_align_loss_history_zero_steps(self):
        result = DynamicScoreAggregator._align_loss_history({}, [1.0], 0)
        assert result == []


class TestComputeDynamicImportance:
    def test_convenience_function(self):
        score_history = {
            "history": {"fc1": {"rq": [0.5, 0.6]}},
            "tensor_history": {"fc1": {"rq": [torch.rand(4), torch.rand(4)]}},
        }
        loss_history = [1.0, 0.5]
        result = compute_dynamic_importance(score_history, loss_history, "fc1")
        assert result.shape == (4,)


# =========================================================================
# JSONReporter
# =========================================================================


class TestJSONReporter:
    def test_init(self):
        reporter = JSONReporter(title="Test Report")
        assert reporter.title == "Test Report"
        assert "title" in reporter.data

    def test_add_section_dict(self):
        reporter = JSONReporter()
        reporter.add_section("metrics", {"rq": 0.5, "red": 0.3})
        assert "metrics" in reporter.data["sections"]

    def test_add_section_dataframe(self):
        reporter = JSONReporter()
        df = pd.DataFrame({"layer": ["conv1", "fc1"], "rq": [0.5, 0.7]})
        reporter.add_section("layers", df)
        assert isinstance(reporter.data["sections"]["layers"], list)

    def test_generate(self, tmp_path):
        reporter = JSONReporter(title="Test")
        reporter.add_section("info", {"key": "value"})
        output = tmp_path / "report.json"
        reporter.generate(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["title"] == "Test"
        assert "info" in data["sections"]


# =========================================================================
# MarkdownReporter
# =========================================================================


class TestMarkdownReporter:
    def test_init(self):
        reporter = MarkdownReporter(title="MD Report")
        assert reporter.title == "MD Report"
        assert len(reporter.sections) == 0

    def test_add_section(self):
        reporter = MarkdownReporter()
        reporter.add_section("Summary", "This is a summary.")
        assert len(reporter.sections) == 1

    def test_add_table(self):
        pytest.importorskip("tabulate")
        reporter = MarkdownReporter()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        reporter.add_table("Data", df)
        assert len(reporter.sections) == 1

    def test_generate(self, tmp_path):
        reporter = MarkdownReporter(title="Test Report")
        reporter.add_section("Intro", "Hello world.")
        output = tmp_path / "report.md"
        reporter.generate(output)
        assert output.exists()
        content = output.read_text()
        assert "# Test Report" in content
        assert "## Intro" in content
        assert "Hello world." in content


# =========================================================================
# DeltaAlignment
# =========================================================================


class TestDeltaAlignment:
    def test_basic(self):
        metric = DeltaAlignment()
        inputs = torch.randn(20, 8)
        weights = torch.randn(4, 8)
        initial_weights = torch.zeros(4, 8)
        scores = metric.compute(
            inputs=inputs, weights=weights, initial_weights=initial_weights
        )
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_with_stored_initial_weights(self):
        metric = DeltaAlignment()
        metric.set_initial_weights("conv1", torch.zeros(4, 8))
        scores = metric.compute(
            inputs=torch.randn(20, 8),
            weights=torch.randn(4, 8),
            layer_name="conv1",
        )
        assert scores.shape == (4,)

    def test_no_initial_weights_uses_zeros(self):
        metric = DeltaAlignment()
        scores = metric.compute(
            inputs=torch.randn(20, 8),
            weights=torch.randn(4, 8),
        )
        assert scores.shape == (4,)

    def test_requires_inputs_and_weights(self):
        metric = DeltaAlignment()
        with pytest.raises(ValueError, match="requires both"):
            metric.compute(inputs=None, weights=torch.randn(4, 8))

    def test_shape_mismatch_handled(self):
        metric = DeltaAlignment()
        scores = metric.compute(
            inputs=torch.randn(20, 8),
            weights=torch.randn(4, 8),
            initial_weights=torch.randn(4, 8).flatten().reshape(4, 8),
        )
        assert scores.shape == (4,)


class TestNormalizedDeltaAlignment:
    def test_basic(self):
        metric = NormalizedDeltaAlignment()
        inputs = torch.randn(20, 8)
        weights = torch.randn(4, 8)
        initial_weights = torch.zeros(4, 8)
        scores = metric.compute(
            inputs=inputs, weights=weights, initial_weights=initial_weights
        )
        assert scores.shape == (4,)
        assert torch.isfinite(scores).all()

    def test_no_change_gives_zero(self):
        metric = NormalizedDeltaAlignment()
        w = torch.randn(4, 8)
        scores = metric.compute(
            inputs=torch.randn(20, 8),
            weights=w,
            initial_weights=w.clone(),
        )
        # When weights == initial, diff is zero, so normalized should be 0
        assert torch.allclose(scores, torch.zeros(4), atol=1e-6)


# =========================================================================
# PairwiseMetric (abstract - test via concrete subclass)
# =========================================================================


class _CorrelationMetric(PairwiseMetric):
    """Concrete pairwise metric for testing: pairwise correlation."""

    @property
    def requires_inputs(self):
        return False

    @property
    def requires_weights(self):
        return False

    @property
    def requires_outputs(self):
        return True

    def compute_pairwise(self, inputs=None, weights=None, outputs=None, **kwargs):
        if outputs is None:
            raise ValueError("Need outputs")
        # Compute correlation matrix
        outputs_centered = outputs - outputs.mean(dim=0)
        std = outputs.std(dim=0, keepdim=True)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        z = outputs_centered / std
        corr = (z.T @ z) / max(1, outputs.shape[0] - 1)
        return corr.abs()


class TestPairwiseMetric:
    def test_mean_aggregation(self):
        metric = _CorrelationMetric(aggregation="mean")
        outputs = torch.randn(20, 4)
        scores = metric.compute(outputs=outputs)
        assert scores.shape == (4,)

    def test_median_aggregation(self):
        metric = _CorrelationMetric(aggregation="median")
        outputs = torch.randn(20, 4)
        scores = metric.compute(outputs=outputs)
        assert scores.shape == (4,)

    def test_max_aggregation(self):
        metric = _CorrelationMetric(aggregation="max")
        outputs = torch.randn(20, 4)
        scores = metric.compute(outputs=outputs)
        assert scores.shape == (4,)

    def test_sum_aggregation(self):
        metric = _CorrelationMetric(aggregation="sum")
        outputs = torch.randn(20, 4)
        scores = metric.compute(outputs=outputs)
        assert scores.shape == (4,)

    def test_unknown_aggregation_raises(self):
        metric = _CorrelationMetric(aggregation="bogus")
        with pytest.raises(ValueError, match="Unknown aggregation"):
            metric.compute(outputs=torch.randn(20, 4))

    def test_include_diagonal(self):
        metric = _CorrelationMetric(aggregation="mean", exclude_diagonal=False)
        outputs = torch.randn(20, 4)
        scores = metric.compute(outputs=outputs)
        assert scores.shape == (4,)

    def test_include_diagonal_all_aggregations(self):
        outputs = torch.randn(20, 4)
        for agg in ["mean", "median", "max", "sum"]:
            metric = _CorrelationMetric(aggregation=agg, exclude_diagonal=False)
            scores = metric.compute(outputs=outputs)
            assert scores.shape == (4,)

    def test_include_diagonal_unknown_raises(self):
        metric = _CorrelationMetric(aggregation="bogus", exclude_diagonal=False)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            metric.compute(outputs=torch.randn(20, 4))

    def test_return_matrix(self):
        metric = _CorrelationMetric()
        outputs = torch.randn(20, 4)
        matrix = metric.compute(outputs=outputs, return_matrix=True)
        assert matrix.shape == (4, 4)

    def test_compute_for_subset(self):
        metric = _CorrelationMetric()
        outputs = torch.randn(20, 8)
        indices = torch.tensor([0, 3, 5])
        subset_scores = metric.compute_for_subset(indices, outputs=outputs)
        assert subset_scores.shape == (3,)
