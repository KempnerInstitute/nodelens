"""
Tests for aggregation modules: MetricAggregator, LayerAggregator, ResultAggregator.
"""

import json

import pytest

from alignment.analysis.aggregation.layers import LayerAggregator
from alignment.analysis.aggregation.metrics import MetricAggregator
from alignment.analysis.aggregation.results import ResultAggregator

# =========================================================================
# MetricAggregator
# =========================================================================


class TestMetricAggregator:
    def test_add_step_stores_data(self):
        agg = MetricAggregator()
        agg.add_step(0, {"rq": {"conv1": 1.0, "conv2": 2.0}})
        agg.add_step(1, {"rq": {"conv1": 1.5, "conv2": 2.5}})
        assert len(agg.steps) == 2

    def test_get_metric_evolution(self):
        agg = MetricAggregator()
        agg.add_step(0, {"rq": {"conv1": 1.0}})
        agg.add_step(5, {"rq": {"conv1": 2.0}})
        steps, values = agg.get_metric_evolution("rq", "conv1")
        assert steps == [0, 5]
        assert values == [1.0, 2.0]

    def test_get_metric_evolution_missing(self):
        agg = MetricAggregator()
        steps, values = agg.get_metric_evolution("missing", "layer")
        assert steps == []
        assert values == []

    def test_scalar_metric(self):
        agg = MetricAggregator()
        agg.add_step(0, {"loss": 0.5})
        agg.add_step(1, {"loss": 0.3})
        steps, values = agg.get_metric_evolution("loss", "value")
        assert values == [0.5, 0.3]

    def test_compute_trends(self):
        agg = MetricAggregator()
        for i in range(20):
            agg.add_step(i, {"rq": {"conv1": float(i) * 0.1}})
        trends = agg.compute_trends("rq", "conv1", window_size=5)
        assert "slope" in trends
        assert trends["slope"] > 0  # monotonically increasing
        assert "initial_value" in trends
        assert "final_value" in trends
        assert "moving_average" in trends
        assert len(trends["moving_average"]) > 0

    def test_compute_trends_short_series(self):
        agg = MetricAggregator()
        agg.add_step(0, {"rq": {"conv1": 1.0}})
        trends = agg.compute_trends("rq", "conv1")
        assert trends == {}  # Too short

    def test_compute_trends_change_points(self):
        agg = MetricAggregator()
        # Flat then jump
        for i in range(10):
            agg.add_step(i, {"rq": {"conv1": 1.0}})
        agg.add_step(10, {"rq": {"conv1": 100.0}})
        for i in range(11, 20):
            agg.add_step(i, {"rq": {"conv1": 1.0}})
        trends = agg.compute_trends("rq", "conv1")
        assert len(trends["change_points"]) > 0


# =========================================================================
# LayerAggregator
# =========================================================================


class TestLayerAggregator:
    def test_add_and_summarize(self):
        agg = LayerAggregator()
        agg.add_metrics({"rq": {"conv1": 1.0, "conv2": 2.0}})
        agg.add_metrics({"rq": {"conv1": 1.5, "conv2": 2.5}})
        summary = agg.get_layer_summary("conv1")
        assert "rq" in summary
        assert summary["rq"]["mean"] == pytest.approx(1.25)
        assert summary["rq"]["count"] == 2

    def test_get_layer_summary_missing(self):
        agg = LayerAggregator()
        assert agg.get_layer_summary("missing") == {}

    def test_rank_layers(self):
        agg = LayerAggregator()
        agg.add_metrics({"rq": {"layer1": 1.0, "layer2": 3.0, "layer3": 2.0}})
        ranked = agg.rank_layers("rq", criterion="mean", ascending=True)
        names = [name for name, _ in ranked]
        assert names == ["layer1", "layer3", "layer2"]

    def test_rank_layers_descending(self):
        agg = LayerAggregator()
        agg.add_metrics({"rq": {"layer1": 1.0, "layer2": 3.0}})
        ranked = agg.rank_layers("rq", ascending=False)
        assert ranked[0][0] == "layer2"

    def test_rank_layers_criterion_max(self):
        agg = LayerAggregator()
        agg.add_metrics({"rq": {"a": 1.0, "b": 5.0}})
        agg.add_metrics({"rq": {"a": 10.0, "b": 2.0}})
        ranked = agg.rank_layers("rq", criterion="max", ascending=False)
        assert ranked[0][0] == "a"  # max(1, 10) = 10 > max(5, 2) = 5

    def test_find_anomalous_layers(self):
        agg = LayerAggregator()
        # Need multiple add_metrics calls (or multiple values per layer) to get
        # enough layers, but only 1 value per layer is fine since the code uses
        # means across layers. Use 4 normal + 1 outlier with sufficient separation.
        for _ in range(3):
            agg.add_metrics({"rq": {"a": 1.0, "b": 1.1, "c": 0.9, "d": 1.05, "outlier": 100.0}})
        anomalous = agg.find_anomalous_layers("rq", threshold_std=1.5)
        assert "outlier" in anomalous

    def test_find_anomalous_too_few_layers(self):
        agg = LayerAggregator()
        agg.add_metrics({"rq": {"a": 1.0, "b": 100.0}})
        assert agg.find_anomalous_layers("rq") == []  # < 3 layers


# =========================================================================
# ResultAggregator
# =========================================================================


class TestResultAggregator:
    def test_add_and_retrieve(self):
        agg = ResultAggregator()
        agg.add_results("exp1", {"metrics": {"0": {"rq": {"conv1": 1.0}}}})
        assert "exp1" in agg.results

    def test_get_metric_values(self):
        agg = ResultAggregator()
        agg.add_results("exp1", {"metrics": {"0": {"rq": {"conv1": 1.0}}}})
        agg.add_results("exp2", {"metrics": {"0": {"rq": {"conv1": 2.0}}}})
        vals = agg.get_metric_values("rq", "conv1")
        assert vals["exp1"] == 1.0
        assert vals["exp2"] == 2.0

    def test_get_metric_values_missing_experiment(self):
        agg = ResultAggregator()
        vals = agg.get_metric_values("rq", "conv1", experiment_names=["nonexistent"])
        assert vals == {}

    def test_compute_statistics(self):
        agg = ResultAggregator()
        for i in range(5):
            agg.add_results(f"exp{i}", {"metrics": {str(i): {"rq": {"conv1": float(i)}}}})
        stats = agg.compute_statistics("rq", "conv1")
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["count"] == 5

    def test_compute_statistics_with_pattern(self):
        agg = ResultAggregator()
        agg.add_results("cifar_exp1", {"metrics": {"0": {"rq": {"c": 1.0}}}})
        agg.add_results("mnist_exp1", {"metrics": {"0": {"rq": {"c": 5.0}}}})
        stats = agg.compute_statistics("rq", "c", experiment_pattern="cifar")
        assert stats["count"] == 1

    def test_load_from_file(self, tmp_path):
        data = {"metrics": {"0": {"acc": {"layer1": 0.9}}}}
        fpath = tmp_path / "test_results.json"
        fpath.write_text(json.dumps(data))
        agg = ResultAggregator()
        agg.load_from_file(fpath)
        assert "test_results" in agg.results

    def test_to_dataframe(self):
        agg = ResultAggregator()
        agg.add_results("exp1", {"metrics": {"0": {"acc": 0.9}}})
        df = agg.to_dataframe()
        assert len(df) == 1
        assert "experiment" in df.columns
