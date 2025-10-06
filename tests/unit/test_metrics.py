"""
Unit tests for alignment metrics using registry pattern.
"""

from typing import Tuple

import numpy as np
import pytest
import torch

from alignment.metrics import get_metric, list_metrics


class TestMetricRegistry:
    """Test suite for metric registry functionality."""

    def test_list_metrics(self):
        """Test listing available metrics."""
        metrics = list_metrics()

        assert isinstance(metrics, (list, tuple))
        assert len(metrics) > 0

    def test_get_rayleigh_metric(self):
        """Test getting Rayleigh quotient metric from registry."""
        metric = get_metric("rayleigh_quotient")

        assert metric is not None
        assert hasattr(metric, "compute")

    def test_get_mutual_information_metric(self):
        """Test getting mutual information metric from registry."""
        metric = get_metric("mutual_information_gaussian")

        assert metric is not None
        assert hasattr(metric, "compute")

    def test_invalid_metric_name(self):
        """Test that invalid metric name raises error."""
        with pytest.raises(Exception):
            get_metric("nonexistent_metric")


class TestRayleighQuotient:
    """Test suite for Rayleigh Quotient metric."""

    @pytest.fixture
    def sample_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample data for testing."""
        batch_size, input_dim, output_dim = 32, 10, 5
        inputs = torch.randn(batch_size, input_dim)
        weights = torch.randn(output_dim, input_dim)
        return inputs, weights

    def test_compute_basic(self, sample_data):
        """Test basic RQ computation."""
        inputs, weights = sample_data
        metric = get_metric("rayleigh_quotient")

        scores = metric.compute(inputs=inputs, weights=weights)

        assert scores.shape == (weights.shape[0],)
        assert torch.all(torch.isfinite(scores))

    def test_with_regularization(self, sample_data):
        """Test RQ with regularization."""
        inputs, weights = sample_data

        metric = get_metric("rayleigh_quotient", regularization=1e-4)
        scores = metric.compute(inputs=inputs, weights=weights)

        assert scores.shape == (weights.shape[0],)
        assert torch.all(torch.isfinite(scores))


class TestMutualInformation:
    """Test suite for Mutual Information metric."""

    @pytest.fixture
    def sample_activations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sample activations for testing."""
        batch_size, num_neurons = 100, 10
        activations = torch.randn(batch_size, num_neurons)
        targets = torch.randint(0, 5, (batch_size,))
        return activations, targets

    def test_gaussian_mi_basic(self, sample_activations):
        """Test basic Gaussian MI computation."""
        activations, targets = sample_activations

        metric = get_metric("mutual_information_gaussian")
        scores = metric.compute(outputs=activations, targets=targets)

        assert scores.shape == (activations.shape[1],)
        assert torch.all(torch.isfinite(scores))

    def test_mi_with_bins(self, sample_activations):
        """Test MI with binning method."""
        activations, targets = sample_activations

        metric = get_metric("mutual_information_binning", num_bins=10)
        scores = metric.compute(outputs=activations, targets=targets)

        assert scores.shape == (activations.shape[1],)
        assert torch.all(torch.isfinite(scores))


class TestRedundancyMetrics:
    """Test suite for redundancy metrics."""

    @pytest.fixture
    def sample_activations(self) -> torch.Tensor:
        """Generate sample activations."""
        return torch.randn(100, 20)

    def test_pairwise_redundancy(self, sample_activations):
        """Test pairwise redundancy computation."""
        metric = get_metric("pairwise_redundancy_gaussian")

        scores = metric.compute(outputs=sample_activations)

        # Should return per-neuron scores
        assert scores.shape == (sample_activations.shape[1],)
        assert torch.all(torch.isfinite(scores))


class TestSimilarityMetrics:
    """Test suite for similarity metrics."""

    @pytest.fixture
    def sample_weights(self) -> torch.Tensor:
        """Generate sample weights."""
        return torch.randn(10, 20)

    def test_weight_cosine_similarity(self, sample_weights):
        """Test weight cosine similarity computation."""
        metric = get_metric("weight_cosine_similarity")

        scores = metric.compute(weights=sample_weights)

        # Returns per-weight scores (not matrix)
        assert scores.shape == (10,)
        assert torch.all(torch.isfinite(scores))


class TestClassConditionedMetrics:
    """Test suite for class-conditioned metrics."""

    @pytest.fixture
    def sample_class_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate class-conditioned sample data."""
        batch_size, input_dim, output_dim = 100, 15, 8
        inputs = torch.randn(batch_size, input_dim)
        weights = torch.randn(output_dim, input_dim)
        labels = torch.randint(0, 5, (batch_size,))
        return inputs, weights, labels

    def test_class_selectivity(self, sample_class_data):
        """Test class selectivity computation."""
        inputs, weights, labels = sample_class_data
        outputs = inputs @ weights.T

        metric = get_metric("class_selectivity")
        scores = metric.compute(inputs=inputs, outputs=outputs, weights=weights, targets=labels)

        assert scores.shape == (weights.shape[0],)
        assert torch.all(torch.isfinite(scores))


class TestMetricParameters:
    """Test suite for metric parameter handling."""

    def test_metric_with_custom_params(self):
        """Test creating metric with custom parameters."""
        metric = get_metric("rayleigh_quotient", regularization=1e-3, relative_scale=True)

        assert metric.regularization == 1e-3

    def test_metric_default_params(self):
        """Test metric uses default parameters when not specified."""
        metric = get_metric("rayleigh_quotient")

        # Should have default regularization
        assert hasattr(metric, "regularization")


@pytest.mark.integration
class TestMetricIntegration:
    """Integration tests for metrics."""

    def test_multiple_metrics_same_data(self):
        """Test multiple metrics on the same data."""
        inputs = torch.randn(50, 10)
        weights = torch.randn(5, 10)
        targets = torch.randint(0, 3, (50,))

        # Test multiple metrics
        rq_metric = get_metric("rayleigh_quotient")
        rq_scores = rq_metric.compute(inputs=inputs, weights=weights)

        mi_metric = get_metric("mutual_information_gaussian")
        outputs = inputs @ weights.T
        mi_scores = mi_metric.compute(outputs=outputs, targets=targets)

        # Both should produce valid scores
        assert torch.all(torch.isfinite(rq_scores))
        assert torch.all(torch.isfinite(mi_scores))

    def test_metric_pipeline(self):
        """Test using metrics in a pipeline."""
        # Simulate a layer's computation
        inputs = torch.randn(100, 20)
        weights = torch.randn(10, 20)
        outputs = inputs @ weights.T
        targets = torch.randint(0, 5, (100,))

        # Apply multiple metrics
        metrics_to_test = ["rayleigh_quotient", "mutual_information_gaussian", "pairwise_redundancy_gaussian"]

        results = {}
        for metric_name in metrics_to_test:
            metric = get_metric(metric_name)

            if metric_name == "rayleigh_quotient":
                scores = metric.compute(inputs=inputs, weights=weights)
            elif metric_name == "mutual_information_gaussian":
                scores = metric.compute(outputs=outputs, targets=targets)
            else:
                scores = metric.compute(outputs=outputs)

            results[metric_name] = scores
            assert torch.all(torch.isfinite(scores))

        # Check all metrics produced results
        assert len(results) == 3
