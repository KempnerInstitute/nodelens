"""
Unit tests for similarity metrics.
"""

import pytest
import torch

from alignment.metrics.similarity import (
    NodeCorrelation,
    NodeRedundancy,
    WeightCosineSimilarity,
    WeightDotSimilarity,
    WeightEuclideanDistance,
)


class TestNodeRedundancy:
    """Test NodeRedundancy metric."""

    def test_basic_computation(self):
        """Test basic redundancy computation."""
        metric = NodeRedundancy()

        # Create correlated inputs
        batch_size = 100
        inputs = torch.randn(batch_size, 10)
        # Make some features correlated
        inputs[:, 1] = inputs[:, 0] * 0.9 + torch.randn(batch_size) * 0.1
        inputs[:, 2] = inputs[:, 0] * 0.8 + torch.randn(batch_size) * 0.2

        scores = metric.compute(inputs=inputs)

        assert scores.shape == (10,)
        assert scores[0] > 0  # First feature should have high redundancy
        assert scores[1] > 0  # Second feature should have high redundancy
        assert not torch.isnan(scores).any()

    def test_single_feature(self):
        """Test with single feature input."""
        metric = NodeRedundancy()
        inputs = torch.randn(50, 1)
        scores = metric.compute(inputs=inputs)

        assert scores.shape == (1,)
        assert scores[0] == 0  # Single feature has no redundancy

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        metric = NodeRedundancy()
        inputs = torch.randn(1, 10)  # Only 1 sample
        scores = metric.compute(inputs=inputs)

        assert scores.shape == (10,)
        assert (scores == 0).all()  # Should return zeros


class TestWeightSimilarityMetrics:
    """Test weight similarity metrics."""

    def test_cosine_similarity(self):
        """Test weight cosine similarity."""
        metric = WeightCosineSimilarity()

        # Create weights with known similarity
        weights = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]  # Same as first  # Orthogonal to first  # Opposite to first
        )

        scores = metric.compute(weights=weights)

        assert scores.shape == (4,)
        assert abs(scores[0] - 0.0) < 0.1  # Average of 1, 0, -1
        assert abs(scores[1] - 0.0) < 0.1  # Average of 1, 0, -1
        assert not torch.isnan(scores).any()

    def test_dot_similarity(self):
        """Test weight dot product similarity."""
        metric = WeightDotSimilarity()

        weights = torch.eye(3)  # Orthonormal vectors
        scores = metric.compute(weights=weights)

        assert scores.shape == (3,)
        assert (scores == 0).all()  # All orthogonal

    def test_euclidean_distance(self):
        """Test weight Euclidean distance."""
        metric = WeightEuclideanDistance()

        # Create weights with known distances
        weights = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],  # Distance 1 from first
                [1.0, 1.0],  # Distance 1 from first
            ]
        )

        scores = metric.compute(weights=weights)

        assert scores.shape == (3,)
        assert scores[0] > 0  # Average distance to others
        assert not torch.isnan(scores).any()

    def test_single_neuron(self):
        """Test with single neuron."""
        for metric_class in [WeightCosineSimilarity, WeightDotSimilarity, WeightEuclideanDistance]:
            metric = metric_class()
            weights = torch.randn(1, 10)
            scores = metric.compute(weights=weights)

            assert scores.shape == (1,)
            assert scores[0] == 0  # No other neurons to compare


class TestNodeCorrelation:
    """Test NodeCorrelation metric."""

    def test_basic_computation(self):
        """Test basic correlation computation."""
        metric = NodeCorrelation()

        # Create correlated outputs
        batch_size = 100
        outputs = torch.randn(batch_size, 5)
        # Make some outputs correlated
        outputs[:, 1] = outputs[:, 0] * 0.9 + torch.randn(batch_size) * 0.1

        scores = metric.compute(outputs=outputs)

        assert scores.shape == (5,)
        assert scores[0] > 0  # Should have correlation with neuron 1
        assert scores[1] > 0  # Should have correlation with neuron 0
        assert not torch.isnan(scores).any()

    def test_absolute_correlation(self):
        """Test absolute correlation option."""
        metric = NodeCorrelation(absolute=True)

        outputs = torch.randn(50, 3)
        outputs[:, 1] = -outputs[:, 0]  # Perfect negative correlation

        scores = metric.compute(outputs=outputs)

        assert scores[0] > 0.5  # High absolute correlation
        assert scores[1] > 0.5  # High absolute correlation

    def test_no_absolute_correlation(self):
        """Test without absolute correlation."""
        metric = NodeCorrelation(absolute=False)

        outputs = torch.randn(50, 3)
        outputs[:, 1] = -outputs[:, 0]  # Perfect negative correlation

        scores = metric.compute(outputs=outputs)

        assert scores[0] < -0.5  # Negative correlation preserved
        assert scores[1] < -0.5  # Negative correlation preserved

    def test_zero_variance_neurons(self):
        """Test handling of zero variance neurons."""
        metric = NodeCorrelation()

        outputs = torch.randn(50, 3)
        outputs[:, 1] = 1.0  # Constant output

        scores = metric.compute(outputs=outputs)

        assert scores.shape == (3,)
        assert scores[1] == 0  # Zero variance neuron should have 0 correlation
        assert not torch.isnan(scores).any()


class TestEdgeCases:
    """Test edge cases for all similarity metrics."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        metrics = [NodeRedundancy(), WeightCosineSimilarity(), NodeCorrelation()]

        for metric in metrics:
            with pytest.raises(ValueError):
                if isinstance(metric, NodeRedundancy):
                    metric.compute(inputs=None)
                elif isinstance(metric, WeightCosineSimilarity):
                    metric.compute(weights=None)
                else:
                    metric.compute(outputs=None)

    def test_wrong_dimensions(self):
        """Test with wrong tensor dimensions."""
        # Test 1D inputs
        metric = NodeRedundancy()
        inputs = torch.randn(10)  # 1D instead of 2D
        scores = metric.compute(inputs=inputs)
        assert scores.shape[0] > 0

        # Test 3D inputs (should flatten)
        inputs = torch.randn(10, 5, 5)
        scores = metric.compute(inputs=inputs)
        assert scores.shape == (25,)  # Flattened to 25 features

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        metric = WeightCosineSimilarity()

        # Very small weights
        weights = torch.randn(3, 10) * 1e-8
        scores = metric.compute(weights=weights)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

        # Very large weights
        weights = torch.randn(3, 10) * 1e8
        scores = metric.compute(weights=weights)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
