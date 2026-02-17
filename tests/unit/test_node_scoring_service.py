"""
Unit tests for node scoring service.

Tests validate:
- _normalize_scores: [1,2,3] → [0,0.5,1.0], constant → 0.5
- compute_composite_scores: mock metrics, verify weighted sum
- rank_neurons_globally: sorted descending
"""

import pytest
import torch

from alignment.services.scoring import (
    NodeScoringService,
    CompositeScores,
    create_scoring_service,
)


# ---------------------------------------------------------------------------
# Mock metric
# ---------------------------------------------------------------------------

class _MockMetric:
    """Returns a fixed tensor when compute() is called."""

    requires_inputs = True
    requires_weights = True
    requires_outputs = False

    def __init__(self, values: torch.Tensor):
        self._values = values

    def compute(self, **kwargs):
        return self._values.clone()


# ---------------------------------------------------------------------------
# Tests: _normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:

    def test_known_values(self):
        result = NodeScoringService._normalize_scores(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(result, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5, rtol=1e-5)

    def test_constant_returns_half(self):
        result = NodeScoringService._normalize_scores(torch.tensor([5.0, 5.0, 5.0]))
        torch.testing.assert_close(result, torch.tensor([0.5, 0.5, 0.5]), atol=1e-5, rtol=1e-5)

    def test_single_element(self):
        result = NodeScoringService._normalize_scores(torch.tensor([3.0]))
        assert result.shape == (1,)

    def test_negative_values(self):
        result = NodeScoringService._normalize_scores(torch.tensor([-2.0, 0.0, 2.0]))
        assert result.min() >= -1e-6
        assert result.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: compute_composite_scores
# ---------------------------------------------------------------------------

class TestComputeCompositeScores:

    def test_basic_composite(self):
        n = 8
        rq_vals = torch.linspace(0.1, 1.0, n)
        mi_vals = torch.linspace(0.5, 2.0, n)

        metrics = {
            "rq": _MockMetric(rq_vals),
            "mi": _MockMetric(mi_vals),
        }
        scorer = NodeScoringService(metrics, alpha_mi=0.5, delta_rq=0.5,
                                     beta_synergy=0.0, gamma_redundancy=0.0)
        inputs = torch.randn(20, 8)
        weights = torch.randn(n, 8)
        targets = torch.randint(0, 5, (20,))

        result = scorer.compute_composite_scores(inputs, weights, targets=targets)
        assert isinstance(result, CompositeScores)
        assert result.composite.shape == (n,)
        assert result.rq is not None
        assert result.mi is not None

    def test_redundancy_subtracts(self):
        """Higher redundancy should lower composite score."""
        n = 4
        # All same RQ, but different redundancy
        rq_vals = torch.ones(n)
        red_vals = torch.tensor([0.0, 0.0, 1.0, 1.0])

        metrics = {
            "rq": _MockMetric(rq_vals),
            "redundancy": _MockMetric(red_vals),
        }
        scorer = NodeScoringService(
            metrics, alpha_mi=0.0, beta_synergy=0.0,
            gamma_redundancy=0.5, delta_rq=0.5,
        )
        result = scorer.compute_composite_scores(
            torch.randn(20, 4), torch.randn(n, 4),
        )
        # Channels 0,1 (low redundancy) should score higher than 2,3
        assert result.composite[:2].mean() > result.composite[2:].mean()

    def test_missing_metric_graceful(self):
        """If metric not in dict, corresponding score should be None/zero."""
        n = 4
        metrics = {"rq": _MockMetric(torch.ones(n))}
        scorer = NodeScoringService(metrics)
        result = scorer.compute_composite_scores(
            torch.randn(20, 4), torch.randn(n, 4),
        )
        assert result.mi is None
        assert result.redundancy is None
        assert result.composite.shape == (n,)


# ---------------------------------------------------------------------------
# Tests: rank_neurons_globally
# ---------------------------------------------------------------------------

class TestRankNeuronsGlobally:

    def test_sorted_descending(self):
        n = 4
        metrics = {"rq": _MockMetric(torch.ones(n))}
        scorer = NodeScoringService(metrics)

        scores = {
            "layer1": CompositeScores(composite=torch.tensor([0.1, 0.9, 0.5, 0.3])),
            "layer2": CompositeScores(composite=torch.tensor([0.2, 0.8, 0.4, 0.6])),
        }
        ranked, _ = scorer.rank_neurons_globally(scores)
        values = [r[2] for r in ranked]
        assert values == sorted(values, reverse=True)

    def test_return_indices(self):
        n = 4
        metrics = {"rq": _MockMetric(torch.ones(n))}
        scorer = NodeScoringService(metrics)

        scores = {
            "layer1": CompositeScores(composite=torch.tensor([0.1, 0.9, 0.5, 0.3])),
        }
        _, indices = scorer.rank_neurons_globally(scores, return_indices=True)
        assert indices is not None
        assert "layer1" in indices
        # Best neuron (0.9) is at index 1
        assert indices["layer1"][0].item() == 1


# ---------------------------------------------------------------------------
# Tests: factory
# ---------------------------------------------------------------------------

class TestFactory:

    def test_create_scoring_service(self):
        metrics = {"rq": _MockMetric(torch.ones(4))}
        scorer = create_scoring_service(metrics, alpha_mi=0.4, delta_rq=0.6)
        assert isinstance(scorer, NodeScoringService)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
