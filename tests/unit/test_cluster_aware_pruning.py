"""
Unit tests for cluster-aware pruning strategy.

Tests validate:
- Composite score computation with precomputed metrics
- Critical protection constraint
- Redundant-first targeting
- Normalize helper
- CompositePruning baseline (no constraints)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from alignment.pruning.strategies.cluster_aware import ClusterAwarePruning, ClusterAwarePruningConfig, CompositePruning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_precomputed(n_channels: int = 32, seed: int = 42):
    """Create precomputed metrics + clusters for a layer."""
    rng = np.random.default_rng(seed)

    # Split into 4 roughly equal groups
    q = n_channels // 4
    rq = np.concatenate(
        [
            rng.uniform(5.0, 10.0, q),  # critical
            rng.uniform(0.1, 1.0, q),  # redundant
            rng.uniform(2.0, 4.0, q),  # synergistic
            rng.uniform(0.1, 0.5, n_channels - 3 * q),  # background
        ]
    )
    red = np.concatenate(
        [
            rng.uniform(0.0, 0.1, q),
            rng.uniform(0.7, 1.0, q),
            rng.uniform(0.0, 0.1, q),
            rng.uniform(0.0, 0.2, n_channels - 3 * q),
        ]
    )
    syn = np.concatenate(
        [
            rng.uniform(0.2, 0.4, q),
            rng.uniform(0.0, 0.1, q),
            rng.uniform(0.7, 1.0, q),
            rng.uniform(0.0, 0.1, n_channels - 3 * q),
        ]
    )

    metrics = {"rq": rq, "redundancy": red, "synergy": syn}

    # Build matching cluster labels
    labels = np.concatenate(
        [
            np.full(q, 0),  # critical
            np.full(q, 1),  # redundant
            np.full(q, 2),  # synergistic
            np.full(n_channels - 3 * q, 3),  # background
        ]
    ).astype(int)

    clusters = {
        "labels": labels,
        "centroids": np.zeros((4, 3)),
        "type_mapping": {0: "critical", 1: "redundant", 2: "synergistic", 3: "background"},
        "type_counts": {"critical": q, "redundant": q, "synergistic": q, "background": n_channels - 3 * q},
    }

    return metrics, clusters


# ---------------------------------------------------------------------------
# Tests: importance score computation
# ---------------------------------------------------------------------------


class TestComputeImportanceScores:
    def test_output_shape_and_finiteness(self):
        n_channels = 32
        conv = nn.Conv2d(16, n_channels, 3, padding=1)
        metrics, clusters = _make_precomputed(n_channels)

        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(amount=0.5),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )

        scores = cap.compute_importance_scores(
            module=conv,
            layer_name="conv1",
        )

        assert scores.shape == (n_channels,)
        assert torch.all(torch.isfinite(scores))

    def test_scores_vary_across_channels(self):
        n_channels = 32
        conv = nn.Conv2d(16, n_channels, 3, padding=1)
        metrics, clusters = _make_precomputed(n_channels)

        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(amount=0.5),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )

        scores = cap.compute_importance_scores(module=conv, layer_name="conv1")
        assert scores.std() > 0, "Scores should not all be identical"

    def test_critical_channels_get_higher_scores(self):
        """Critical channels (high RQ, low Red) should score higher on average."""
        n_channels = 32
        q = n_channels // 4
        conv = nn.Conv2d(16, n_channels, 3, padding=1)
        metrics, clusters = _make_precomputed(n_channels)

        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(amount=0.5),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )

        scores = cap.compute_importance_scores(module=conv, layer_name="conv1")
        critical_mean = scores[:q].mean()
        redundant_mean = scores[q : 2 * q].mean()
        assert critical_mean > redundant_mean, f"Critical mean ({critical_mean:.3f}) should exceed redundant mean ({redundant_mean:.3f})"


# ---------------------------------------------------------------------------
# Tests: channel selection with constraints
# ---------------------------------------------------------------------------


class TestSelectChannelsToPrune:
    def test_correct_number_pruned(self):
        n_channels = 32
        metrics, clusters = _make_precomputed(n_channels)
        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(amount=0.5, protect_critical_frac=1.0),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )
        cap._cluster_cache["conv1"] = clusters
        cap._metrics_cache["conv1"] = metrics

        scores = torch.randn(n_channels)
        n_prune = 10
        selected = cap.select_channels_to_prune(scores, n_prune, layer_name="conv1")
        assert len(selected) == n_prune

    def test_critical_protection_constraint(self):
        """At most protect_critical_frac of critical channels should be pruned."""
        n_channels = 32
        q = n_channels // 4  # 8 critical channels
        metrics, clusters = _make_precomputed(n_channels)

        protect_frac = 0.25  # at most 25% of critical -> at most 2 of 8
        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(
                amount=0.5,
                protect_critical_frac=protect_frac,
                target_redundant=False,
                synergy_pair_constraint=False,
            ),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )
        cap._cluster_cache["conv1"] = clusters
        cap._metrics_cache["conv1"] = metrics

        # Give critical channels the lowest scores so pruner WANTS to prune them
        scores = torch.zeros(n_channels)
        scores[:q] = -10.0  # critical channels have lowest scores

        n_prune = 16  # try to prune half
        selected = cap.select_channels_to_prune(scores, n_prune, layer_name="conv1")

        critical_pruned = sum(1 for idx in selected if idx < q)
        max_allowed = int(q * protect_frac)
        assert critical_pruned <= max_allowed, f"Pruned {critical_pruned} critical channels, max allowed {max_allowed}"

    def test_target_redundant_prunes_redundant_first(self):
        """With target_redundant=True, redundant/background should be pruned before others."""
        n_channels = 32
        q = n_channels // 4
        metrics, clusters = _make_precomputed(n_channels)

        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(
                amount=0.5,
                protect_critical_frac=1.0,
                target_redundant=True,
                synergy_pair_constraint=False,
            ),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )
        cap._cluster_cache["conv1"] = clusters
        cap._metrics_cache["conv1"] = metrics

        # Uniform scores so only priority matters
        scores = torch.ones(n_channels)
        n_prune = q  # prune exactly one group's worth

        selected = cap.select_channels_to_prune(scores, n_prune, layer_name="conv1")

        # All pruned should be redundant (idx q..2q) or background (idx 3q..)
        redundant_bg_idx = set(range(q, 2 * q)) | set(range(3 * q, n_channels))
        pruned_from_target = sum(1 for idx in selected if idx in redundant_bg_idx)
        # Most should come from redundant/background
        assert pruned_from_target >= n_prune * 0.8, f"Expected ≥{int(n_prune*0.8)} from redundant/bg, got {pruned_from_target}"

    def test_protected_indices_respected(self):
        n_channels = 16
        metrics, clusters = _make_precomputed(n_channels)
        cap = ClusterAwarePruning(
            config=ClusterAwarePruningConfig(
                amount=0.5,
                protect_critical_frac=1.0,
                target_redundant=False,
                synergy_pair_constraint=False,
            ),
            precomputed_metrics=metrics,
            precomputed_clusters=clusters,
        )
        cap._cluster_cache["conv1"] = clusters
        cap._metrics_cache["conv1"] = metrics

        scores = torch.randn(n_channels)
        protected = [0, 1, 2]
        selected = cap.select_channels_to_prune(
            scores,
            5,
            layer_name="conv1",
            protected_indices=protected,
        )
        for p in protected:
            assert p not in selected, f"Protected index {p} should not be pruned"


# ---------------------------------------------------------------------------
# Tests: normalize helper
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_known_values(self):
        cap = ClusterAwarePruning()
        result = cap._normalize(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_constant_input(self):
        cap = ClusterAwarePruning()
        result = cap._normalize(np.array([5.0, 5.0, 5.0]))
        # When all equal, x_max == x_min, return x unchanged
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_list_input(self):
        cap = ClusterAwarePruning()
        result = cap._normalize([1.0, 3.0, 5.0])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Tests: CompositePruning baseline
# ---------------------------------------------------------------------------


class TestCompositePruning:
    def test_constraints_disabled(self):
        cp = CompositePruning()
        assert cp.config.protect_critical_frac == 1.0
        assert cp.config.target_redundant is False
        assert cp.config.synergy_pair_constraint is False
        assert cp.config.lambda_halo == 0.0

    def test_simple_selection(self):
        """CompositePruning should prune lowest-scoring channels regardless of type."""
        cp = CompositePruning()
        scores = torch.arange(16, dtype=torch.float)  # 0..15
        selected = cp.select_channels_to_prune(scores, n_prune=4)
        assert sorted(selected) == [0, 1, 2, 3]

    def test_selection_respects_protected(self):
        cp = CompositePruning()
        scores = torch.arange(16, dtype=torch.float)
        selected = cp.select_channels_to_prune(
            scores,
            n_prune=4,
            protected_indices=[0, 1],
        )
        assert 0 not in selected
        assert 1 not in selected
        assert len(selected) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
