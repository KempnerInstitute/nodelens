"""
Unit tests for cross-layer halo analysis.

Tests validate:
- compute_influence: shape, non-negativity, activation weighting
- find_halo: non-empty for dominant column, percentile threshold
- compute_cluster_to_cluster_flow: rows sum to ~1.0
- permutation_baseline: z-scores and p-values computed
"""

import numpy as np
import pytest

from nodelens.analysis.clustering.cross_layer_halo import CrossLayerHaloAnalysis, HaloResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_weights(rng):
    """Weight matrix [out=16, in=8]."""
    return rng.standard_normal((16, 8)).astype(np.float64)


@pytest.fixture
def small_activations(rng):
    """Activation matrix [batch=50, in=8]."""
    return rng.standard_normal((50, 8)).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests: compute_influence
# ---------------------------------------------------------------------------


class TestComputeInfluence:
    def test_shape(self, small_weights):
        halo = CrossLayerHaloAnalysis()
        infl = halo.compute_influence(small_weights)
        assert infl.shape == small_weights.shape

    def test_non_negative(self, small_weights):
        halo = CrossLayerHaloAnalysis()
        infl = halo.compute_influence(small_weights)
        assert np.all(infl >= 0)

    def test_without_activation_weight(self, small_weights):
        halo = CrossLayerHaloAnalysis(use_activation_weight=False)
        infl = halo.compute_influence(small_weights)
        np.testing.assert_allclose(infl, np.abs(small_weights))

    def test_activation_weighting_effect(self, small_weights, small_activations):
        halo = CrossLayerHaloAnalysis(use_activation_weight=True)
        infl_w = halo.compute_influence(small_weights, small_activations)
        infl_no = halo.compute_influence(small_weights)
        # Should differ when activations are provided
        assert not np.allclose(infl_w, infl_no)


# ---------------------------------------------------------------------------
# Tests: find_halo
# ---------------------------------------------------------------------------


class TestFindHalo:
    def test_dominant_column_produces_nonempty_halo(self, rng):
        """When one source column dominates, its halo should be non-empty."""
        # Make column 0 dominate for a few receivers
        W = rng.standard_normal((20, 8)).astype(np.float64)
        W[:5, 0] = 100.0  # receivers 0..4 dominated by source 0
        halo = CrossLayerHaloAnalysis(percentile=80)
        infl = halo.compute_influence(W)
        halo_idx, rel_infl = halo.find_halo(infl, cluster_indices=np.array([0]))
        assert len(halo_idx) > 0

    def test_returned_indices_valid(self, small_weights):
        halo = CrossLayerHaloAnalysis(percentile=90)
        infl = halo.compute_influence(small_weights)
        halo_idx, rel_infl = halo.find_halo(infl, cluster_indices=np.array([0, 1]))
        assert all(0 <= i < small_weights.shape[0] for i in halo_idx)
        assert rel_infl.shape == (small_weights.shape[0],)

    def test_relative_influence_bounded(self, small_weights):
        halo = CrossLayerHaloAnalysis()
        infl = halo.compute_influence(small_weights)
        _, rel_infl = halo.find_halo(infl, cluster_indices=np.array([0]))
        assert np.all(rel_infl >= 0)
        assert np.all(rel_infl <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Tests: analyze_halo
# ---------------------------------------------------------------------------


class TestAnalyzeHalo:
    def test_empty_halo(self):
        halo = CrossLayerHaloAnalysis()
        result = halo.analyze_halo(
            halo_indices=np.array([], dtype=int),
            redundancy=np.ones(10),
            synergy=np.ones(10),
        )
        assert isinstance(result, HaloResult)
        assert result.halo_size == 0

    def test_nonempty_halo_stats(self, rng):
        halo = CrossLayerHaloAnalysis()
        red = rng.uniform(0, 1, 20)
        syn = rng.uniform(0, 1, 20)
        idx = np.array([2, 5, 7])
        result = halo.analyze_halo(idx, red, syn, layer_name="conv1", cluster_name="critical")
        assert result.halo_size == 3
        assert result.layer_name == "conv1"
        assert result.source_cluster == "critical"
        np.testing.assert_almost_equal(result.halo_redundancy_mean, red[idx].mean())
        np.testing.assert_almost_equal(result.halo_synergy_mean, syn[idx].mean())


# ---------------------------------------------------------------------------
# Tests: cluster_to_cluster_flow
# ---------------------------------------------------------------------------


class TestClusterToClusterFlow:
    def test_rows_sum_to_one(self, rng):
        n_out, n_in = 16, 8
        W = np.abs(rng.standard_normal((n_out, n_in)))
        source_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        target_labels = np.array([0] * 8 + [1] * 8)
        source_types = {0: "critical", 1: "redundant"}
        target_types = {0: "critical", 1: "redundant"}

        halo = CrossLayerHaloAnalysis()
        infl = halo.compute_influence(W)
        flow = halo.compute_cluster_to_cluster_flow(
            infl,
            source_labels,
            target_labels,
            source_types,
            target_types,
        )
        for src_type, row in flow.items():
            row_sum = sum(row.values())
            assert abs(row_sum - 1.0) < 0.05, f"Row {src_type} sums to {row_sum}, expected ~1.0"

    def test_keys_match_types(self, rng):
        n_out, n_in = 12, 6
        W = np.abs(rng.standard_normal((n_out, n_in)))
        source_labels = np.array([0, 0, 1, 1, 2, 2])
        target_labels = np.array([0] * 4 + [1] * 4 + [2] * 4)
        source_types = {0: "A", 1: "B", 2: "C"}
        target_types = {0: "X", 1: "Y", 2: "Z"}

        halo = CrossLayerHaloAnalysis()
        infl = halo.compute_influence(W)
        flow = halo.compute_cluster_to_cluster_flow(
            infl,
            source_labels,
            target_labels,
            source_types,
            target_types,
        )
        assert set(flow.keys()) == {"A", "B", "C"}
        for row in flow.values():
            assert set(row.keys()) == {"X", "Y", "Z"}


# ---------------------------------------------------------------------------
# Tests: permutation_baseline
# ---------------------------------------------------------------------------


class TestPermutationBaseline:
    def test_returns_z_scores_and_pvalues(self, rng):
        n_out, n_in = 20, 10
        W = np.abs(rng.standard_normal((n_out, n_in)))
        labels = np.array([0] * 5 + [1] * 5)
        type_mapping = {0: "critical", 1: "redundant"}
        red = rng.uniform(0, 1, n_out)
        syn = rng.uniform(0, 1, n_out)

        halo = CrossLayerHaloAnalysis(percentile=80)
        infl = halo.compute_influence(W)
        results = halo.permutation_baseline(
            infl,
            labels,
            type_mapping,
            red,
            syn,
            n_permutations=50,
            seed=42,
        )
        for ctype, stats in results.items():
            assert "z_red" in stats
            assert "z_syn" in stats
            assert "p_red" in stats
            assert "p_syn" in stats
            assert 0 <= stats["p_red"] <= 1.0
            assert 0 <= stats["p_syn"] <= 1.0
            assert stats["n_permutations"] == 50

    def test_n_permutations_respected(self, rng):
        n_out, n_in = 12, 6
        W = np.abs(rng.standard_normal((n_out, n_in)))
        labels = np.array([0, 0, 0, 1, 1, 1])
        type_mapping = {0: "A", 1: "B"}
        red = rng.uniform(0, 1, n_out)
        syn = rng.uniform(0, 1, n_out)

        halo = CrossLayerHaloAnalysis(percentile=80)
        infl = halo.compute_influence(W)
        results = halo.permutation_baseline(
            infl,
            labels,
            type_mapping,
            red,
            syn,
            n_permutations=20,
            seed=0,
        )
        for stats in results.values():
            assert stats["n_permutations"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
