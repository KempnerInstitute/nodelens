"""
Unit tests for NeurIPS control knobs used in cluster-analysis experiments.
"""

import numpy as np

from nodelens.experiments.cluster_experiments import _maybe_permute_task_target, _mi_in_proxy_from_signal_power


def test_target_permutation_none_is_identity():
    rng = np.random.default_rng(7)
    t = np.array([0.1, 0.3, 0.2, -0.5], dtype=np.float64)
    out = _maybe_permute_task_target(t, "none", rng)
    np.testing.assert_allclose(out, t)


def test_target_permutation_batch_preserves_multiset():
    rng = np.random.default_rng(11)
    t = np.array([0.1, 0.3, 0.2, -0.5, 0.9], dtype=np.float64)
    out = _maybe_permute_task_target(t, "batch", rng)
    assert out.shape == t.shape
    assert sorted(out.tolist()) == sorted(t.tolist())


def test_mi_in_proxy_default_uses_median_reference():
    signal_power = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    mi, sigma0 = _mi_in_proxy_from_signal_power(signal_power, sigma_mode="median")
    expected_sigma0 = float(np.median(signal_power))
    expected = 0.5 * np.log1p(signal_power / expected_sigma0)
    assert abs(sigma0 - expected_sigma0) < 1e-12
    np.testing.assert_allclose(mi, expected, rtol=1e-10, atol=1e-12)


def test_mi_in_proxy_sigma_modes_change_scale():
    signal_power = np.array([1.0, 2.0, 4.0, 16.0, 32.0], dtype=np.float64)
    mi_med, sigma_med = _mi_in_proxy_from_signal_power(signal_power, sigma_mode="median")
    mi_p90, sigma_p90 = _mi_in_proxy_from_signal_power(signal_power, sigma_mode="p90")

    assert sigma_p90 > sigma_med
    assert np.mean(mi_p90) < np.mean(mi_med)
