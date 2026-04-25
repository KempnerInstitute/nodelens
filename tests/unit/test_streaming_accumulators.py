"""
Unit tests for streaming covariance/variance accumulators.

Tests validate:
- _CovAccumulator: streaming vs batch covariance match
- _VarAccumulator: streaming vs np.var match
- Edge cases: single sample, empty update
"""

import numpy as np
import pytest

from nodelens.experiments.cluster_experiments import _CovAccumulator, _VarAccumulator

# ---------------------------------------------------------------------------
# Tests: _CovAccumulator
# ---------------------------------------------------------------------------


class TestCovAccumulator:
    def test_streaming_matches_batch_covariance(self):
        """Streaming accumulation should match batch np.cov."""
        rng = np.random.default_rng(42)
        n, c = 200, 5
        y = rng.standard_normal((n, c))
        t = rng.standard_normal(n)

        # Batch
        batch_cov_yy = np.cov(y, rowvar=False)
        batch_var_y = np.var(y, axis=0, ddof=1)

        # Streaming in chunks
        acc = _CovAccumulator(c)
        chunk_size = 20
        for i in range(0, n, chunk_size):
            acc.update(y[i : i + chunk_size], t[i : i + chunk_size])

        var_t, var_y, cov_yy, cov_ty = acc.finalize()

        np.testing.assert_allclose(cov_yy, batch_cov_yy, atol=1e-10)
        np.testing.assert_allclose(var_y, batch_var_y, atol=1e-10)

    def test_single_update_matches_batch(self):
        rng = np.random.default_rng(0)
        n, c = 50, 3
        y = rng.standard_normal((n, c))
        t = rng.standard_normal(n)

        acc = _CovAccumulator(c)
        acc.update(y, t)
        var_t, var_y, cov_yy, cov_ty = acc.finalize()

        np.testing.assert_allclose(var_y, np.var(y, axis=0, ddof=1), atol=1e-10)
        np.testing.assert_allclose(var_t, np.var(t, ddof=1), atol=1e-10)

    def test_cov_ty_correct(self):
        """Covariance between target and channels should match manual computation."""
        rng = np.random.default_rng(42)
        n, c = 100, 4
        y = rng.standard_normal((n, c))
        t = rng.standard_normal(n)

        acc = _CovAccumulator(c)
        acc.update(y, t)
        _, _, _, cov_ty = acc.finalize()

        # Manual: cov(t, y_j) for each j
        manual_cov_ty = np.array([np.cov(t, y[:, j])[0, 1] for j in range(c)])
        np.testing.assert_allclose(cov_ty, manual_cov_ty, atol=1e-10)

    def test_empty_update_ignored(self):
        acc = _CovAccumulator(3)
        acc.update(np.empty((0, 3)), np.empty(0))
        assert acc.n == 0

    def test_single_sample_returns_zeros(self):
        acc = _CovAccumulator(3)
        acc.update(np.ones((1, 3)), np.array([1.0]))
        var_t, var_y, cov_yy, cov_ty = acc.finalize()
        # n < 2 -> should return zeros
        assert var_t == 0.0
        np.testing.assert_array_equal(var_y, np.zeros(3))

    def test_invalid_shape_raises(self):
        acc = _CovAccumulator(3)
        with pytest.raises(ValueError):
            acc.update(np.ones(5), np.ones(5))  # 1D y instead of 2D

    def test_mismatched_n_raises(self):
        acc = _CovAccumulator(3)
        with pytest.raises(ValueError):
            acc.update(np.ones((5, 3)), np.ones(3))  # 5 vs 3 samples


# ---------------------------------------------------------------------------
# Tests: _VarAccumulator
# ---------------------------------------------------------------------------


class TestVarAccumulator:
    def test_streaming_matches_np_var(self):
        rng = np.random.default_rng(42)
        n, c = 200, 5
        y = rng.standard_normal((n, c))

        acc = _VarAccumulator(c)
        chunk = 25
        for i in range(0, n, chunk):
            acc.update(y[i : i + chunk])

        var = acc.variance()
        np.testing.assert_allclose(var, np.var(y, axis=0, ddof=1), atol=1e-10)

    def test_single_batch(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal((50, 3))
        acc = _VarAccumulator(3)
        acc.update(y)
        np.testing.assert_allclose(acc.variance(), np.var(y, axis=0, ddof=1), atol=1e-10)

    def test_single_sample_returns_zeros(self):
        acc = _VarAccumulator(4)
        acc.update(np.ones((1, 4)))
        var = acc.variance()
        np.testing.assert_array_equal(var, np.zeros(4))

    def test_empty_update_ignored(self):
        acc = _VarAccumulator(3)
        acc.update(np.empty((0, 3)))
        assert acc.n == 0

    def test_constant_data_zero_variance(self):
        y = np.ones((50, 3)) * 7.0
        acc = _VarAccumulator(3)
        acc.update(y)
        var = acc.variance()
        np.testing.assert_allclose(var, 0.0, atol=1e-10)

    def test_invalid_shape_raises(self):
        acc = _VarAccumulator(3)
        with pytest.raises(ValueError):
            acc.update(np.ones(5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
