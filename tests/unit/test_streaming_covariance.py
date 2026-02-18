"""
Unit tests for StreamingCovariance utilities.
"""

import torch

from alignment.core.streaming import StreamingCovariance


class TestStreamingCovariance:
    """Tests for StreamingCovariance class."""

    def test_matches_full_covariance_small(self):
        """Streaming covariance should closely match torch.cov on small data."""
        torch.manual_seed(0)
        n, d = 128, 6
        x = torch.randn(n, d)

        # Full covariance using torch.cov (rowvar=False equivalent)
        x_centered = x - x.mean(dim=0, keepdim=True)
        full_cov = x_centered.t().mm(x_centered) / (n - 1)

        # Streaming covariance with small batches
        streamer = StreamingCovariance(input_dim=d)
        batch_size = 16
        for i in range(0, n, batch_size):
            streamer.update(x[i : i + batch_size])

        stream_cov = streamer.get_covariance()

        assert stream_cov.shape == (d, d)
        assert torch.allclose(stream_cov, full_cov, atol=1e-5, rtol=1e-4)

    def test_handles_single_sample_batches_and_empty(self):
        """Streaming covariance should handle single-sample and empty updates."""
        torch.manual_seed(1)
        d = 4
        streamer = StreamingCovariance(input_dim=d)

        # Empty batch should be a no-op
        empty = torch.empty(0, d)
        streamer.update(empty)
        assert streamer.n_samples == 0

        # Single-sample updates
        for _ in range(5):
            x = torch.randn(d)
            streamer.update(x)

        cov = streamer.get_covariance()
        assert cov.shape == (d, d)
        # With very few samples, covariance will be noisy but finite
        assert not torch.isnan(cov).any()
        assert not torch.isinf(cov).any()
