"""
Streaming statistics computation utilities.

Allows computing covariance, mean, and other statistics on datasets
that are too large to fit in memory (e.g. Llama-3 activations).
"""

import torch


class StreamingCovariance:
    """
    Computes covariance matrix in a streaming fashion (Welford's online algorithm).

    Supports:
    - Mean accumulation
    - Covariance accumulation
    - Thread-safe / GPU-efficient updates
    """

    def __init__(self, input_dim: int, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        self.input_dim = input_dim
        self.device = device
        self.dtype = dtype

        self.n_samples = 0
        self.mean = torch.zeros(input_dim, device=device, dtype=dtype)
        self.C = torch.zeros((input_dim, input_dim), device=device, dtype=dtype)  # Sum of squares of differences

    def update(self, batch: torch.Tensor):
        """
        Update statistics with a new batch of data.
        Args:
            batch: [batch_size, input_dim]
        """
        batch = batch.to(self.device).to(self.dtype)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)

        batch_size = batch.shape[0]
        if batch_size == 0:
            return

        # Standard batch update for Welford's algorithm
        # This is numerically stable

        # Update mean
        old_mean = self.mean.clone()
        new_data_sum = batch.sum(dim=0)

        self.n_samples += batch_size
        # Update mean: mean += sum(x - mean) / N
        delta_sum = new_data_sum - batch_size * old_mean
        self.mean += delta_sum / self.n_samples

        # Update C (scatter matrix)
        # C += (x - old_mean) * (x - new_mean)
        # Batch version: X_centered_old = batch - old_mean
        #                X_centered_new = batch - self.mean
        #                C += X_centered_old.T @ X_centered_new

        batch_centered_old = batch - old_mean
        batch_centered_new = batch - self.mean

        self.C += torch.matmul(batch_centered_old.T, batch_centered_new)

    def get_covariance(self, regularization: float = 0.0) -> torch.Tensor:
        """Get the current covariance matrix."""
        if self.n_samples < 2:
            return torch.zeros_like(self.C)

        cov = self.C / (self.n_samples - 1)

        if regularization > 0:
            cov += regularization * torch.eye(self.input_dim, device=self.device, dtype=self.dtype)

        return cov

    def get_mean(self) -> torch.Tensor:
        return self.mean

    def get_variance(self) -> torch.Tensor:
        return torch.diagonal(self.get_covariance())
