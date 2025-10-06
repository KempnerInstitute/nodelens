from typing import Optional, Tuple, Union

import torch


class GPUBinning:
    """GPU-accelerated binning operations using PyTorch operations."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU binning.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    @staticmethod
    @torch.jit.script
    def _compute_bin_indices_1d(data: torch.Tensor, min_val: float, max_val: float, n_bins: int) -> torch.Tensor:
        """JIT-compiled function to compute 1D bin indices."""
        # Normalize to [0, 1]
        normalized = (data - min_val) / (max_val - min_val + 1e-10)
        # Clamp to valid range
        normalized = torch.clamp(normalized, 0.0, 0.999999)
        # Convert to bin indices
        indices = (normalized * n_bins).long()
        return indices

    def fast_histogram_1d(
        self, data: torch.Tensor, n_bins: int = 256, range_min: Optional[float] = None, range_max: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute 1D histogram on GPU.

        Args:
            data: Input tensor (will be flattened)
            n_bins: Number of bins
            range_min: Minimum value (computed from data if None)
            range_max: Maximum value (computed from data if None)

        Returns:
            Histogram counts
        """
        data = data.to(self.device).flatten()

        # Determine range
        if range_min is None:
            range_min = data.min().item()
        if range_max is None:
            range_max = data.max().item()

        # Compute bin indices
        indices = self._compute_bin_indices_1d(data, range_min, range_max, n_bins)

        # Count using scatter_add
        hist = torch.zeros(n_bins, dtype=torch.float32, device=self.device)
        ones = torch.ones_like(indices, dtype=torch.float32)
        hist.scatter_add_(0, indices, ones)

        return hist

    def fast_histogram_2d(self, data_x: torch.Tensor, data_y: torch.Tensor, n_bins: Union[int, Tuple[int, int]] = 64) -> torch.Tensor:
        """
        Compute 2D histogram on GPU.

        Args:
            data_x: First dimension data
            data_y: Second dimension data
            n_bins: Number of bins (single value or tuple for each dimension)

        Returns:
            2D histogram
        """
        data_x = data_x.to(self.device).flatten()
        data_y = data_y.to(self.device).flatten()

        if isinstance(n_bins, int):
            n_bins_x = n_bins_y = n_bins
        else:
            n_bins_x, n_bins_y = n_bins

        # Compute ranges
        x_min, x_max = data_x.min().item(), data_x.max().item()
        y_min, y_max = data_y.min().item(), data_y.max().item()

        # Compute bin indices
        x_indices = self._compute_bin_indices_1d(data_x, x_min, x_max, n_bins_x)
        y_indices = self._compute_bin_indices_1d(data_y, y_min, y_max, n_bins_y)

        # Convert to linear indices
        linear_indices = y_indices * n_bins_x + x_indices

        # Count
        hist = torch.zeros(n_bins_x * n_bins_y, dtype=torch.float32, device=self.device)
        ones = torch.ones_like(linear_indices, dtype=torch.float32)
        hist.scatter_add_(0, linear_indices, ones)

        return hist.view(n_bins_y, n_bins_x)

    def mutual_information_gpu(self, x: torch.Tensor, y: torch.Tensor, n_bins: int = 64) -> float:
        """
        Compute mutual information using GPU-accelerated binning.

        Args:
            x: First variable
            y: Second variable
            n_bins: Number of bins

        Returns:
            Mutual information estimate
        """
        # Compute 2D histogram
        joint_hist = self.fast_histogram_2d(x, y, n_bins)
        joint_hist = joint_hist + 1e-10  # Avoid log(0)

        # Normalize to get joint probability
        joint_prob = joint_hist / joint_hist.sum()

        # Marginal probabilities
        p_x = joint_prob.sum(dim=0)
        p_y = joint_prob.sum(dim=1)

        # Compute MI
        log_ratio = torch.log(joint_prob / (p_x.unsqueeze(0) * p_y.unsqueeze(1)))
        mi = (joint_prob * log_ratio).sum().item()

        return float(mi)
