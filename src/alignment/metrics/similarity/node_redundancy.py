"""
Node redundancy metric for measuring input feature correlations.
"""

import logging
from typing import Optional

import torch

from ...core.base import BaseMetric

logger = logging.getLogger(__name__)


class NodeRedundancy(BaseMetric):
    """
    Compute redundancy between input features based on correlation of activations.

    This represents feature redundancy rather than per-output-node scores.
    For each feature, it computes the average absolute correlation with other features.
    """

    name = "node_redundancy"
    requires_weights = False
    requires_inputs = True
    requires_outputs = False

    def __init__(self, force_cpu: bool = False):
        """
        Initialize the node redundancy metric.

        Args:
            force_cpu: Whether to force CPU computation for large operations
        """
        self.force_cpu = force_cpu

    @torch.no_grad()
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute node redundancy scores.

        Args:
            inputs: Input activations [batch_size, num_features]
            weights: Not used
            outputs: Not used
            **kwargs: Additional arguments

        Returns:
            Redundancy scores per input feature [num_features]
        """
        if inputs is None:
            raise ValueError("Node redundancy requires inputs")

        # Handle different input dimensions
        if inputs.ndim != 2:
            if inputs.ndim > 2:
                # Flatten spatial dimensions
                inputs = inputs.flatten(start_dim=1)
            else:
                logger.warning(f"Input has unexpected shape: {inputs.shape}")
                return torch.zeros(1, device=inputs.device)

        batch_size, num_features = inputs.shape

        # Need at least 2 samples to compute correlation
        if batch_size < 2:
            logger.warning(f"Need at least 2 samples for correlation, got {batch_size}")
            return torch.zeros(num_features, device=inputs.device)

        # Single feature case
        if num_features == 1:
            return torch.zeros(1, device=inputs.device)

        redundancy_scores = torch.zeros(num_features, device=inputs.device)

        try:
            # Compute correlation matrix
            corr_matrix = self._compute_correlation(inputs)

            # Take absolute values (we care about strength, not direction)
            abs_corr = torch.abs(corr_matrix)

            # For each feature, compute average correlation with other features
            for i in range(num_features):
                # Exclude self-correlation (always 1.0)
                mask = torch.ones(num_features, dtype=torch.bool, device=inputs.device)
                mask[i] = False

                if mask.sum() > 0:
                    redundancy_scores[i] = abs_corr[i, mask].mean()

        except Exception as e:
            logger.error(f"Error computing node redundancy: {e}")
            return torch.zeros(num_features, device=inputs.device)

        return torch.nan_to_num(redundancy_scores, nan=0.0)

    def _compute_correlation(self, X: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix."""
        device = X.device

        # Move to CPU if requested and tensor is large
        if self.force_cpu and X.is_cuda and X.numel() > 1e6:
            X = X.cpu()

        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)

        # Compute covariance
        cov = torch.matmul(X_centered.T, X_centered) / (X.size(0) - 1)

        # Compute standard deviations
        std = torch.sqrt(torch.diag(cov) + 1e-10)

        # Compute correlation
        outer_std = torch.outer(std, std)
        corr = torch.where(outer_std > 1e-10, cov / outer_std, torch.zeros_like(cov))

        # Move back to original device
        if corr.device != device:
            corr = corr.to(device)

        return corr
