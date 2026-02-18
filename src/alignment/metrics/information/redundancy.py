"""
Redundancy metrics for neural network alignment analysis.

These metrics measure the redundancy between neurons or features,
indicating how much overlapping information they capture.
"""

import logging
from typing import Any, Optional

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("average_redundancy", aliases=["redundancy_gaussian"])
class AverageRedundancy(BaseMetric):
    """
    Average redundancy between neurons using Gaussian approximation.

    For each neuron, computes the average mutual information with all
    other neurons, indicating how much its information is redundant
    with the rest of the layer.
    """

    def __init__(self, min_samples: int = 2, use_correlation: bool = True, **config: Any):
        """
        Initialize the redundancy metric.

        Args:
            min_samples: Minimum samples for computation
            use_correlation: If True, use correlation; if False, use covariance
            **config: Additional configuration
        """
        super().__init__(**config)
        self.min_samples = min_samples
        self.use_correlation = use_correlation

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return False

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute average redundancy for each neuron.

        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [num_neurons, input_features]
            outputs: Not used (computed from inputs and weights)

        Returns:
            Average redundancy scores [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("AverageRedundancy requires inputs and weights")

        # Flatten if needed
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)

        batch_size = inputs.shape[0]
        num_neurons = weights.shape[0]

        if batch_size < self.min_samples:
            logger.warning(f"Redundancy: Only {batch_size} samples, returning zeros")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)

        if num_neurons <= 1:
            logger.warning("Redundancy: Need at least 2 neurons")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)

        # Compute projected outputs
        projected = torch.matmul(inputs, weights.T)  # [batch_size, num_neurons]

        # Move to CPU for large computations
        if self._should_use_cpu(projected) or num_neurons > 4096:
            projected = projected.cpu()

        # Convert to float32 for numerical stability
        projected = projected.float()

        # Normalize for correlation computation
        proj_mean = projected.mean(dim=0, keepdim=True)
        proj_std = projected.std(dim=0, keepdim=True)
        proj_std = torch.where(proj_std > 1e-12, proj_std, torch.ones_like(proj_std))
        projected_norm = (projected - proj_mean) / proj_std  # [batch_size, num_neurons]

        # For large layers, compute redundancy by sampling neighbors per neuron
        # This gives a score for EVERY neuron without creating full NxN matrix
        min(512, num_neurons - 1)  # Sample 512 neighbors per neuron

        if num_neurons > 2048:
            # Efficient stochastic approach: sample a fixed set of reference neurons
            # and compute each neuron's correlation with this reference set
            num_refs = min(1024, num_neurons)
            ref_indices = torch.randperm(num_neurons, device=projected.device)[:num_refs]

            # Compute correlation of ALL neurons with reference neurons
            # projected_norm: [batch_size, num_neurons]
            # ref_activations: [batch_size, num_refs]
            ref_activations = projected_norm[:, ref_indices]

            # Correlation matrix: [num_neurons, num_refs]
            corr_with_refs = (projected_norm.T @ ref_activations) / (batch_size - 1)

            # For self-correlations (neuron i's corr with itself in ref set), set to 0
            # This happens when neuron i is in the reference set
            for idx, ref_idx in enumerate(ref_indices):
                corr_with_refs[ref_idx, idx] = 0.0

            rho_sq = corr_with_refs**2
            rho_sq = torch.clamp(rho_sq, 0, 0.999999)

            # MI approximation for each neuron
            mi_with_refs = -0.5 * torch.log(1.0 - rho_sq)

            # Average MI with reference set (approximate redundancy)
            redundancy_scores = mi_with_refs.mean(dim=1)
        else:
            # Full correlation matrix for smaller layers
            corr_matrix = torch.matmul(projected_norm.T, projected_norm) / (batch_size - 1)
            rho_sq = corr_matrix**2
            rho_sq = torch.clamp(rho_sq, 0, 0.999999)
            mi_matrix = -0.5 * torch.log(1.0 - rho_sq)
            mi_matrix.fill_diagonal_(0)
            redundancy_scores = mi_matrix.sum(dim=1) / max(1, num_neurons - 1)

        return torch.nan_to_num(redundancy_scores.to(weights.device))


@register_metric("node_redundancy", aliases=["input_redundancy"])
class NodeRedundancy(BaseMetric):
    """
    Redundancy between input features based on correlation.

    This metric measures how redundant each input feature is with
    respect to other input features, useful for identifying
    correlated inputs.
    """

    def __init__(self, min_samples: int = 2, exclude_self: bool = True, **config: Any):
        """
        Initialize node redundancy metric.

        Args:
            min_samples: Minimum samples for correlation
            exclude_self: Whether to exclude self-correlation
            **config: Additional configuration
        """
        super().__init__(**config)
        self.min_samples = min_samples
        self.exclude_self = exclude_self

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return False

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute redundancy for each input feature.

        Args:
            inputs: Input activations [batch_size, num_features]
            weights: Not used
            outputs: Not used

        Returns:
            Redundancy scores for each input feature [num_features]
        """
        if inputs is None:
            raise ValueError("NodeRedundancy requires inputs")

        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)

        batch_size, num_features = inputs.shape

        if batch_size < self.min_samples:
            logger.warning(f"NodeRedundancy: Only {batch_size} samples")
            return torch.zeros(num_features, device=inputs.device, dtype=inputs.dtype)

        # Compute correlation matrix
        compute_device = inputs.device
        if self._should_use_cpu(inputs):
            inputs = inputs.cpu()

        # Standardize features
        inputs_mean = inputs.mean(dim=0, keepdim=True)
        inputs_std = inputs.std(dim=0, keepdim=True)
        inputs_std = torch.where(inputs_std > 1e-10, inputs_std, torch.ones_like(inputs_std))
        inputs_norm = (inputs - inputs_mean) / inputs_std

        # Correlation matrix
        corr_matrix = torch.matmul(inputs_norm.T, inputs_norm) / (batch_size - 1)

        # Take absolute correlations (strength matters, not direction)
        abs_corr = torch.abs(corr_matrix)

        # Compute average correlation with other features
        redundancy_scores = torch.zeros(num_features, device=inputs.device)

        for i in range(num_features):
            if self.exclude_self:
                # Average correlation with other features
                mask = torch.ones(num_features, dtype=torch.bool, device=corr_matrix.device)
                mask[i] = False
                if mask.sum() > 0:
                    redundancy_scores[i] = abs_corr[i, mask].mean()
            else:
                # Include self-correlation
                redundancy_scores[i] = abs_corr[i].mean()

        # Move back to original device
        if compute_device != inputs.device:
            redundancy_scores = redundancy_scores.to(compute_device)

        return torch.nan_to_num(redundancy_scores)


@register_metric("layer_redundancy")
class LayerRedundancy(BaseMetric):
    """
    Overall redundancy measure for an entire layer.

    Computes a single redundancy score for the layer based on
    the average pairwise mutual information between neurons.
    """

    def __init__(self, return_matrix: bool = False, **config: Any):
        """
        Initialize layer redundancy metric.

        Args:
            return_matrix: If True, return full redundancy matrix
            **config: Additional configuration
        """
        super().__init__(**config)
        self.return_matrix = return_matrix
        self._avg_redundancy = AverageRedundancy(**config)

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return False

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute overall layer redundancy.

        Returns:
            If return_matrix=False: scalar redundancy score
            If return_matrix=True: redundancy matrix [num_neurons, num_neurons]
        """
        # Get per-neuron redundancy scores
        neuron_redundancies = self._avg_redundancy.compute(inputs=inputs, weights=weights, outputs=outputs, **kwargs)

        if self.return_matrix:
            # Return full redundancy matrix (would need to modify avg_redundancy)
            logger.warning("Full matrix return not yet implemented, returning average")

        # Return average redundancy across all neurons
        return neuron_redundancies.mean().unsqueeze(0)
