"""
Synergy metric using Gaussian approximation and MMI redundancy.

Computes target-conditional synergy between neurons using:
    S_MMI(Z; Y_i, Y_j) = I(Z; Y_i, Y_j) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j))

where Z is a discrete target and Y_i, Y_j are continuous neuron outputs.
"""

import logging
from typing import Any, Optional

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("synergy_gaussian_mmi")
class SynergyGaussianMMI(BaseMetric):
    """
    Compute per-neuron synergy using Gaussian MI and MMI redundancy.

    For each neuron, computes average synergy with K sampled partner neurons
    relative to a discrete target Z.

    Synergy measures information that emerges only from the joint outputs,
    using the MMI (Minimum Mutual Information) redundancy axiom:
        S_MMI = I(Z; Y_i, Y_j) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j))

    Example:
        >>> synergy_metric = SynergyGaussianMMI(num_pairs=10)
        >>> synergy = synergy_metric.compute(
        ...     inputs=inputs,
        ...     weights=weights,
        ...     targets=labels
        ... )
        >>> print(synergy.shape)  # [num_neurons]
    """

    def __init__(self, num_pairs: int = 10, sampling_strategy: str = "random", **config: Any):
        """
        Initialize synergy metric.

        Args:
            num_pairs: Number of partner neurons to sample per neuron
            sampling_strategy: How to sample pairs ('random', 'nearest', 'all')
            **config: Additional configuration
        """
        super().__init__(**config)
        self.num_pairs = num_pairs
        self.sampling_strategy = sampling_strategy

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return True  # We'll use outputs if provided, else compute from inputs @ weights.T

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute per-neuron synergy scores.

        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [num_neurons, input_features]
            outputs: Layer outputs [batch_size, num_neurons] (optional)
            targets: Target labels [batch_size] (required)
            **kwargs: Additional parameters

        Returns:
            Per-neuron synergy scores [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("SynergyGaussianMMI requires inputs and weights")

        if targets is None:
            raise ValueError("SynergyGaussianMMI requires targets")

        # Flatten if needed
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()

        # Compute outputs if not provided
        if outputs is None:
            outputs = inputs @ weights.T  # [batch_size, num_neurons]

        num_neurons = weights.shape[0]

        # Compute synergy for each neuron
        synergy = torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)

        for i in range(num_neurons):
            # Sample partner neurons
            partner_indices = self._sample_partners(i, num_neurons)

            if len(partner_indices) == 0:
                continue

            # Compute synergy with each partner
            synergy_values = []
            for j in partner_indices:
                s = self._compute_pairwise_synergy(outputs[:, i], outputs[:, j], targets)
                synergy_values.append(s)

            # Average synergy with partners
            if synergy_values:
                synergy[i] = torch.stack(synergy_values).mean()

        return synergy

    def _compute_pairwise_synergy(self, y_i: torch.Tensor, y_j: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute synergy between two neurons relative to target.

        Args:
            y_i: Output of neuron i [batch_size]
            y_j: Output of neuron j [batch_size]
            targets: Target labels [batch_size]

        Returns:
            Synergy S_MMI(Z; Y_i, Y_j)
        """
        # Compute individual MIs
        mi_i = self._gaussian_mi_categorical(y_i, targets)
        mi_j = self._gaussian_mi_categorical(y_j, targets)

        # Compute joint MI
        y_joint = torch.stack([y_i, y_j], dim=1)  # [batch_size, 2]
        mi_joint = self._gaussian_mi_categorical_multivariate(y_joint, targets)

        # MMI redundancy
        redundancy_mmi = torch.min(mi_i, mi_j)

        # Synergy
        synergy = mi_joint - mi_i - mi_j + redundancy_mmi

        return synergy

    def _gaussian_mi_categorical(self, y: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute MI between continuous y and categorical z using Gaussian approximation.

        I(Z; Y) = H(Y) - H(Y|Z)
                = 0.5 * log(2πe σ²_Y) - Σ p(z) * 0.5 * log(2πe σ²_{Y|z})

        Args:
            y: Continuous variable [batch_size]
            z: Categorical variable [batch_size]
            eps: Small value for numerical stability

        Returns:
            Mutual information (scalar)
        """
        # Overall variance
        var_y = torch.var(y, unbiased=True) + eps

        # Conditional variances
        classes = torch.unique(z)
        conditional_entropy = 0.0

        for c in classes:
            mask = z == c
            n_c = mask.sum()

            if n_c < 2:
                continue

            y_c = y[mask]
            var_y_c = torch.var(y_c, unbiased=True) + eps

            # Weight by class probability
            p_c = n_c.float() / len(z)
            conditional_entropy += p_c * 0.5 * torch.log(2 * torch.pi * torch.e * var_y_c)

        # MI = H(Y) - H(Y|Z)
        marginal_entropy = 0.5 * torch.log(2 * torch.pi * torch.e * var_y)
        mi = marginal_entropy - conditional_entropy

        # MI should be non-negative
        mi = torch.clamp(mi, min=0.0)

        return mi

    def _gaussian_mi_categorical_multivariate(self, y: torch.Tensor, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Compute MI between multivariate continuous y and categorical z.

        I(Z; Y) = 0.5 * log(det(Σ_Y) / det(Σ_{Y|Z}))

        Args:
            y: Continuous variables [batch_size, dim]
            z: Categorical variable [batch_size]
            eps: Regularization for covariance

        Returns:
            Mutual information (scalar)
        """
        batch_size, dim = y.shape

        # Overall covariance
        y_centered = y - y.mean(dim=0, keepdim=True)
        cov_y = (y_centered.T @ y_centered) / max(1, batch_size - 1)
        cov_y = cov_y + eps * torch.eye(dim, device=y.device, dtype=y.dtype)

        # Conditional covariances (weighted average)
        classes = torch.unique(z)
        cov_y_given_z = torch.zeros_like(cov_y)
        total_weight = 0.0

        for c in classes:
            mask = z == c
            n_c = mask.sum()

            if n_c < dim + 1:  # Need enough samples
                continue

            y_c = y[mask]
            y_c_centered = y_c - y_c.mean(dim=0, keepdim=True)
            cov_c = (y_c_centered.T @ y_c_centered) / max(1, n_c - 1)
            cov_c = cov_c + eps * torch.eye(dim, device=y.device, dtype=y.dtype)

            # Weight by class probability
            weight = n_c.float()
            cov_y_given_z += cov_c * weight
            total_weight += weight

        if total_weight > 0:
            cov_y_given_z = cov_y_given_z / total_weight
        else:
            # Fallback: return zero MI
            return torch.tensor(0.0, device=y.device, dtype=y.dtype)

        # MI via determinants
        # Add small regularization for numerical stability
        det_y = torch.det(cov_y)
        det_y_given_z = torch.det(cov_y_given_z)

        if det_y <= 0 or det_y_given_z <= 0:
            # Numerical issues, return zero
            return torch.tensor(0.0, device=y.device, dtype=y.dtype)

        mi = 0.5 * torch.log(det_y / (det_y_given_z + eps))
        mi = torch.clamp(mi, min=0.0)

        return mi

    def _sample_partners(self, neuron_idx: int, num_neurons: int) -> torch.Tensor:
        """
        Sample partner neurons for synergy computation.

        Args:
            neuron_idx: Index of current neuron
            num_neurons: Total number of neurons

        Returns:
            Indices of partner neurons
        """
        # Exclude self
        available = list(range(num_neurons))
        available.remove(neuron_idx)

        if self.sampling_strategy == "all":
            return torch.tensor(available, dtype=torch.long)

        elif self.sampling_strategy == "random":
            num_to_sample = min(self.num_pairs, len(available))
            if num_to_sample == 0:
                return torch.tensor([], dtype=torch.long)

            indices = torch.randperm(len(available))[:num_to_sample]
            return torch.tensor([available[i] for i in indices], dtype=torch.long)

        elif self.sampling_strategy == "nearest":
            num_to_sample = min(self.num_pairs, len(available))
            if num_to_sample == 0:
                return torch.tensor([], dtype=torch.long)

            distances = torch.abs(torch.tensor(available) - neuron_idx)
            _, nearest_indices = torch.topk(distances, num_to_sample, largest=False)
            return torch.tensor([available[i] for i in nearest_indices], dtype=torch.long)

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
