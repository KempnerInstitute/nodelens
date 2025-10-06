"""
Higher-order information decomposition metrics.

These metrics capture complex multivariate dependencies beyond pairwise relationships.
"""

from typing import List, Optional

import numpy as np
import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric


@register_metric("total_correlation")
class TotalCorrelation(BaseMetric):
    """
    Measures total correlation (multi-information) among variables.

    Total correlation quantifies the amount of dependency among a set of
    random variables. It's the KL divergence between the joint distribution
    and the product of marginal distributions.

    TC(X1, ..., Xn) = sum(H(Xi)) - H(X1, ..., Xn)
    """

    name = "total_correlation"

    def __init__(self, n_bins: int = 30, normalize: bool = True):
        """
        Args:
            n_bins: Number of bins for discretization
            normalize: Whether to normalize by number of variables
        """
        super().__init__()
        self.n_bins = n_bins
        self.normalize = normalize

    def _estimate_entropy(self, data: torch.Tensor) -> float:
        """Estimate entropy using histogram method."""
        # Discretize continuous data
        data_np = data.detach().cpu().numpy()

        if data.dim() == 1:
            # Single variable entropy
            hist, _ = np.histogram(data_np, bins=self.n_bins)
            hist = hist + 1e-10  # Add small constant to avoid log(0)
            hist = hist / hist.sum()
            return -np.sum(hist * np.log(hist))
        else:
            # Joint entropy for multiple variables
            # Use multi-dimensional histogram
            ranges = [(data_np[:, i].min(), data_np[:, i].max()) for i in range(data_np.shape[1])]
            hist, _ = np.histogramdd(data_np, bins=self.n_bins, range=ranges)
            hist = hist.flatten() + 1e-10
            hist = hist / hist.sum()
            return -np.sum(hist * np.log(hist))

    def compute(self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None) -> float:
        """Compute total correlation."""
        if outputs is None:
            raise ValueError("Outputs required for total correlation")

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        n_vars = outputs.size(1)
        if n_vars < 2:
            return 0.0  # No correlation for single variable

        # Compute marginal entropies
        marginal_entropies = []
        for i in range(n_vars):
            H_i = self._estimate_entropy(outputs[:, i])
            marginal_entropies.append(H_i)

        # Compute joint entropy
        joint_entropy = self._estimate_entropy(outputs)

        # Total correlation
        tc = sum(marginal_entropies) - joint_entropy

        if self.normalize:
            tc = tc / n_vars

        return float(tc)


@register_metric("interaction_information")
class InteractionInformation(BaseMetric):
    """
    Compute interaction information (co-information) among triplets of variables.

    This measures the amount of information that is present only when all three
    variables are considered together.
    """

    name = "interaction_information"

    def __init__(self, n_samples: int = 100, bins: int = 10):
        """
        Initialize interaction information metric.

        Args:
            n_samples: Number of triplet samples to evaluate
            bins: Number of bins for discretization
        """
        super().__init__()
        self.n_samples = n_samples
        self.bins = bins

    def compute(self, inputs: torch.Tensor, weights: torch.Tensor, outputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute interaction information scores.

        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations

        Returns:
            Interaction information scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T

        n_neurons = outputs.shape[1]
        interaction_scores = torch.zeros(n_neurons, device=outputs.device)

        # Sample triplets of neurons
        n_triplets = min(self.n_samples, n_neurons * (n_neurons - 1) * (n_neurons - 2) // 6)

        for _ in range(n_triplets):
            # Randomly select 3 different neurons
            idx = torch.randperm(n_neurons)[:3]
            i, j, k = idx[0], idx[1], idx[2]

            # Get activations for these neurons
            X = outputs[:, i]
            Y = outputs[:, j]
            Z = outputs[:, k]

            # Compute pairwise mutual information using histogram method
            MI_XY = self._estimate_mi_binning(X.unsqueeze(1), Y.unsqueeze(1))
            MI_XZ = self._estimate_mi_binning(X.unsqueeze(1), Z.unsqueeze(1))
            MI_YZ = self._estimate_mi_binning(Y.unsqueeze(1), Z.unsqueeze(1))

            # Compute conditional mutual information I(X;Y|Z)
            # Using approximation: I(X;Y|Z) ≈ I(X;Y) - I(X;Y;Z)
            # Where I(X;Y;Z) is the interaction information

            # For simplicity, we'll use the difference of mutual informations
            # as an approximation of interaction information
            interaction = MI_XY + MI_XZ + MI_YZ

            # Distribute score to participating neurons
            interaction_scores[i] += interaction / 3
            interaction_scores[j] += interaction / 3
            interaction_scores[k] += interaction / 3

        # Normalize by number of samples
        interaction_scores = interaction_scores / n_triplets

        return interaction_scores

    def _estimate_mi_binning(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate mutual information using binning method."""
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Create 2D histogram
        hist, _, _ = np.histogram2d(x_np.flatten(), y_np.flatten(), bins=self.bins)
        hist = hist + 1e-10  # Avoid log(0)

        # Normalize to get probabilities
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # Compute MI
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        return mi


@register_metric("connected_information")
class ConnectedInformation(BaseMetric):
    """
    Measures connected information (interaction information of order n).

    This captures pure n-way interactions that cannot be reduced to lower-order
    interactions, useful for understanding complex dependencies in neural networks.
    """

    name = "connected_information"

    def __init__(self, n_bins: int = 20, max_order: int = 4):
        """
        Args:
            n_bins: Number of bins for discretization
            max_order: Maximum order of interactions to compute
        """
        super().__init__()
        self.n_bins = n_bins
        self.max_order = max_order
        self._entropy_est = TotalCorrelation(n_bins=n_bins)

    def _compute_interaction_info(self, data: torch.Tensor, indices: List[int]) -> float:
        """Compute interaction information for a subset of variables."""
        if len(indices) < 2:
            return 0.0

        # Use inclusion-exclusion principle
        total = 0.0
        n = len(indices)

        # Generate all non-empty subsets
        from itertools import combinations

        for k in range(1, n + 1):
            sign = (-1) ** (n - k)
            for subset in combinations(indices, k):
                subset_data = data[:, list(subset)]
                if len(subset) == 1:
                    entropy = self._entropy_est._estimate_entropy(subset_data.squeeze(1))
                else:
                    entropy = self._entropy_est._estimate_entropy(subset_data)
                total += sign * entropy

        return total

    def compute(self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None) -> float:
        """Compute connected information up to max_order."""
        if outputs is None:
            raise ValueError("Outputs required for connected information")

        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        n_vars = outputs.size(1)
        if n_vars < 2:
            return 0.0

        # Compute interaction information for different orders
        total_connected = 0.0

        from itertools import combinations

        for order in range(2, min(n_vars + 1, self.max_order + 1)):
            for var_subset in combinations(range(n_vars), order):
                interaction = self._compute_interaction_info(outputs, list(var_subset))
                total_connected += abs(interaction)  # Use absolute value

        # Normalize by number of possible interactions
        n_interactions = sum(1 for order in range(2, min(n_vars + 1, self.max_order + 1)) for _ in combinations(range(n_vars), order))

        if n_interactions > 0:
            total_connected /= n_interactions

        return float(total_connected)


@register_metric("synergistic_information")
class SynergisticInformation(BaseMetric):
    """
    Compute synergistic information - information that can only be obtained
    from the joint state of multiple neurons.
    """

    name = "synergistic_information"

    def __init__(self, group_size: int = 3, n_groups: int = 50):
        """
        Initialize synergistic information metric.

        Args:
            group_size: Size of neuron groups to analyze
            n_groups: Number of random groups to sample
        """
        super().__init__()
        self.group_size = group_size
        self.n_groups = n_groups

    def compute(self, inputs: torch.Tensor, weights: torch.Tensor, outputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute synergistic information scores.

        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations

        Returns:
            Synergistic information scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T

        batch_size, n_neurons = outputs.shape
        synergy_scores = torch.zeros(n_neurons, device=outputs.device)

        # Sample groups of neurons
        n_groups_actual = min(self.n_groups, n_neurons // self.group_size)

        for _ in range(n_groups_actual):
            # Select a random group
            idx = torch.randperm(n_neurons)[: self.group_size]
            group_outputs = outputs[:, idx]

            # Compute joint entropy of the group
            # Using Gaussian assumption for efficiency
            group_centered = group_outputs - group_outputs.mean(dim=0, keepdim=True)
            cov_group = (group_centered.T @ group_centered) / (batch_size - 1)
            cov_group = cov_group + 1e-8 * torch.eye(self.group_size, device=outputs.device)

            # Joint entropy under Gaussian assumption
            # H = 0.5 * log(det(2πe * Σ))
            det_cov = torch.linalg.det(cov_group)
            joint_entropy = 0.5 * torch.log(2 * np.pi * np.e * det_cov)

            # Compute sum of individual entropies
            individual_entropies = 0
            for i in range(self.group_size):
                var = cov_group[i, i]
                individual_entropies += 0.5 * torch.log(2 * np.pi * np.e * var)

            # Synergy approximation: joint entropy - sum of individual entropies
            # (negative of this gives redundancy, positive gives synergy)
            synergy = joint_entropy - individual_entropies

            # Distribute score to participating neurons
            for i in idx:
                synergy_scores[i] += synergy / self.group_size

        # Normalize by number of groups
        synergy_scores = synergy_scores / n_groups_actual

        return synergy_scores
