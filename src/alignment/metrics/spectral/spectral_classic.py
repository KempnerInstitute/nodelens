"""
Classic spectral alignment metrics based on eigenvalue analysis.
These metrics were originally in metrics/spectral.py
"""

from typing import Optional

import torch
import torch.linalg as LA

from ...core.base import BaseMetric
from ...core.registry import register_metric


@register_metric("spectral_alignment")
class SpectralAlignment(BaseMetric):
    """
    Compute spectral alignment based on eigenvalue decomposition of the covariance matrix.

    This metric analyzes how well neuron activations align with the principal components
    of the input distribution.
    """

    name = "spectral_alignment"

    def __init__(
        self,
        n_components: Optional[int] = None,
        normalize: bool = True,
        epsilon: float = 1e-8
    ):
        """
        Initialize spectral alignment metric.

        Args:
            n_components: Number of top components to consider (None for all)
            normalize: Whether to normalize by total variance
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.n_components = n_components
        self.normalize = normalize
        self.epsilon = epsilon

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute spectral alignment scores.

        Args:
            inputs: Input activations (batch_size, input_dim)
            weights: Weight matrix (output_dim, input_dim)
            outputs: Output activations (optional)

        Returns:
            Alignment scores for each neuron
        """
        batch_size, input_dim = inputs.shape
        output_dim, _ = weights.shape

        # Center the inputs
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)

        # Compute input covariance matrix
        cov_matrix = (inputs_centered.T @ inputs_centered) / (batch_size - 1)
        cov_matrix = cov_matrix + self.epsilon * torch.eye(input_dim, device=inputs.device)

        # Compute eigenvalue decomposition
        eigenvalues, eigenvectors = LA.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top components if specified
        if self.n_components is not None:
            n_comp = min(self.n_components, input_dim)
            eigenvalues = eigenvalues[:n_comp]
            eigenvectors = eigenvectors[:, :n_comp]

        # Compute alignment of each weight vector with principal components
        # Weight vectors projected onto principal components
        projections = weights @ eigenvectors  # (output_dim, n_components)

        # Weighted by eigenvalues (importance of each component)
        weighted_projections = projections * eigenvalues.unsqueeze(0)

        # Compute alignment score for each neuron
        alignment_scores = weighted_projections.norm(dim=1)

        if self.normalize:
            # Normalize by weight norm and total variance
            weight_norms = weights.norm(dim=1)
            total_variance = eigenvalues.sum()
            alignment_scores = alignment_scores / (weight_norms * total_variance.sqrt() + self.epsilon)

        return alignment_scores


@register_metric("spectral_norm_ratio")
class SpectralNormRatio(BaseMetric):
    """
    Compute the ratio of spectral norm to Frobenius norm for weight matrices.

    This metric indicates how concentrated the weight matrix is along its principal direction.
    """

    name = "spectral_norm_ratio"

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize spectral norm ratio metric.

        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute spectral norm ratio for each layer.

        Args:
            inputs: Input activations (not used, included for interface compatibility)
            weights: Weight matrix (output_dim, input_dim)
            outputs: Output activations (not used)

        Returns:
            Spectral norm ratio (single value per layer)
        """
        # Compute spectral norm (largest singular value)
        spectral_norm = LA.matrix_norm(weights, ord=2)

        # Compute Frobenius norm
        frobenius_norm = weights.norm(p='fro')

        # Compute ratio
        ratio = spectral_norm / (frobenius_norm + self.epsilon)

        # Return as tensor with one value per neuron (repeated)
        return ratio.repeat(weights.shape[0])


@register_metric("eigenvalue_entropy")
class EigenvalueEntropy(BaseMetric):
    """
    Compute the entropy of eigenvalue distribution of the Gram matrix.

    This metric measures how evenly distributed the neuron activations are
    across different modes of variation.
    """

    name = "eigenvalue_entropy"

    def __init__(self, temperature: float = 1.0, epsilon: float = 1e-8):
        """
        Initialize eigenvalue entropy metric.

        Args:
            temperature: Temperature parameter for softmax
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute eigenvalue entropy for neuron activations.

        Args:
            inputs: Input activations (batch_size, input_dim)
            weights: Weight matrix (output_dim, input_dim)
            outputs: Output activations (optional)

        Returns:
            Entropy values for each neuron
        """
        batch_size = inputs.shape[0]
        output_dim = weights.shape[0]

        # Compute neuron activations if not provided
        if outputs is None:
            outputs = inputs @ weights.T

        entropies = []

        for i in range(output_dim):
            # Get activations for this neuron
            neuron_acts = outputs[:, i].unsqueeze(1)

            # Compute Gram matrix
            gram = neuron_acts @ neuron_acts.T

            # Add small diagonal for stability
            gram = gram + self.epsilon * torch.eye(batch_size, device=gram.device)

            # Compute eigenvalues
            eigenvalues = LA.eigvalsh(gram)
            eigenvalues = eigenvalues[eigenvalues > self.epsilon]

            # Normalize eigenvalues
            eigenvalues = eigenvalues / eigenvalues.sum()

            # Apply temperature scaling
            eigenvalues = eigenvalues / self.temperature

            # Compute entropy
            entropy = -(eigenvalues * eigenvalues.log()).sum()
            entropies.append(entropy)

        return torch.stack(entropies)


@register_metric("spectral_clustering_score")
class SpectralClusteringScore(BaseMetric):
    """
    Compute how well neurons cluster based on spectral analysis of their similarity matrix.
    """

    name = "spectral_clustering_score"

    def __init__(
        self,
        n_clusters: int = 5,
        similarity_type: str = "correlation",
        epsilon: float = 1e-8
    ):
        """
        Initialize spectral clustering score.

        Args:
            n_clusters: Number of clusters to form
            similarity_type: Type of similarity ('correlation', 'cosine', 'rbf')
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.similarity_type = similarity_type
        self.epsilon = epsilon

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute spectral clustering scores.

        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations

        Returns:
            Clustering quality scores for each neuron
        """
        output_dim = weights.shape[0]

        # Compute similarity matrix between neurons
        if self.similarity_type == "correlation":
            # Normalize weights
            weights_norm = weights - weights.mean(dim=1, keepdim=True)
            weights_std = weights_norm.std(dim=1, keepdim=True)
            weights_norm = weights_norm / (weights_std + self.epsilon)
            similarity = weights_norm @ weights_norm.T / weights.shape[1]

        elif self.similarity_type == "cosine":
            weights_norm = weights / (weights.norm(dim=1, keepdim=True) + self.epsilon)
            similarity = weights_norm @ weights_norm.T

        elif self.similarity_type == "rbf":
            # RBF kernel similarity
            dist_matrix = torch.cdist(weights, weights, p=2)
            sigma = dist_matrix.mean()
            similarity = torch.exp(-dist_matrix**2 / (2 * sigma**2))

        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

        # Compute degree matrix
        degree = similarity.sum(dim=1)

        # Compute normalized Laplacian
        D_sqrt_inv = torch.diag(1.0 / (degree.sqrt() + self.epsilon))
        L_norm = torch.eye(output_dim, device=weights.device) - D_sqrt_inv @ similarity @ D_sqrt_inv

        # Compute eigenvalues of normalized Laplacian
        eigenvalues = LA.eigvalsh(L_norm)

        # Sort eigenvalues
        eigenvalues = eigenvalues.sort()[0]

        # Compute spectral gap (difference between k-th and (k+1)-th eigenvalue)
        if self.n_clusters < output_dim:
            spectral_gap = eigenvalues[self.n_clusters] - eigenvalues[self.n_clusters - 1]
        else:
            spectral_gap = eigenvalues[-1] - eigenvalues[-2]

        # Higher spectral gap indicates better clustering
        # Return as score for each neuron (can be refined to per-neuron scores)
        scores = torch.full((output_dim,), spectral_gap.item(), device=weights.device)

        return scores
