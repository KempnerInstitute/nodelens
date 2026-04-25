"""
Eigenvector-based pruning strategy.

This module implements pruning based on PCA/eigendecomposition,
dropping neurons based on their contribution to principal components.
Neurons aligned with low-variance directions are pruned first.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..base import BasePruningStrategy, PruningConfig

logger = logging.getLogger(__name__)


class EigenvectorPruning(BasePruningStrategy):
    """
    Eigenvector-based pruning strategy.

    This strategy computes eigendecomposition of the activation covariance
    matrix and prunes neurons based on their contribution to the principal
    components. Neurons with low eigenvalue contributions are considered
    less important and pruned first.

    Two modes are supported:
    - 'low': Prune neurons aligned with low-variance directions (default)
    - 'high': Prune neurons aligned with high-variance directions (ablation)

    Examples:
        >>> from nodelens.pruning.strategies import EigenvectorPruning
        >>> from nodelens.pruning import PruningConfig
        >>>
        >>> config = PruningConfig(amount=0.5, structured=True, pruning_mode='low')
        >>> strategy = EigenvectorPruning(config=config)
        >>>
        >>> # Compute scores from activations
        >>> scores = strategy.compute_importance_scores(module, inputs=activations)
        >>> strategy.prune(module, inputs=activations)
    """

    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        variance_threshold: float = 0.99,
        use_correlation: bool = False,
        regularization: float = 1e-6,
    ):
        """
        Initialize eigenvector pruning.

        Args:
            config: Pruning configuration
            variance_threshold: Fraction of variance to explain (for ranking)
            use_correlation: If True, use correlation matrix instead of covariance
            regularization: Small value added to diagonal for numerical stability
        """
        super().__init__(config)
        self.variance_threshold = variance_threshold
        self.use_correlation = use_correlation
        self.regularization = regularization

        # Force structured pruning (eigenvector pruning is inherently structured)
        if self.config:
            self.config.structured = True

    def _compute_activation_covariance(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute covariance matrix and mean of activations.

        Args:
            activations: Input activations [batch, neurons, ...] or [batch, neurons]

        Returns:
            Tuple of (covariance matrix, mean vector)
        """
        # Flatten spatial dimensions if present (for conv layers)
        if activations.dim() > 2:
            # [B, C, H, W] -> [B*H*W, C]
            batch_size, channels = activations.shape[:2]
            activations = activations.permute(0, 2, 3, 1).reshape(-1, channels)
        elif activations.dim() == 2:
            # [B, N] is already correct
            pass
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")

        # Compute mean
        mean = activations.mean(dim=0)

        # Center activations
        centered = activations - mean

        # Compute covariance: (X^T @ X) / (n-1)
        n_samples = centered.shape[0]
        cov = (centered.T @ centered) / max(n_samples - 1, 1)

        # Add regularization for numerical stability
        cov = cov + self.regularization * torch.eye(cov.shape[0], device=cov.device)

        # Optionally convert to correlation matrix
        if self.use_correlation:
            std = torch.sqrt(torch.diag(cov))
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            cov = cov / (std.unsqueeze(0) * std.unsqueeze(1))

        return cov, mean

    def _compute_eigendecomposition(self, cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigendecomposition of covariance matrix.

        Args:
            cov: Covariance matrix [N, N]

        Returns:
            Tuple of (eigenvalues, eigenvectors) sorted by eigenvalue descending
        """
        # Compute eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        except Exception as e:
            logger.warning(f"Eigendecomposition failed: {e}. Using SVD fallback.")
            # SVD fallback for numerical issues
            U, S, Vh = torch.linalg.svd(cov)
            eigenvalues = S
            eigenvectors = U

        # Sort by eigenvalue descending (largest first)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors

    def compute_importance_scores(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute neuron importance based on eigenvalue contributions.

        Each neuron's importance is based on its contribution to the
        variance explained by the principal components.

        Args:
            module: Module to compute scores for
            inputs: Input activations [batch, neurons, ...]

        Returns:
            Importance scores per neuron (higher = more important)
        """
        if inputs is None:
            raise ValueError("EigenvectorPruning requires input activations")

        # Get number of neurons from module
        if hasattr(module, "weight"):
            module.weight.shape[0]
        else:
            raise ValueError("Module must have weight attribute")

        # Compute covariance of activations
        cov, _ = self._compute_activation_covariance(inputs)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = self._compute_eigendecomposition(cov)

        # Compute importance scores for each neuron
        # Score = sum of (eigenvalue * squared loading) for each neuron
        # This measures how much variance each neuron contributes to

        # Normalize eigenvalues to get variance explained
        total_variance = eigenvalues.sum()
        if total_variance > 0:
            variance_explained = eigenvalues / total_variance
        else:
            variance_explained = torch.ones_like(eigenvalues) / len(eigenvalues)

        # Compute neuron scores
        # For each neuron i: score_i = sum_j(lambda_j * v_ij^2)
        # where lambda_j is the j-th eigenvalue and v_ij is the loading
        loadings_squared = eigenvectors**2  # [neurons, components]
        neuron_scores = (loadings_squared * variance_explained.unsqueeze(0)).sum(dim=1)

        # Ensure scores are positive and on correct device
        neuron_scores = torch.clamp(neuron_scores, min=0)

        if neuron_scores.device != module.weight.device:
            neuron_scores = neuron_scores.to(module.weight.device)

        logger.debug(f"Eigenvalue importance: min={neuron_scores.min():.4f}, " f"max={neuron_scores.max():.4f}, mean={neuron_scores.mean():.4f}")

        return neuron_scores

    def get_variance_explained(self, module: nn.Module, inputs: torch.Tensor, n_components: Optional[int] = None) -> Tuple[torch.Tensor, float]:
        """
        Get cumulative variance explained by top N components.

        Args:
            module: Module to analyze
            inputs: Input activations
            n_components: Number of components (None = all)

        Returns:
            Tuple of (cumulative variance ratios, total variance)
        """
        cov, _ = self._compute_activation_covariance(inputs)
        eigenvalues, _ = self._compute_eigendecomposition(cov)

        total_variance = eigenvalues.sum().item()
        cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance

        if n_components is not None:
            cumulative = cumulative[:n_components]

        return cumulative, total_variance

    def prune(self, module: nn.Module, inputs: Optional[torch.Tensor] = None, amount: Optional[float] = None, **kwargs) -> torch.Tensor:
        """
        Prune module based on eigenvector importance.

        Args:
            module: Module to prune
            inputs: Input activations (required)
            amount: Fraction to prune (overrides config)

        Returns:
            Pruning mask
        """
        if inputs is None:
            raise ValueError("EigenvectorPruning requires input activations")

        amount = amount if amount is not None else self.config.amount

        # Compute importance scores
        scores = self.compute_importance_scores(module, inputs)

        # Create structured mask
        n_neurons = scores.numel()
        k = int(amount * n_neurons)

        if k == 0:
            mask = torch.ones_like(module.weight)
        else:
            keep_mask = torch.ones(n_neurons, dtype=torch.bool, device=scores.device)

            if self.config.pruning_mode == "low":
                # Prune neurons with LOWEST eigenvalue contribution
                _, indices_to_prune = torch.topk(scores, k, largest=False)
            else:  # 'high' mode
                # Prune neurons with HIGHEST eigenvalue contribution (ablation)
                _, indices_to_prune = torch.topk(scores, k, largest=True)

            keep_mask[indices_to_prune] = False

            # Expand to weight dimensions
            if len(module.weight.shape) == 2:  # Linear
                mask = keep_mask.unsqueeze(1).expand_as(module.weight).float()
            else:  # Conv
                mask = keep_mask.view(-1, 1, 1, 1).expand_as(module.weight).float()

        # Apply pruning
        self.apply_pruning(module, mask)

        return mask
