"""
Robust covariance estimation with shrinkage methods.

This module provides stable covariance matrix estimation for small samples
using shrinkage methods like Ledoit-Wolf.
"""

import logging
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)


class CovarianceEstimator:
    """
    Robust covariance estimation with various shrinkage methods.

    Shrinkage improves covariance estimates on small samples by
    regularizing towards a structured target (e.g., scaled identity).

    Methods:
    - 'none': Standard sample covariance
    - 'diagonal': Add ε to diagonal
    - 'ledoit_wolf': Optimal shrinkage towards scaled identity
    - 'oracle_approximating_shrinkage': Oracle approximating shrinkage (OAS)

    Example:
        >>> estimator = CovarianceEstimator(method='ledoit_wolf')
        >>> cov = estimator.estimate(X)  # More stable than torch.cov
    """

    def __init__(self, method: Literal["none", "diagonal", "ledoit_wolf", "oas"] = "ledoit_wolf", regularization: float = 1e-6):
        """
        Initialize covariance estimator.

        Args:
            method: Shrinkage method to use
            regularization: Baseline regularization (added to all methods)
        """
        self.method = method
        self.regularization = regularization

    def estimate(self, X: torch.Tensor, center: bool = True) -> torch.Tensor:
        """
        Estimate covariance matrix with shrinkage.

        Args:
            X: Data matrix [n_samples, n_features]
            center: Whether to center the data

        Returns:
            Covariance matrix [n_features, n_features]
        """
        if X.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {X.shape}")

        n_samples, n_features = X.shape

        # Center data
        if center:
            X_centered = X - X.mean(dim=0, keepdim=True)
        else:
            X_centered = X

        # Sample covariance
        cov_sample = (X_centered.T @ X_centered) / max(1, n_samples - 1)

        # Apply shrinkage method
        if self.method == "none":
            cov = cov_sample

        elif self.method == "diagonal":
            cov = cov_sample + self.regularization * torch.eye(n_features, device=X.device, dtype=X.dtype)

        elif self.method == "ledoit_wolf":
            cov = self._ledoit_wolf_shrinkage(X_centered, cov_sample)

        elif self.method == "oas":
            cov = self._oas_shrinkage(X_centered, cov_sample)

        else:
            raise ValueError(f"Unknown shrinkage method: {self.method}")

        # Always add baseline regularization
        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(n_features, device=X.device, dtype=X.dtype)

        return cov

    def _ledoit_wolf_shrinkage(self, X_centered: torch.Tensor, cov_sample: torch.Tensor) -> torch.Tensor:
        """
        Ledoit-Wolf shrinkage towards scaled identity.

        Σ_shrunk = (1-λ)·Σ_sample + λ·μ·I
        where λ is chosen to minimize MSE, μ = tr(Σ_sample)/d

        Reference: Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix"
        """
        n_samples, n_features = X_centered.shape

        # Target: scaled identity
        mu = torch.trace(cov_sample) / n_features
        target = mu * torch.eye(n_features, device=X_centered.device, dtype=X_centered.dtype)

        # Estimate optimal shrinkage intensity (simplified formula)
        # Full formula is complex; we use a practical approximation
        if n_samples >= n_features:
            # Oracle shrinkage intensity (simplified)
            delta = cov_sample - target
            delta_norm_sq = torch.sum(delta**2)

            # Estimate of denominator (variance of sample covariance)
            # Simplified: use trace-based estimate
            numerator = delta_norm_sq / n_features
            denominator = torch.trace(cov_sample @ cov_sample) / n_features

            shrinkage = numerator / (denominator + 1e-8)
            shrinkage = torch.clamp(shrinkage, 0.0, 1.0)
        else:
            # For n_samples < n_features, use fixed shrinkage
            shrinkage = min(1.0, (n_features - n_samples + 1) / (n_samples + 1))
            shrinkage = torch.tensor(shrinkage, device=X_centered.device)

        # Shrunk covariance
        cov_shrunk = (1 - shrinkage) * cov_sample + shrinkage * target

        logger.debug(f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f}")

        return cov_shrunk

    def _oas_shrinkage(self, X_centered: torch.Tensor, cov_sample: torch.Tensor) -> torch.Tensor:
        """
        Oracle Approximating Shrinkage (OAS).

        Similar to Ledoit-Wolf but with different shrinkage intensity formula.
        Better for very small samples (n < d).
        """
        n_samples, n_features = X_centered.shape

        # Target: scaled identity
        mu = torch.trace(cov_sample) / n_features
        target = mu * torch.eye(n_features, device=X_centered.device, dtype=X_centered.dtype)

        # OAS shrinkage intensity
        if n_samples > 1:
            trace_S2 = torch.trace(cov_sample @ cov_sample)
            trace_S = torch.trace(cov_sample)

            rho = min(
                1.0,
                ((1 - 2 / n_features) * trace_S2 + trace_S**2) / ((n_samples + 1 - 2 / n_features) * (trace_S2 - trace_S**2 / n_features) + 1e-8),
            )
            shrinkage = torch.tensor(rho, device=X_centered.device)
        else:
            shrinkage = torch.tensor(1.0, device=X_centered.device)

        shrinkage = torch.clamp(shrinkage, 0.0, 1.0)

        # Shrunk covariance
        cov_shrunk = (1 - shrinkage) * cov_sample + shrinkage * target

        logger.debug(f"OAS shrinkage intensity: {shrinkage:.4f}")

        return cov_shrunk

    @staticmethod
    def estimate_condition_number(cov: torch.Tensor) -> float:
        """
        Estimate condition number of covariance matrix.

        Args:
            cov: Covariance matrix

        Returns:
            Condition number κ = λ_max / λ_min
        """
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero

        if len(eigenvalues) == 0:
            return float("inf")

        kappa = eigenvalues.max() / eigenvalues.min()
        return kappa.item()

    @staticmethod
    def compare_methods(X: torch.Tensor, methods: Optional[list] = None) -> dict:
        """
        Compare different covariance estimation methods.

        Args:
            X: Data matrix
            methods: List of methods to compare (None = all)

        Returns:
            Dictionary with covariance estimates and condition numbers
        """
        if methods is None:
            methods = ["none", "diagonal", "ledoit_wolf", "oas"]

        results = {}

        for method in methods:
            estimator = CovarianceEstimator(method=method)
            cov = estimator.estimate(X)
            kappa = CovarianceEstimator.estimate_condition_number(cov)

            results[method] = {
                "covariance": cov,
                "condition_number": kappa,
                "trace": torch.trace(cov).item(),
                "frobenius_norm": torch.norm(cov, p="fro").item(),
            }

        return results


def estimate_covariance(X: torch.Tensor, method: str = "ledoit_wolf", regularization: float = 1e-6) -> torch.Tensor:
    """
    Convenience function for covariance estimation.

    Args:
        X: Data matrix [n_samples, n_features]
        method: Shrinkage method
        regularization: Regularization parameter

    Returns:
        Estimated covariance matrix
    """
    estimator = CovarianceEstimator(method=method, regularization=regularization)
    return estimator.estimate(X)
