"""
Conditional mutual information metric for neural network analysis.
"""

import logging
from typing import Optional

import numpy as np
import torch

from ...core.base import BaseMetric

logger = logging.getLogger(__name__)


class ConditionalMutualInformation(BaseMetric):
    """
    Compute conditional mutual information I(Y;Z|X) where:
    - Y is the output of a neuron
    - Z is a reference signal (e.g., target outputs or mean of other neurons)
    - X is the input

    This measures how much information Y provides about Z beyond what X already provides.
    """

    name = "conditional_mutual_information"
    requires_weights = False
    requires_inputs = True
    requires_outputs = True

    def __init__(self, bins: int = 10, use_gaussian: bool = False):
        """
        Initialize the conditional MI metric.

        Args:
            bins: Number of bins for discretization (if not using Gaussian approximation)
            use_gaussian: Whether to use Gaussian approximation instead of binning
        """
        self.bins = bins
        self.use_gaussian = use_gaussian

    @torch.no_grad()
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        target_outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute conditional MI scores for each neuron.

        Args:
            inputs: Input activations [batch_size, num_input_features]
            weights: Not used
            outputs: Output activations [batch_size, num_neurons]
            target_outputs: Reference signal [batch_size, num_targets] (optional)
            **kwargs: Additional arguments

        Returns:
            CMI scores per neuron [num_neurons]
        """
        if inputs is None or outputs is None:
            raise ValueError("Conditional MI requires both inputs and outputs")

        # Handle dimensions
        if inputs.ndim != 2:
            if inputs.ndim > 2:
                inputs = inputs.flatten(start_dim=1)
            else:
                logger.warning(f"Inputs have unexpected shape: {inputs.shape}")
                return torch.zeros(1, device=outputs.device)

        if outputs.ndim != 2:
            logger.warning(f"Outputs have unexpected shape: {outputs.shape}")
            return torch.zeros(1, device=outputs.device)

        batch_size, num_neurons = outputs.shape

        if batch_size < 10:  # Need enough samples for estimation
            logger.warning(f"Too few samples for CMI estimation: {batch_size}")
            return torch.zeros(num_neurons, device=outputs.device)

        # Use first principal component of inputs as conditioning variable
        if inputs.shape[1] > 1:
            # Simple PCA approximation using SVD
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            _, _, V = torch.linalg.svd(inputs_centered, full_matrices=False)
            X = torch.matmul(inputs_centered, V[0:1, :].T)  # First PC
        else:
            X = inputs

        # Determine reference signal Z
        if target_outputs is not None:
            Z = target_outputs
            if Z.ndim == 1:
                Z = Z.unsqueeze(1)
        else:
            # Use mean of other neurons as reference
            Z = None

        if self.use_gaussian:
            return self._compute_gaussian(X, outputs, Z)
        else:
            return self._compute_binning(X, outputs, Z)

    def _compute_gaussian(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute CMI using Gaussian approximation."""
        num_neurons = Y.shape[1]
        cmi_scores = torch.zeros(num_neurons, device=Y.device)

        for i in range(num_neurons):
            # Current neuron output
            y_i = Y[:, i:i+1]

            # Reference signal
            if Z is not None:
                z = Z
            else:
                # Mean of other neurons
                mask = torch.ones(num_neurons, dtype=torch.bool)
                mask[i] = False
                if mask.sum() > 0:
                    z = Y[:, mask].mean(dim=1, keepdim=True)
                else:
                    continue

            try:
                # Compute I(Y;Z|X) = H(Y|X) + H(Z|X) - H(Y,Z|X)
                # Under Gaussian assumption, we can compute this from covariances

                # Stack variables
                x_flat = X.reshape(X.shape[0], -1)
                xyz = torch.cat([x_flat, y_i, z], dim=1)

                # Compute covariance matrix
                cov = self._covariance(xyz)

                # Indices
                n_x = x_flat.shape[1]
                idx_x = slice(0, n_x)
                idx_y = n_x
                slice(n_x + 1, cov.shape[0])

                # Compute conditional entropies using Schur complement
                # H(Y|X)
                cov_x = cov[idx_x, idx_x]
                cov_y = cov[idx_y, idx_y]
                cov_xy = cov[idx_x, idx_y]

                if torch.linalg.matrix_rank(cov_x) == cov_x.shape[0]:
                    cov_y_given_x = cov_y - cov_xy.T @ torch.linalg.inv(cov_x) @ cov_xy
                    0.5 * torch.log(2 * np.pi * np.e * torch.clamp(cov_y_given_x, min=1e-10))
                else:
                    0.5 * torch.log(2 * np.pi * np.e * torch.clamp(cov_y, min=1e-10))

                # Similar for H(Z|X) and H(Y,Z|X)
                # Simplified: just use correlation-based approximation
                corr_yz_given_x = self._partial_correlation(y_i, z, x_flat)

                # Approximate CMI
                if torch.abs(corr_yz_given_x) < 0.999:
                    cmi_scores[i] = -0.5 * torch.log(1 - corr_yz_given_x**2)

            except Exception as e:
                logger.debug(f"Error computing Gaussian CMI for neuron {i}: {e}")
                continue

        return torch.nan_to_num(cmi_scores, nan=0.0)

    def _compute_binning(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute CMI using binning/discretization."""
        num_neurons = Y.shape[1]
        cmi_scores = torch.zeros(num_neurons, device=Y.device)

        # Convert to numpy for binning
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()

        # Discretize X
        X_discrete = self._discretize(X_np)

        for i in range(num_neurons):
            # Current neuron output
            y_i_np = Y_np[:, i]

            # Reference signal
            if Z is not None:
                z_np = Z.cpu().numpy()
                if z_np.ndim > 1:
                    z_np = z_np[:, 0]  # Use first dimension
            else:
                # Mean of other neurons
                mask = np.ones(num_neurons, dtype=bool)
                mask[i] = False
                if mask.sum() > 0:
                    z_np = Y_np[:, mask].mean(axis=1)
                else:
                    continue

            # Discretize Y and Z
            y_discrete = self._discretize(y_i_np)
            z_discrete = self._discretize(z_np)

            try:
                # Compute CMI using chain rule:
                # I(Y;Z|X) = I(Y;Z,X) - I(Y;X)
                mi_yzx = self._mutual_information_3way(y_discrete, z_discrete, X_discrete)
                mi_yx = self._mutual_information(y_discrete, X_discrete)

                cmi = mi_yzx - mi_yx
                cmi_scores[i] = max(0, cmi)  # CMI should be non-negative

            except Exception as e:
                logger.debug(f"Error computing binned CMI for neuron {i}: {e}")
                continue

        return cmi_scores.to(Y.device)

    def _covariance(self, X: torch.Tensor) -> torch.Tensor:
        """Compute covariance matrix."""
        X_centered = X - X.mean(dim=0, keepdim=True)
        return torch.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)

    def _partial_correlation(
        self,
        Y: torch.Tensor,
        Z: torch.Tensor,
        X: torch.Tensor
    ) -> torch.Tensor:
        """Compute partial correlation between Y and Z given X."""
        # Stack all variables
        all_vars = torch.cat([Y, Z, X], dim=1)

        # Compute correlation matrix
        corr = self._correlation_matrix(all_vars)

        # Extract submatrices
        # corr = [[r_yy, r_yz, r_yx],
        #         [r_zy, r_zz, r_zx],
        #         [r_xy, r_xz, r_xx]]

        n_x = X.shape[1]

        # Partial correlation formula
        try:
            # Get relevant correlations
            r_yz = corr[0, 1]
            r_yx = corr[0, 2:2+n_x]
            r_zx = corr[1, 2:2+n_x]
            R_xx = corr[2:2+n_x, 2:2+n_x]

            # Compute partial correlation
            if torch.linalg.matrix_rank(R_xx) == R_xx.shape[0]:
                R_xx_inv = torch.linalg.inv(R_xx)
                partial_corr = r_yz - r_yx @ R_xx_inv @ r_zx

                # Normalize
                var_y_given_x = 1 - r_yx @ R_xx_inv @ r_yx
                var_z_given_x = 1 - r_zx @ R_xx_inv @ r_zx

                if var_y_given_x > 1e-10 and var_z_given_x > 1e-10:
                    partial_corr = partial_corr / torch.sqrt(var_y_given_x * var_z_given_x)
                else:
                    partial_corr = torch.tensor(0.0)
            else:
                # Fallback to simple correlation
                partial_corr = r_yz

        except Exception:
            # Fallback to simple correlation
            partial_corr = corr[0, 1]

        return torch.clamp(partial_corr, -1.0, 1.0)

    def _correlation_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix."""
        cov = self._covariance(X)
        std = torch.sqrt(torch.diag(cov) + 1e-10)
        outer_std = torch.outer(std, std)
        return torch.where(outer_std > 1e-10, cov / outer_std, torch.zeros_like(cov))

    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        min_val, max_val = np.min(x), np.max(x)

        if max_val - min_val < 1e-10:
            return np.zeros_like(x, dtype=int)

        bins = np.linspace(min_val, max_val, self.bins + 1)
        digitized = np.digitize(x, bins[:-1]) - 1
        return np.clip(digitized, 0, self.bins - 1)

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two discrete variables."""
        # Joint probability
        joint_hist = np.zeros((self.bins, self.bins))
        for i in range(len(x)):
            joint_hist[x[i], y[i]] += 1

        joint_prob = joint_hist / len(x)

        # Marginal probabilities
        p_x = joint_prob.sum(axis=1)
        p_y = joint_prob.sum(axis=0)

        # Compute MI
        mi = 0.0
        for i in range(self.bins):
            for j in range(self.bins):
                if joint_prob[i, j] > 1e-10 and p_x[i] > 1e-10 and p_y[j] > 1e-10:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (p_x[i] * p_y[j]))

        return mi / np.log(2)  # Convert to bits

    def _mutual_information_3way(
        self,
        y: np.ndarray,
        z: np.ndarray,
        x: np.ndarray
    ) -> float:
        """Compute I(Y;Z,X) = I(Y;(Z,X)) treating (Z,X) as a joint variable."""
        # Create joint variable for (Z,X)
        zx_joint = z * self.bins + x  # Simple encoding

        # Ensure we don't exceed the range
        max_val = self.bins * self.bins - 1
        zx_joint = np.clip(zx_joint, 0, max_val)

        # Compute MI between Y and the joint variable
        # Need to adjust bins for the joint variable
        original_bins = self.bins
        self.bins = self.bins * self.bins

        mi = self._mutual_information(y, zx_joint)

        # Restore original bins
        self.bins = original_bins

        return mi
