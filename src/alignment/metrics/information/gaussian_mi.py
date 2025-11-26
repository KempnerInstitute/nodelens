"""
Analytic Gaussian Mutual Information with non-Gaussian corrections.

This module implements mutual information computation assuming approximately
Gaussian distributions, with Edgeworth expansions for higher-order corrections.
"""

from typing import Dict, Optional

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric


@register_metric("gaussian_mi_analytic")
class GaussianMIAnalytic(BaseMetric):
    """
    Compute mutual information between input and output assuming Gaussian distributions.

    For linear transformations Y = WX + ε where X and ε are Gaussian:
    - I(X;Y) = 1/2 * log(det(Σ_Y) / det(Σ_Y|X))

    With Edgeworth expansion for non-Gaussian corrections up to specified order.
    """

    name = "gaussian_mi_analytic"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False

    def __init__(
        self,
        expansion_order: int = 2,
        noise_std: float = 0.1,
        regularization: float = 1e-6,
        per_neuron: bool = True,
        use_entropy_edgeworth: bool = True,
    ):
        """
        Initialize Gaussian MI metric with expansion.

        Args:
            expansion_order: Order of Edgeworth expansion (0=pure Gaussian, 1-3 for corrections)
            noise_std: Assumed noise standard deviation
            regularization: Small value added to covariance matrices for stability
            per_neuron: If True, compute MI for each neuron separately
        """
        super().__init__()
        self.expansion_order = expansion_order
        self.noise_std = noise_std
        self.regularization = regularization
        self.per_neuron = per_neuron
        self.use_entropy_edgeworth = use_entropy_edgeworth

    def _compute_cumulants(self, data: torch.Tensor, max_order: int = 4) -> Dict[int, torch.Tensor]:
        """
        Compute cumulants up to specified order.

        For centered data:
        - κ₁ = 0 (mean)
        - κ₂ = variance
        - κ₃ = E[X³] (skewness * σ³)
        - κ₄ = E[X⁴] - 3σ⁴ (excess kurtosis * σ⁴)
        """
        # Center the data
        data_centered = data - data.mean(dim=0, keepdim=True)
        data.shape[0]

        cumulants = {}

        # Second cumulant (variance)
        cumulants[2] = torch.var(data_centered, dim=0, unbiased=True)

        if max_order >= 3:
            # Third cumulant (related to skewness)
            cumulants[3] = torch.mean(data_centered**3, dim=0)

        if max_order >= 4:
            # Fourth cumulant (related to kurtosis)
            fourth_moment = torch.mean(data_centered**4, dim=0)
            cumulants[4] = fourth_moment - 3 * cumulants[2] ** 2

        return cumulants

    def _univariate_entropy_edgeworth(self, data: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Differential entropy of near-Gaussian scalar variable using exact first corrections:
          h(X) ≈ 0.5 * log(2π e σ^2) - (γ1^2)/12 - (γ2^2)/48
        where γ1 is skewness, γ2 is excess kurtosis.
        data: [B]
        Returns scalar entropy in nats.
        """
        x = data
        x = x - x.mean()
        var = torch.clamp(x.var(unbiased=True), min=eps)
        std = torch.sqrt(var)
        if std <= eps:
            return 0.5 * torch.log(2 * torch.pi * torch.e * torch.clamp(var, min=eps))
        z = x / std
        gamma1 = torch.mean(z**3)
        gamma2 = torch.mean(z**4) - 3.0  # excess kurtosis
        h_gauss = 0.5 * torch.log(2 * torch.pi * torch.e * var)
        corr = -(gamma1**2) / 12.0 - (gamma2**2) / 48.0
        return h_gauss + corr

    def _gaussian_mi(self, cov_x: torch.Tensor, cov_y: torch.Tensor, cov_xy: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian mutual information.

        I(X;Y) = 1/2 * log(det(Σ_X) * det(Σ_Y) / det(Σ))
        where Σ is the joint covariance matrix.
        """
        # Construct joint covariance matrix
        n_x = cov_x.shape[0]
        n_y = cov_y.shape[0]

        joint_cov = torch.zeros(n_x + n_y, n_x + n_y, device=cov_x.device)
        joint_cov[:n_x, :n_x] = cov_x
        joint_cov[n_x:, n_x:] = cov_y
        joint_cov[:n_x, n_x:] = cov_xy
        joint_cov[n_x:, :n_x] = cov_xy.T

        # Add regularization for numerical stability
        reg_eye_x = self.regularization * torch.eye(n_x, device=cov_x.device)
        reg_eye_y = self.regularization * torch.eye(n_y, device=cov_y.device)
        reg_eye_joint = self.regularization * torch.eye(n_x + n_y, device=joint_cov.device)

        cov_x = cov_x + reg_eye_x
        cov_y = cov_y + reg_eye_y
        joint_cov = joint_cov + reg_eye_joint

        # Compute determinants
        det_x = torch.linalg.det(cov_x)
        det_y = torch.linalg.det(cov_y)
        det_joint = torch.linalg.det(joint_cov)

        # Mutual information
        mi = 0.5 * torch.log(det_x * det_y / det_joint)

        return mi

    def _edgeworth_correction(
        self, cumulants_x: Dict[int, torch.Tensor], cumulants_y: Dict[int, torch.Tensor], cov_xy: torch.Tensor, order: int
    ) -> torch.Tensor:
        """
        Compute Edgeworth expansion corrections to mutual information.

        The corrections involve higher-order cumulants and capture deviations
        from Gaussianity.
        """
        correction = 0.0

        if order >= 1 and 3 in cumulants_x and 3 in cumulants_y:
            # First-order correction involves third cumulants (skewness)
            # This is a simplified approximation
            kappa3_x = cumulants_x[3].mean() if cumulants_x[3].numel() > 1 else cumulants_x[3].item()
            kappa3_y = cumulants_y[3].item() if hasattr(cumulants_y[3], "item") else cumulants_y[3]
            var_x = cumulants_x[2].mean() if cumulants_x[2].numel() > 1 else cumulants_x[2].item()
            var_y = cumulants_y[2].item() if hasattr(cumulants_y[2], "item") else cumulants_y[2]

            # Normalized third cumulants
            gamma1_x = kappa3_x / (var_x**1.5 + 1e-8)
            gamma1_y = kappa3_y / (var_y**1.5 + 1e-8)

            # Correlation coefficient (cov_xy is already scalar)
            corr = cov_xy / (torch.sqrt(torch.tensor(var_x * var_y)) + 1e-8)

            # First-order correction (simplified form)
            correction += (1 / 6) * corr * gamma1_x * gamma1_y

        if order >= 2 and 4 in cumulants_x and 4 in cumulants_y:
            # Second-order correction involves fourth cumulants (kurtosis)
            kappa4_x = cumulants_x[4].mean() if cumulants_x[4].numel() > 1 else cumulants_x[4].item()
            kappa4_y = cumulants_y[4].item() if hasattr(cumulants_y[4], "item") else cumulants_y[4]
            var_x = cumulants_x[2].mean() if cumulants_x[2].numel() > 1 else cumulants_x[2].item()
            var_y = cumulants_y[2].item() if hasattr(cumulants_y[2], "item") else cumulants_y[2]

            # Normalized fourth cumulants (excess kurtosis)
            gamma2_x = kappa4_x / (var_x**2 + 1e-8)
            gamma2_y = kappa4_y / (var_y**2 + 1e-8)

            # Correlation coefficient
            corr = cov_xy / (torch.sqrt(torch.tensor(var_x * var_y)) + 1e-8)

            # Second-order correction (simplified form)
            correction += (1 / 24) * (corr**2) * (gamma2_x + gamma2_y)

            if 3 in cumulants_x and 3 in cumulants_y:
                # Mixed term involving third cumulants
                kappa3_x = cumulants_x[3].mean() if cumulants_x[3].numel() > 1 else cumulants_x[3].item()
                kappa3_y = cumulants_y[3].item() if hasattr(cumulants_y[3], "item") else cumulants_y[3]
                gamma1_x = kappa3_x / (var_x**1.5 + 1e-8)
                gamma1_y = kappa3_y / (var_y**1.5 + 1e-8)
                correction += (1 / 72) * (gamma1_x**2 + gamma1_y**2)

        if order >= 3:
            # Third-order corrections become quite complex
            # Here we provide a simplified version
            if 4 in cumulants_x and 4 in cumulants_y and 3 in cumulants_x and 3 in cumulants_y:
                var_x = cumulants_x[2].mean() if cumulants_x[2].numel() > 1 else cumulants_x[2].item()
                var_y = cumulants_y[2].item() if hasattr(cumulants_y[2], "item") else cumulants_y[2]
                corr = cov_xy / (torch.sqrt(torch.tensor(var_x * var_y)) + 1e-8)

                kappa3_x = cumulants_x[3].mean() if cumulants_x[3].numel() > 1 else cumulants_x[3].item()
                kappa3_y = cumulants_y[3].item() if hasattr(cumulants_y[3], "item") else cumulants_y[3]
                kappa4_x = cumulants_x[4].mean() if cumulants_x[4].numel() > 1 else cumulants_x[4].item()
                kappa4_y = cumulants_y[4].item() if hasattr(cumulants_y[4], "item") else cumulants_y[4]

                gamma1_x = kappa3_x / (var_x**1.5 + 1e-8)
                gamma1_y = kappa3_y / (var_y**1.5 + 1e-8)
                gamma2_x = kappa4_x / (var_x**2 + 1e-8)
                gamma2_y = kappa4_y / (var_y**2 + 1e-8)

                # Third-order correction (highly simplified)
                correction += (1 / 144) * corr * (gamma1_x * gamma2_y + gamma1_y * gamma2_x)

        # Convert to scalar if needed
        if hasattr(correction, "item"):
            correction = correction.item()

        return correction

    def compute(self, inputs: torch.Tensor, weights: torch.Tensor, outputs: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Compute Gaussian MI with non-Gaussian corrections.

        Args:
            inputs: Input activations [batch_size, input_dim]
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations (computed if not provided)

        Returns:
            MI scores for each neuron [output_dim] or single score
        """
        # Flatten inputs if needed (handle CNN activations)
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
            
        batch_size, input_dim = inputs.shape
        output_dim, weight_dim = weights.shape

        # Handle dimension mismatch (common for CNN layers where inputs aren't unfolded)
        if input_dim != weight_dim:
            # Use the minimum dimension
            min_dim = min(input_dim, weight_dim)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_dim = min_dim

        # Compute outputs if not provided
        if outputs is None:
            outputs = inputs @ weights.T
            # Add small noise to simulate realistic conditions
            if self.noise_std > 0:
                noise = torch.randn_like(outputs) * self.noise_std
                outputs = outputs + noise

        if self.per_neuron:
            # Compute MI for each output neuron separately
            mi_scores = torch.zeros(output_dim, device=inputs.device)

            # Compute input statistics once
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            cov_x = (inputs_centered.T @ inputs_centered) / (batch_size - 1)
            cumulants_x = self._compute_cumulants(inputs, self.expansion_order + 2)

            for i in range(output_dim):
                # Get single output
                y = outputs[:, i].unsqueeze(1)
                y_centered = y - y.mean(dim=0, keepdim=True)

                # Compute covariances
                cov_y = (y_centered.T @ y_centered) / (batch_size - 1)
                cov_xy = (inputs_centered.T @ y_centered) / (batch_size - 1)

                # Gaussian MI baseline
                mi_gaussian = self._gaussian_mi(cov_x, cov_y, cov_xy)

                if self.expansion_order > 0 and self.use_entropy_edgeworth:
                    # Compute MI via entropy difference with Edgeworth entropy corrections (univariate)
                    # Fit linear regression to get residual r = y - E[y|X]
                    # Using population-style coefficients from sample covariances: beta = Σ_x^{-1} cov_xy
                    try:
                        beta = torch.linalg.solve(cov_x + self.regularization * torch.eye(cov_x.shape[0], device=cov_x.device), cov_xy).squeeze()
                    except RuntimeError:
                        beta = torch.linalg.pinv(cov_x) @ cov_xy
                        beta = beta.squeeze()
                    y_hat_centered = inputs_centered @ beta
                    r_centered = y_centered.squeeze() - y_hat_centered

                    # Entropy corrections for y and residual r
                    h_y = self._univariate_entropy_edgeworth(y_centered.squeeze())
                    h_r = self._univariate_entropy_edgeworth(r_centered)
                    mi_edge = torch.clamp(h_y - h_r, min=0.0)
                    mi_scores[i] = mi_edge
                else:
                    mi_scores[i] = mi_gaussian

            return mi_scores

        else:
            # Compute joint MI between all inputs and all outputs
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            outputs_centered = outputs - outputs.mean(dim=0, keepdim=True)

            # Compute covariances
            cov_x = (inputs_centered.T @ inputs_centered) / (batch_size - 1)
            cov_y = (outputs_centered.T @ outputs_centered) / (batch_size - 1)
            cov_xy = (inputs_centered.T @ outputs_centered) / (batch_size - 1)

            # Gaussian MI
            mi_gaussian = self._gaussian_mi(cov_x, cov_y, cov_xy)

            # Add corrections if requested
            if self.expansion_order > 0:
                cumulants_x = self._compute_cumulants(inputs, self.expansion_order + 2)
                self._compute_cumulants(outputs, self.expansion_order + 2)

                # For joint MI, we need to handle the corrections differently
                # Here we provide a simplified scalar correction
                avg_correction = 0.0
                for i in range(output_dim):
                    y_single = outputs[:, i].unsqueeze(1)
                    cumulants_y_single = self._compute_cumulants(y_single, self.expansion_order + 2)
                    cov_xy_single = cov_xy[:, i].unsqueeze(1)

                    correction = self._edgeworth_correction(cumulants_x, cumulants_y_single, cov_xy_single.squeeze(), self.expansion_order)
                    avg_correction += correction

                avg_correction = avg_correction / output_dim
                total_mi = mi_gaussian + avg_correction
            else:
                total_mi = mi_gaussian

            # Return same value for all neurons
            return torch.full((output_dim,), total_mi.item(), device=inputs.device)
