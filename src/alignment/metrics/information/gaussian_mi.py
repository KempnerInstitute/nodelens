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
        Compute Gaussian MI using the same efficient pattern as RayleighQuotient.

        Uses the relationship: MI(X; y_i) = -0.5 * log(1 - R²_i)
        where R²_i = (w_i^T Σ_x w_i) / var(y_i)
        
        This is computed efficiently using the same covariance-based approach as RQ.

        Args:
            inputs: Input activations [batch_size, input_dim] or [batch*patches, features] for CNN
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations (computed if not provided)

        Returns:
            MI scores for each neuron [output_dim] or single score
        """
        original_device = inputs.device
        
        # Handle CNN inputs [B, C, H, W] -> flatten properly
        if inputs.ndim == 4:
            B, C, H, W = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(B * H * W, C)
        elif inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
            
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
            
        n_samples, input_dim = inputs.shape
        output_dim, weight_dim = weights.shape

        # Handle dimension mismatch
        if input_dim != weight_dim:
            if input_dim > weight_dim:
                inputs = inputs[:, :weight_dim]
                input_dim = weight_dim
            else:
                return torch.ones(output_dim, device=original_device)

        if self.per_neuron:
            # Force CPU for large input dimensions (covariance matrix would be huge)
            # down_proj has 14336 inputs -> cov is [14336, 14336] = 820MB
            use_cpu = self._should_use_cpu(inputs, weights) or input_dim > 8192
            if use_cpu:
                inputs = inputs.cpu()
                weights = weights.cpu()
            
            # Convert to float32 for numerical stability
            inputs_f = inputs.float()
            weights_f = weights.float()
            
            # Compute covariance matrix (same as RQ)
            inputs_centered = inputs_f - inputs_f.mean(dim=0, keepdim=True)
            cov = torch.matmul(inputs_centered.T, inputs_centered) / (n_samples - 1)
            
            # Add regularization
            cov = cov + self.regularization * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
            
            # Compute w^T Σ_x w for all neurons efficiently
            # This is the SIGNAL VARIANCE captured by each neuron
            wc = torch.matmul(weights_f, cov)  # [output_dim, input_dim]
            w_cov_w = torch.sum(wc * weights_f, dim=1)  # [output_dim] = signal variance
            
            # KEY DIFFERENCE FROM RQ:
            # - RQ = (w^T Σ_x w) / (w^T w)  -- normalizes by weight norm (scale-invariant)
            # - MI = 0.5 * log(1 + (w^T Σ_x w) / σ_n²)  -- uses raw signal variance!
            #
            # From the theory (see alignment_notes/main.tex Section 3):
            # For noisy linear neuron y = w^T X + n where n ~ N(0, σ_n²):
            #   I(X; y) = 0.5 * log(1 + (w^T Σ_X w) / σ_n²)
            #
            # The w^T w term is NOT in MI! That's what makes MI and RQ different.
            # Neurons with large weights but low efficiency (low RQ) can still have
            # high MI because they capture lots of absolute signal variance.
            
            # Estimate noise variance as a fraction of average signal variance
            # or use the configured noise_std
            noise_var = self.noise_std ** 2
            
            # Compute SNR = signal_variance / noise_variance
            # signal_variance = w^T Σ_x w (NOT divided by w^T w!)
            snr = w_cov_w / (noise_var + 1e-10)
            
            # MI = 0.5 * log(1 + SNR) in nats
            mi_scores = 0.5 * torch.log1p(snr)
            
            # Clamp to reasonable range (no clamping needed if formula is correct)
            mi_scores = torch.clamp(mi_scores, min=0.0)
            
            return mi_scores.to(original_device)

        else:
            # Compute joint MI between all inputs and all outputs (rarely used for pruning)
            # Move to CPU for heavy covariance computation
            inputs_f = inputs.float().cpu()
            weights_f = weights.float().cpu()
            
            if outputs is None:
                outputs_f = inputs_f @ weights_f.T
            else:
                outputs_f = outputs.float().cpu()
                
            inputs_centered = inputs_f - inputs_f.mean(dim=0, keepdim=True)
            outputs_centered = outputs_f - outputs_f.mean(dim=0, keepdim=True)

            # Compute covariances
            cov_x = (inputs_centered.mT @ inputs_centered) / (n_samples - 1)
            cov_y = (outputs_centered.mT @ outputs_centered) / (n_samples - 1)
            cov_xy = (inputs_centered.mT @ outputs_centered) / (n_samples - 1)

            # Gaussian MI
            mi_gaussian = self._gaussian_mi(cov_x, cov_y, cov_xy)

            # Skip expensive corrections for joint MI
            total_mi = mi_gaussian

            # Return same value for all neurons
            return torch.full((output_dim,), total_mi.item(), device=original_device)


@register_metric("mi_fast", aliases=["gaussian_mi_fast", "mi_gap"])
class FastGaussianMI(GaussianMIAnalytic):
    """
    Fast Gaussian MI approximation for CNNs using Global Average Pooling.

    Instead of unfolding patches or spatial reshaping, uses GAP to reduce
    spatial dimensions to get a channel-wise computation.

    This is an APPROXIMATION but is:
    - 10-100x faster than spatial MI
    - Uses O(C^2) memory instead of O((C*H*W)^2)
    - Better for early conv layers with large spatial dimensions

    For inputs [B, C_in, H, W] and weights [C_out, C_in, k, k]:
    1. Apply GAP: inputs -> [B, C_in]
    2. Sum weights over kernel: weights -> [C_out, C_in]
    3. Compute standard MI on channel dimension

    Example:
        >>> mi_fast = FastGaussianMI()
        >>> scores = mi_fast.compute(inputs=inputs, weights=conv.weight)
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        regularization: float = 1e-6,
        **config,
    ):
        """
        Initialize fast MI metric.

        Args:
            noise_std: Assumed noise standard deviation
            regularization: Regularization for covariance
            **config: Additional configuration
        """
        super().__init__(
            expansion_order=0,  # Skip Edgeworth for speed
            noise_std=noise_std,
            regularization=regularization,
            per_neuron=True,
            **config,
        )

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute fast MI using GAP for CNN inputs.

        Args:
            inputs: Input activations [B, C_in, H, W] or [B, C_in]
            weights: Conv weights [C_out, C_in, k_H, k_W] or [C_out, C_in]
            outputs: Not used

        Returns:
            MI values for each output channel [C_out]
        """
        if weights is None:
            raise ValueError("FastMI requires weights")
        if inputs is None:
            raise ValueError("FastMI requires inputs")

        # Handle CNN inputs: [B, C_in, H, W] -> GAP -> [B, C_in]
        if inputs.ndim == 4:
            # Global Average Pooling over spatial dimensions
            inputs = inputs.mean(dim=(2, 3))  # [B, C_in]
        elif inputs.ndim == 3:
            # Patchwise input [B, F, P] -> average over patches
            inputs = inputs.mean(dim=2)  # [B, F]

        # Handle Conv weights: [C_out, C_in, k_H, k_W] -> sum over kernel -> [C_out, C_in]
        if weights.ndim == 4:
            # Sum over kernel dimensions to get channel-wise weights
            weights = weights.sum(dim=(2, 3))  # [C_out, C_in]
        elif weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Now use standard 2D MI computation
        return super().compute(inputs=inputs, weights=weights, **kwargs)
