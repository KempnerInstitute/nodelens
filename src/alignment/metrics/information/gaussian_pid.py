"""
Gaussian PID (MMI-based) synergy approximation without BROJA.

Defines a per-neuron synergy metric using the MMI redundancy axiom:
  S(Z; Y_i, Y_j) = I(Z; [Y_i,Y_j]) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j)).

We compute I terms under Gaussian assumptions from covariance estimates.
Returns average synergy per neuron over sampled partners j.
"""

from typing import Optional, Any
import torch
import logging

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


def _safe_var(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = x.var(dim=0, unbiased=True, keepdim=False)
    return torch.clamp(v, min=eps)


def _mi_gaussian_scalar_scalar(z: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # z, y: [B]
    zc = z - z.mean()
    yc = y - y.mean()
    vz = torch.clamp(zc.var(unbiased=True), min=eps)
    vy = torch.clamp(yc.var(unbiased=True), min=eps)
    c = (zc @ yc) / max(1, (z.shape[0] - 1))
    rho_sq = torch.clamp((c * c) / (vz * vy), 0.0, 0.999999)
    return -0.5 * torch.log(1.0 - rho_sq)


def _mi_gaussian_scalar_vector(z: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """I(z; Y) for scalar z and Y with shape [B, k] using σ_{z|Y}^2 regression formula."""
    B = z.shape[0]
    zc = z - z.mean()
    Yc = Y - Y.mean(dim=0, keepdim=True)
    vz = torch.clamp(zc.var(unbiased=True), min=eps)
    # Covariances
    cov_y = (Yc.T @ Yc) / max(1, (B - 1))  # [k,k]
    cov_zy = (zc.view(1, -1) @ Yc) / max(1, (B - 1))  # [1,k]
    # σ_{z|Y}^2 = var(z) - cov_{zy} Σ_y^{-1} cov_{yz}
    try:
        inv_cov_y = torch.inverse(cov_y + eps * torch.eye(cov_y.shape[0], device=Y.device, dtype=Y.dtype))
    except RuntimeError:
        inv_cov_y = torch.linalg.pinv(cov_y)
    var_cond = vz - (cov_zy @ inv_cov_y @ cov_zy.T).squeeze()
    var_cond = torch.clamp(var_cond, min=eps)
    return 0.5 * torch.log(vz / var_cond)


@register_metric("gaussian_pid_synergy_mmi")
class GaussianPIDSynergyMMI(BaseMetric):
    """
    Per-neuron Gaussian PID synergy (MMI redundancy) with respect to target Z.

    Args:
        sample_pairs: number of partner neurons j to sample per i (None = all)
        min_samples: minimum batch size for stable covariance estimation
    Inputs:
        outputs: layer outputs [B, N]
        target_outputs: target variable(s) Z [B] or [B, d]
    Returns:
        synergy per neuron [N], averaged over sampled partners
    """

    def __init__(self, sample_pairs: Optional[int] = 100, min_samples: int = 10, **config: Any):
        super().__init__(**config)
        self.sample_pairs = sample_pairs
        self.min_samples = min_samples

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        target_outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if outputs is None or target_outputs is None:
            raise ValueError("GaussianPIDSynergyMMI requires outputs and target_outputs")

        if outputs.ndim != 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        if target_outputs.ndim > 1 and target_outputs.shape[1] == 1:
            target_outputs = target_outputs.squeeze(1)

        B, N = outputs.shape
        if B < self.min_samples:
            logger.warning("gaussian_pid_synergy_mmi: too few samples; returning zeros")
            return torch.zeros(N, device=outputs.device, dtype=outputs.dtype)

        Z = target_outputs
        device = outputs.device

        # If Z is multivariate, average MI across dims
        if Z.ndim == 2:
            d = Z.shape[1]
        else:
            d = 1

        # Precompute I(Z; Y_i) for all i (average across Z dims if needed)
        I_Z_Yi = torch.zeros(N, device=device)
        if d == 1:
            z = Z.view(B)
            for i in range(N):
                I_Z_Yi[i] = _mi_gaussian_scalar_scalar(z, outputs[:, i])
        else:
            # For multivariate Z, average over dims
            for i in range(N):
                mi_sum = 0.0
                for k in range(d):
                    mi_sum += _mi_gaussian_scalar_scalar(Z[:, k], outputs[:, i])
                I_Z_Yi[i] = mi_sum / d

        # Prepare partner sampling
        def partner_indices(i: int) -> torch.Tensor:
            cand = torch.cat([torch.arange(0, i, device=device), torch.arange(i + 1, N, device=device)])
            if self.sample_pairs is None or self.sample_pairs >= cand.numel():
                return cand
            idx = torch.randperm(cand.numel(), device=device)[: self.sample_pairs]
            return cand[idx]

        # Compute synergy per i averaged over j in partners
        synergy = torch.zeros(N, device=device)
        eps = 1e-12
        if d == 1:
            z = Z.view(B)
            for i in range(N):
                js = partner_indices(i)
                if js.numel() == 0:
                    synergy[i] = 0.0
                    continue
                s_sum = 0.0
                yi = outputs[:, i]
                I_i = I_Z_Yi[i]
                for j in js:
                    yj = outputs[:, j]
                    I_j = I_Z_Yi[j]
                    Yij = torch.stack([yi, yj], dim=1)
                    I_joint = _mi_gaussian_scalar_vector(z, Yij)
                    R_mmi = torch.minimum(I_i, I_j)
                    s_sum += (I_joint - I_i - I_j + R_mmi)
                synergy[i] = s_sum / js.numel()
        else:
            # For multivariate Z, average I across dims; use same MMI structure
            for i in range(N):
                js = partner_indices(i)
                if js.numel() == 0:
                    synergy[i] = 0.0
                    continue
                s_sum = 0.0
                yi = outputs[:, i]
                I_i = I_Z_Yi[i]
                for j in js:
                    yj = outputs[:, j]
                    I_j = I_Z_Yi[j]
                    # approximate I(Z; [yi,yj]) by averaging across Z dims
                    I_joint_sum = 0.0
                    for k in range(d):
                        I_joint_sum += _mi_gaussian_scalar_vector(Z[:, k], torch.stack([yi, yj], dim=1))
                    I_joint = I_joint_sum / d
                    R_mmi = torch.minimum(I_i, I_j)
                    s_sum += (I_joint - I_i - I_j + R_mmi)
                synergy[i] = s_sum / js.numel()

        return torch.nan_to_num(synergy, nan=0.0, neginf=0.0, posinf=0.0)


