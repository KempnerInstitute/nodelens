"""
Gaussian pairwise redundancy and synergy metrics.

Implements pairwise redundancy and synergy proxies using Gaussian formulas
consistent with the draft definitions:

- Redundancy(Y_i, Y_j | X) = 0.5 * log(1 + (w_i^T Σ_X w_j)^2 / ((w_i^T Σ_X w_i)(w_j^T Σ_X w_j)))
- Synergy(Y_i, Y_j | X) = 0.5 * log( det Σ_{Y_i Y_j} / (det Σ_{Y_i} det Σ_{Y_j}) )

Returns per-neuron scores by averaging pairwise quantities over sampled partners.
"""

from typing import Optional, Any
import torch
import logging

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


def _compute_cov_x(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2:
        inputs = inputs.reshape(inputs.shape[0], -1)
    n = inputs.shape[0]
    Xc = inputs - inputs.mean(dim=0, keepdim=True)
    return (Xc.T @ Xc) / max(1, (n - 1))


@register_metric("pairwise_redundancy_gaussian")
class PairwiseRedundancyGaussian(BaseMetric):
    """
    Pairwise redundancy per neuron using Gaussian approximation on inputs.

    Uses Σ_X from inputs and layer weights W to compute for each pair (i, j):
        R_ij = 0.5 * log(1 + (c_ij^2) / (v_i v_j))
    where c_ij = w_i^T Σ_X w_j and v_i = w_i^T Σ_X w_i.

    Returns the average R_ij over j != i for each neuron i, optionally sampled.
    """

    def __init__(self, min_samples: int = 2, sample_pairs: int = 100, **config: Any):
        super().__init__(**config)
        self.min_samples = min_samples
        self.sample_pairs = sample_pairs

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
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if inputs is None or weights is None:
            raise ValueError("PairwiseRedundancyGaussian requires inputs and weights")

        if inputs.shape[0] < self.min_samples:
            logger.warning("PairwiseRedundancyGaussian: too few samples; returning zeros")
            return torch.zeros(weights.shape[0], device=weights.device, dtype=weights.dtype)

        # Ensure 2D weights
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Compute Σ_X and induced Σ_Y = W Σ_X W^T
        cov_x = _compute_cov_x(inputs.to(weights.device))
        Sigma_y = weights @ cov_x @ weights.T  # [n, n]

        n = Sigma_y.shape[0]
        v = torch.diag(Sigma_y)  # [n]
        eps = 1e-12
        v = torch.clamp(v, min=eps)

        # Prepare pair indices
        device = weights.device
        all_i = torch.arange(n, device=device)

        # If sampling, pick a subset of j per i
        def avg_over_j(vec_diag: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
            # Compute per-i averages over j != i using sampling for scalability
            if self.sample_pairs is None or self.sample_pairs >= n - 1:
                # Use all pairs
                mask = ~torch.eye(n, dtype=torch.bool, device=device)
                c_sq = mat.pow(2)
                denom = (vec_diag.view(-1, 1) * vec_diag.view(1, -1)).clamp_min(eps)
                term = 0.5 * torch.log1p(c_sq / denom)
                term = term.masked_select(mask).view(n, -1)
                return term.mean(dim=1)
            else:
                # Sample j indices per i
                means = torch.zeros(n, device=device)
                for i in range(n):
                    # sample without replacement excluding i
                    candidates = torch.cat([torch.arange(0, i, device=device), torch.arange(i + 1, n, device=device)])
                    if candidates.numel() == 0:
                        means[i] = 0.0
                        continue
                    k = min(self.sample_pairs, candidates.numel())
                    perm = torch.randperm(candidates.numel(), device=device)[:k]
                    js = candidates[perm]
                    c_ij = mat[i, js]
                    denom_ij = (vec_diag[i] * vec_diag[js]).clamp_min(eps)
                    term_ij = 0.5 * torch.log1p((c_ij.pow(2)) / denom_ij)
                    means[i] = term_ij.mean()
                return means

        redundancy = avg_over_j(v, Sigma_y)
        return torch.nan_to_num(redundancy, nan=0.0)


@register_metric("pairwise_synergy_gaussian")
class PairwiseSynergyGaussian(BaseMetric):
    """
    Pairwise synergy proxy per neuron using Gaussian formula on outputs.

    Uses Σ_Y = W Σ_X W^T (or directly from outputs) to compute for each pair (i, j):
        S_ij = 0.5 * log( det Σ_{ij} / (var_i var_j) )
            = 0.5 * log(1 - (cov_ij^2)/(var_i var_j))
    Note: This expression is ≤ 0 for |ρ|>0. It is provided as in the draft.

    Returns the average S_ij over j != i for each neuron i, optionally sampled.
    """

    def __init__(self, min_samples: int = 2, sample_pairs: int = 100, use_outputs: bool = False, **config: Any):
        super().__init__(**config)
        self.min_samples = min_samples
        self.sample_pairs = sample_pairs
        self.use_outputs = use_outputs  # if True, use provided outputs to estimate Σ_Y

    @property
    def requires_inputs(self) -> bool:
        return not self.use_outputs

    @property
    def requires_weights(self) -> bool:
        return not self.use_outputs

    @property
    def requires_outputs(self) -> bool:
        return self.use_outputs

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.use_outputs:
            if outputs is None:
                raise ValueError("PairwiseSynergyGaussian with use_outputs=True requires outputs")
            if outputs.ndim != 2:
                outputs = outputs.reshape(outputs.shape[0], -1)
            # Estimate Σ_Y from outputs
            n = outputs.shape[0]
            Yc = outputs - outputs.mean(dim=0, keepdim=True)
            Sigma_y = (Yc.T @ Yc) / max(1, (n - 1))
            device = outputs.device
        else:
            if inputs is None or weights is None:
                raise ValueError("PairwiseSynergyGaussian requires inputs and weights when use_outputs=False")
            if weights.ndim > 2:
                weights = weights.reshape(weights.shape[0], -1)
            cov_x = _compute_cov_x(inputs.to(weights.device))
            Sigma_y = weights @ cov_x @ weights.T
            device = weights.device

        n = Sigma_y.shape[0]
        v = torch.diag(Sigma_y).clamp_min(1e-12)
        c = Sigma_y
        eps = 1e-12

        def avg_synergy() -> torch.Tensor:
            if self.sample_pairs is None or self.sample_pairs >= n - 1:
                mask = ~torch.eye(n, dtype=torch.bool, device=device)
                rho_sq = (c.pow(2)) / (v.view(-1, 1) * v.view(1, -1)).clamp_min(eps)
                term = 0.5 * torch.log(torch.clamp(1.0 - rho_sq, min=eps))
                term = term.masked_select(mask).view(n, -1)
                return term.mean(dim=1)
            else:
                means = torch.zeros(n, device=device)
                for i in range(n):
                    candidates = torch.cat([torch.arange(0, i, device=device), torch.arange(i + 1, n, device=device)])
                    if candidates.numel() == 0:
                        means[i] = 0.0
                        continue
                    k = min(self.sample_pairs, candidates.numel())
                    perm = torch.randperm(candidates.numel(), device=device)[:k]
                    js = candidates[perm]
                    rho_sq_ij = (c[i, js].pow(2)) / (v[i] * v[js]).clamp_min(eps)
                    term_ij = 0.5 * torch.log(torch.clamp(1.0 - rho_sq_ij, min=eps))
                    means[i] = term_ij.mean()
                return means

        synergy = avg_synergy()
        return torch.nan_to_num(synergy, nan=0.0)


