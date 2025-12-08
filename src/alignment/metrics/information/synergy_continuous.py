"""
Synergy metric using continuous target (logit margin).

This addresses the mixed discrete-continuous MI issue by using a continuous
decision variable T (e.g., logit margin) instead of discrete labels Z.

Under Gaussian approximation, all MI terms can be computed from covariances
of (T, Y_i, Y_j), making the synergy computation well-defined.

References:
- Barrett (2015): Gaussian PID
- Williams & Beer (2010): PID foundations
"""

import logging
from typing import Any, Optional, Literal

import torch
import numpy as np

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("synergy_continuous_target")
class SynergyContinuousTarget(BaseMetric):
    """
    Compute per-neuron synergy using Gaussian MI with continuous target.
    
    Instead of discrete labels Z, we use a continuous decision variable T:
    - 'logit_margin': T = f_z(x) - max_{c!=z} f_c(x)  (correct - max incorrect)
    - 'correct_logit': T = f_z(x)  (correct class logit)
    - 'logit_pc1': T = PC1 of logits  (first principal component)
    
    This allows proper Gaussian MI estimation for synergy:
        S(T; Y_i, Y_j) = I(T; Y_i, Y_j) - I(T; Y_i) - I(T; Y_j) + min(I(T; Y_i), I(T; Y_j))
    
    Example:
        >>> metric = SynergyContinuousTarget(target_type='logit_margin', num_pairs=10)
        >>> synergy = metric.compute(outputs=activations, logits=logits, labels=labels)
    """
    
    def __init__(
        self,
        target_type: Literal['logit_margin', 'correct_logit', 'logit_pc1'] = 'logit_margin',
        num_pairs: int = 10,
        sampling_strategy: str = 'top_k',  # 'random', 'top_k', 'all'
        eps: float = 1e-8,
        **config: Any,
    ):
        """
        Initialize synergy metric.
        
        Args:
            target_type: Type of continuous target to use
            num_pairs: Number of partner neurons per neuron
            sampling_strategy: 'random', 'top_k' (highest synergy), 'all'
            eps: Numerical stability epsilon
        """
        super().__init__(**config)
        self.target_type = target_type
        self.num_pairs = num_pairs
        self.sampling_strategy = sampling_strategy
        self.eps = eps
    
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
        outputs: torch.Tensor,  # [batch, n_neurons]
        logits: torch.Tensor,   # [batch, n_classes]
        labels: torch.Tensor,   # [batch]
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute per-neuron synergy scores.
        
        Args:
            outputs: Layer activations [batch, n_neurons]
            logits: Model logits [batch, n_classes]
            labels: True labels [batch]
            
        Returns:
            Per-neuron synergy scores [n_neurons]
        """
        device = outputs.device
        dtype = outputs.dtype
        
        # Flatten if needed
        if outputs.ndim > 2:
            # Conv layer: [B, C, H, W] -> [B, C] via GAP
            outputs = outputs.mean(dim=(2, 3)) if outputs.ndim == 4 else outputs.reshape(outputs.shape[0], -1)
        
        batch_size, n_neurons = outputs.shape
        
        # Compute continuous target T
        T = self._compute_target(logits, labels)  # [batch]
        
        if T is None:
            logger.warning("Could not compute continuous target")
            return torch.zeros(n_neurons, device=device, dtype=dtype)
        
        # Compute synergy for each neuron
        synergy = torch.zeros(n_neurons, device=device, dtype=dtype)
        
        # Precompute individual MIs with target
        mi_individual = self._compute_mi_batch(T, outputs)  # [n_neurons]
        
        for i in range(n_neurons):
            partners = self._sample_partners(i, n_neurons, mi_individual)
            if len(partners) == 0:
                continue
            
            syn_values = []
            for j in partners:
                s = self._compute_pairwise_synergy(
                    T, outputs[:, i], outputs[:, j],
                    mi_individual[i], mi_individual[j]
                )
                syn_values.append(s)
            
            if syn_values:
                synergy[i] = torch.stack(syn_values).mean()
        
        return synergy
    
    def _compute_target(self, logits: torch.Tensor, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute continuous target variable T."""
        batch_size, n_classes = logits.shape
        
        if self.target_type == 'logit_margin':
            # T = correct_logit - max_incorrect_logit
            correct_logits = logits[torch.arange(batch_size), labels]
            # Mask out correct class
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[torch.arange(batch_size), labels] = False
            max_incorrect = logits.masked_fill(~mask, float('-inf')).max(dim=1)[0]
            T = correct_logits - max_incorrect
            
        elif self.target_type == 'correct_logit':
            T = logits[torch.arange(batch_size), labels]
            
        elif self.target_type == 'logit_pc1':
            # First principal component of logits
            logits_centered = logits - logits.mean(dim=0)
            _, _, V = torch.linalg.svd(logits_centered, full_matrices=False)
            T = (logits_centered @ V[0]).squeeze()
            
        else:
            logger.warning(f"Unknown target type: {self.target_type}")
            return None
        
        return T
    
    def _compute_mi_batch(self, T: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute I(T; Y_i) for all neurons using Gaussian approximation.
        
        I(T; Y) = 0.5 * log(var(T) / var(T|Y))
                = 0.5 * log(1 / (1 - rho^2))
                = -0.5 * log(1 - rho^2)
        """
        T = T.float()
        Y = Y.float()
        
        # Compute correlations
        T_centered = T - T.mean()
        Y_centered = Y - Y.mean(dim=0)
        
        T_std = T_centered.std() + self.eps
        Y_std = Y_centered.std(dim=0) + self.eps
        
        # Pearson correlation
        rho = (T_centered[:, None] * Y_centered).mean(dim=0) / (T_std * Y_std)
        rho = torch.clamp(rho, -1 + self.eps, 1 - self.eps)
        
        # Gaussian MI
        mi = -0.5 * torch.log(1 - rho ** 2)
        return torch.clamp(mi, min=0.0)
    
    def _compute_pairwise_synergy(
        self,
        T: torch.Tensor,
        Y_i: torch.Tensor,
        Y_j: torch.Tensor,
        mi_i: torch.Tensor,
        mi_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute synergy S(T; Y_i, Y_j) using MMI redundancy axiom.
        
        S = I(T; Y_i, Y_j) - I(T; Y_i) - I(T; Y_j) + min(I(T; Y_i), I(T; Y_j))
        """
        # Joint MI via 3x3 covariance
        joint = torch.stack([T, Y_i, Y_j], dim=1)  # [batch, 3]
        joint_centered = joint - joint.mean(dim=0)
        cov = (joint_centered.T @ joint_centered) / (joint.shape[0] - 1 + self.eps)
        
        # Add regularization
        cov = cov + self.eps * torch.eye(3, device=cov.device, dtype=cov.dtype)
        
        # I(T; [Y_i, Y_j]) = 0.5 * log(det(cov_T) * det(cov_YiYj) / det(cov_all))
        var_T = cov[0, 0]
        cov_Y = cov[1:, 1:]
        det_all = torch.linalg.det(cov)
        det_Y = torch.linalg.det(cov_Y)
        
        if det_all <= 0 or det_Y <= 0 or var_T <= 0:
            return torch.tensor(0.0, device=T.device)
        
        mi_joint = 0.5 * torch.log(var_T * det_Y / det_all)
        mi_joint = torch.clamp(mi_joint, min=0.0)
        
        # MMI redundancy
        redundancy = torch.min(mi_i, mi_j)
        
        # Synergy
        synergy = mi_joint - mi_i - mi_j + redundancy
        
        return synergy
    
    def _sample_partners(
        self,
        neuron_idx: int,
        n_neurons: int,
        mi_scores: Optional[torch.Tensor] = None,
    ) -> list:
        """Sample partner neurons for synergy computation."""
        available = [j for j in range(n_neurons) if j != neuron_idx]
        
        if len(available) == 0:
            return []
        
        k = min(self.num_pairs, len(available))
        
        if self.sampling_strategy == 'all':
            return available
        
        elif self.sampling_strategy == 'random':
            idx = torch.randperm(len(available))[:k]
            return [available[i] for i in idx.tolist()]
        
        elif self.sampling_strategy == 'top_k':
            # Sample partners with highest MI (likely to have interesting synergy)
            if mi_scores is not None:
                partner_mi = mi_scores[available]
                _, top_idx = torch.topk(partner_mi, k)
                return [available[i] for i in top_idx.tolist()]
            else:
                idx = torch.randperm(len(available))[:k]
                return [available[i] for i in idx.tolist()]
        
        return []
