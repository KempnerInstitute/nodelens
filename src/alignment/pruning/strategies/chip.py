"""
CHIP: Channel Independence-based Pruning.

Reference: Sui et al., "CHIP: CHannel Independence-based Pruning for Compact Neural Networks"
           NeurIPS 2021. https://arxiv.org/abs/2110.13981

CHIP prunes channels with low "independence" - i.e., channels that are highly
correlated with (redundant to) other channels in the same layer.

Independence score: I_i = 1 / (1 + sum_j |corr(y_i, y_j)|)

Channels with LOW independence (high correlation) are pruned first.
This is conceptually similar to pruning high-redundancy channels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_chip_scores(
    activations: torch.Tensor,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute CHIP independence scores for channels.
    
    Args:
        activations: [N, C, H, W] or [N, C] tensor of channel activations
        normalize: If True, normalize scores to [0, 1] range
        
    Returns:
        [C] array of independence scores (higher = more independent = keep)
    """
    # Flatten spatial dims if present
    if activations.ndim == 4:
        N, C, H, W = activations.shape
        acts = activations.permute(1, 0, 2, 3).reshape(C, -1)  # [C, N*H*W]
    elif activations.ndim == 3:
        N, C, D = activations.shape
        acts = activations.permute(1, 0, 2).reshape(C, -1)  # [C, N*D]
    elif activations.ndim == 2:
        acts = activations.T  # [C, N]
    else:
        raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    acts = acts.float()
    C = acts.shape[0]
    
    # Compute correlation matrix
    # Center each channel
    acts_centered = acts - acts.mean(dim=1, keepdim=True)
    
    # Compute std
    stds = acts_centered.std(dim=1, keepdim=True).clamp(min=1e-8)
    acts_normed = acts_centered / stds
    
    # Correlation matrix: [C, C]
    corr = (acts_normed @ acts_normed.T) / acts.shape[1]
    corr = corr.clamp(-1, 1)
    
    # Independence score: I_i = 1 / (1 + sum_j!=i |corr(i,j)|)
    abs_corr = torch.abs(corr)
    abs_corr.fill_diagonal_(0)  # Exclude self-correlation
    
    sum_abs_corr = abs_corr.sum(dim=1)  # [C]
    independence = 1.0 / (1.0 + sum_abs_corr)
    
    scores = independence.cpu().numpy()
    
    if normalize and scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    return scores


def chip_prune_layer(
    activations: torch.Tensor,
    num_to_prune: int,
) -> np.ndarray:
    """
    Get indices of channels to prune using CHIP criterion.
    
    Args:
        activations: Channel activations
        num_to_prune: Number of channels to prune
        
    Returns:
        Indices of channels to prune (lowest independence)
    """
    scores = compute_chip_scores(activations, normalize=False)
    
    # Prune channels with LOWEST independence (highest redundancy)
    prune_order = np.argsort(scores)  # Ascending: low independence first
    return prune_order[:num_to_prune]


class CHIPPruning:
    """
    CHIP pruning strategy for structured channel pruning.
    
    This computes per-channel independence scores based on inter-channel
    correlations and prunes the most redundant (least independent) channels.
    """
    
    def __init__(
        self,
        model: nn.Module,
        *,
        calibration_loader: Optional[Any] = None,
        max_samples: int = 1000,
        device: str = "cuda",
    ):
        self.model = model
        self.calibration_loader = calibration_loader
        self.max_samples = max_samples
        self.device = device
        self._activation_cache: Dict[str, torch.Tensor] = {}
    
    def _collect_activations(
        self,
        layer_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Collect activations for specified layers."""
        if self.calibration_loader is None:
            raise ValueError("calibration_loader required for CHIP")
        
        activations: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
        hooks = []
        
        def make_hook(name: str):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                activations[name].append(out.detach().cpu())
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Collect
        self.model.eval()
        samples_collected = 0
        with torch.no_grad():
            for batch in self.calibration_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                self.model(x)
                samples_collected += x.shape[0]
                if samples_collected >= self.max_samples:
                    break
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Concatenate
        result = {}
        for name in layer_names:
            if activations[name]:
                result[name] = torch.cat(activations[name], dim=0)
        
        return result
    
    def compute_scores(
        self,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute CHIP independence scores for all (or specified) layers.
        
        Returns:
            Dict mapping layer_name -> [C] array of independence scores
        """
        if layer_names is None:
            # Find all conv/linear layers
            layer_names = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    layer_names.append(name)
        
        activations = self._collect_activations(layer_names)
        
        scores = {}
        for name, acts in activations.items():
            scores[name] = compute_chip_scores(acts, normalize=True)
            logger.debug(f"CHIP scores for {name}: mean={scores[name].mean():.4f}")
        
        return scores
    
    def get_pruning_mask(
        self,
        layer_name: str,
        sparsity: float,
    ) -> np.ndarray:
        """
        Get binary mask indicating which channels to KEEP.
        
        Args:
            layer_name: Name of layer
            sparsity: Fraction of channels to prune (0.5 = prune 50%)
            
        Returns:
            Boolean mask where True = keep, False = prune
        """
        if layer_name not in self._activation_cache:
            activations = self._collect_activations([layer_name])
            self._activation_cache.update(activations)
        
        acts = self._activation_cache[layer_name]
        C = acts.shape[1] if acts.ndim >= 2 else acts.shape[0]
        
        num_to_prune = int(C * sparsity)
        prune_indices = chip_prune_layer(acts, num_to_prune)
        
        mask = np.ones(C, dtype=bool)
        mask[prune_indices] = False
        
        return mask


# For integration with existing pruning framework
def chip_score_channels(
    activations: Dict[str, torch.Tensor],
    layer_name: str,
) -> np.ndarray:
    """
    Wrapper for use in the generalized pruning framework.
    
    Args:
        activations: Dict of layer_name -> activation tensor
        layer_name: Which layer to score
        
    Returns:
        Per-channel scores (higher = keep)
    """
    if layer_name not in activations:
        raise KeyError(f"No activations for layer {layer_name}")
    
    return compute_chip_scores(activations[layer_name], normalize=True)
