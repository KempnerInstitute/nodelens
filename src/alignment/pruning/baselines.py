"""
Pruning Baselines for Comparison with SCAR

Implements:
- Wanda: Weight × Activation pruning (Sun et al., 2023)
- SparseGPT-style: Second-order one-shot pruning (Frantar & Alistarh, 2023)
- Magnitude: Simple weight magnitude pruning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class WandaPruning:
    """
    Wanda: Pruning by Weights and Activations
    
    Reference: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (2023)
    
    Key idea: Score = |W| * ||X||_2 (weight magnitude × activation norm)
    """
    
    def __init__(
        self,
        sparsity: float = 0.5,
        structured: bool = True,
        prune_dim: int = 0,  # 0 = prune rows (output neurons), 1 = prune columns (input features)
    ):
        self.sparsity = sparsity
        self.structured = structured
        self.prune_dim = prune_dim
        
    def compute_scores(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Wanda importance scores.
        
        Args:
            weight: Weight matrix [out_features, in_features]
            activations: Input activations [batch, seq_len, in_features] or [batch, in_features]
            
        Returns:
            Importance scores per neuron (structured) or per weight (unstructured)
        """
        # Flatten activations if needed
        if activations.ndim == 3:
            activations = activations.reshape(-1, activations.shape[-1])
        
        # Compute activation norms (L2 norm across samples)
        activation_norms = torch.norm(activations, p=2, dim=0)  # [in_features]
        
        if self.structured:
            # Structured: score per output neuron
            # Score_i = sum_j |W_ij| * ||X_j||_2
            weight_abs = torch.abs(weight)  # [out_features, in_features]
            scores = torch.matmul(weight_abs, activation_norms)  # [out_features]
        else:
            # Unstructured: score per weight
            # Score_ij = |W_ij| * ||X_j||_2
            scores = torch.abs(weight) * activation_norms.unsqueeze(0)  # [out_features, in_features]
            
        return scores
    
    def get_pruning_mask(
        self,
        scores: torch.Tensor,
        sparsity: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Create pruning mask based on scores.
        
        Returns:
            Boolean mask where True = keep, False = prune
        """
        if sparsity is None:
            sparsity = self.sparsity
            
        if self.structured:
            # Prune neurons with lowest scores
            num_prune = int(sparsity * scores.numel())
            threshold = torch.kthvalue(scores, num_prune).values
            mask = scores > threshold
        else:
            # Prune weights with lowest scores
            flat_scores = scores.flatten()
            num_prune = int(sparsity * flat_scores.numel())
            threshold = torch.kthvalue(flat_scores, num_prune).values
            mask = scores > threshold
            
        return mask


class SparseGPTStylePruning:
    """
    Simplified SparseGPT-style pruning using second-order information.
    
    Reference: Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be 
    Accurately Pruned in One-Shot" (2023)
    
    Key idea: Use Hessian approximation to minimize reconstruction error
    """
    
    def __init__(
        self,
        sparsity: float = 0.5,
        structured: bool = True,
        block_size: int = 128,
        percdamp: float = 0.01,
    ):
        self.sparsity = sparsity
        self.structured = structured
        self.block_size = block_size
        self.percdamp = percdamp
        
    def compute_hessian_inverse(
        self,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse Hessian approximation from activations.
        
        H ≈ X^T X / n (Fisher approximation)
        """
        # Flatten activations
        if activations.ndim == 3:
            activations = activations.reshape(-1, activations.shape[-1])
        
        n_samples = activations.shape[0]
        
        # Compute H = X^T X / n
        H = torch.matmul(activations.T, activations) / n_samples
        
        # Add damping for numerical stability
        damp = self.percdamp * torch.diag(H).mean()
        H = H + damp * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        
        # Compute inverse (or pseudo-inverse for stability)
        try:
            H_inv = torch.linalg.inv(H)
        except:
            H_inv = torch.linalg.pinv(H)
            
        return H_inv
    
    def compute_scores(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SparseGPT-style importance scores.
        
        Score_i = W_i^2 / [H^{-1}]_{ii}  (optimal brain surgeon criterion)
        """
        H_inv = self.compute_hessian_inverse(activations)
        
        if self.structured:
            # For structured pruning, aggregate across input dimension
            # Score for row i = sum_j W_ij^2 / [H^{-1}]_{jj}
            diag_H_inv = torch.diag(H_inv)  # [in_features]
            weight_sq = weight ** 2  # [out_features, in_features]
            scores = torch.sum(weight_sq / diag_H_inv.unsqueeze(0), dim=1)  # [out_features]
        else:
            # Unstructured: per-weight scores
            diag_H_inv = torch.diag(H_inv)
            scores = (weight ** 2) / diag_H_inv.unsqueeze(0)
            
        return scores
    
    def get_pruning_mask(
        self,
        scores: torch.Tensor,
        sparsity: Optional[float] = None,
    ) -> torch.Tensor:
        """Create pruning mask (same as Wanda)."""
        if sparsity is None:
            sparsity = self.sparsity
            
        if self.structured:
            num_prune = int(sparsity * scores.numel())
            threshold = torch.kthvalue(scores, max(1, num_prune)).values
            mask = scores > threshold
        else:
            flat_scores = scores.flatten()
            num_prune = int(sparsity * flat_scores.numel())
            threshold = torch.kthvalue(flat_scores, max(1, num_prune)).values
            mask = scores > threshold
            
        return mask


class MagnitudePruning:
    """
    Simple magnitude-based pruning baseline.
    
    Score = ||W_i||_p (L1 or L2 norm of weight row/column)
    """
    
    def __init__(
        self,
        sparsity: float = 0.5,
        structured: bool = True,
        norm_type: int = 2,  # L1 or L2
        prune_dim: int = 0,
    ):
        self.sparsity = sparsity
        self.structured = structured
        self.norm_type = norm_type
        self.prune_dim = prune_dim
        
    def compute_scores(
        self,
        weight: torch.Tensor,
        activations: Optional[torch.Tensor] = None,  # Not used, but kept for API consistency
    ) -> torch.Tensor:
        """Compute magnitude-based importance scores."""
        if self.structured:
            # Norm per row (output neuron)
            scores = torch.norm(weight, p=self.norm_type, dim=1)
        else:
            # Per-weight magnitude
            scores = torch.abs(weight)
            
        return scores
    
    def get_pruning_mask(
        self,
        scores: torch.Tensor,
        sparsity: Optional[float] = None,
    ) -> torch.Tensor:
        """Create pruning mask."""
        if sparsity is None:
            sparsity = self.sparsity
            
        if self.structured:
            num_prune = int(sparsity * scores.numel())
            threshold = torch.kthvalue(scores, max(1, num_prune)).values
            mask = scores > threshold
        else:
            flat_scores = scores.flatten()
            num_prune = int(sparsity * flat_scores.numel())
            threshold = torch.kthvalue(flat_scores, max(1, num_prune)).values
            mask = scores > threshold
            
        return mask


def apply_structured_pruning(
    module: nn.Linear,
    mask: torch.Tensor,
    prune_dim: int = 0,
) -> nn.Linear:
    """
    Apply structured pruning to a Linear layer.
    
    Args:
        module: Linear layer to prune
        mask: Boolean mask (True = keep)
        prune_dim: 0 = prune output neurons, 1 = prune input features
        
    Returns:
        New Linear layer with reduced dimensions
    """
    keep_indices = torch.where(mask)[0]
    
    if prune_dim == 0:
        # Prune output neurons (rows)
        new_out_features = keep_indices.numel()
        new_weight = module.weight.data[keep_indices, :]
        new_bias = module.bias.data[keep_indices] if module.bias is not None else None
        
        new_module = nn.Linear(module.in_features, new_out_features, bias=module.bias is not None)
        new_module.weight.data = new_weight
        if new_bias is not None:
            new_module.bias.data = new_bias
    else:
        # Prune input features (columns)
        new_in_features = keep_indices.numel()
        new_weight = module.weight.data[:, keep_indices]
        
        new_module = nn.Linear(new_in_features, module.out_features, bias=module.bias is not None)
        new_module.weight.data = new_weight
        if module.bias is not None:
            new_module.bias.data = module.bias.data
            
    return new_module


# def compare_pruning_methods(
#     model: nn.Module,
#     calibration_data: torch.Tensor,
#     sparsity_levels: List[float] = [0.3, 0.5, 0.7],
#     methods: List[str] = ["magnitude", "wanda", "sparsegpt"],
# ) -> Dict[str, Dict[float, Dict[str, Any]]]:
#     """
#     Compare different pruning methods on a model.
    
#     Args:
#         model: Model to prune
#         calibration_data: Data for computing activation statistics
#         sparsity_levels: List of sparsity levels to test
#         methods: List of pruning methods to compare
        
#     Returns:
#         Dictionary with results per method and sparsity level
#     """
#     results = {method: {} for method in methods}
    
#     # Initialize pruning methods
#     pruners = {
#         "magnitude": MagnitudePruning(structured=True),
#         "wanda": WandaPruning(structured=True),
#         "sparsegpt": SparseGPTStylePruning(structured=True),
#     }
    
#     for method in methods:
#         if method not in pruners:
#             logger.warning(f"Unknown pruning method: {method}")
#             continue
            
#         pruner = pruners[method]
        
#         for sparsity in sparsity_levels:
#             pruner.sparsity = sparsity
            
#             # Collect scores for all layers
#             layer_scores = {}
#             for name, module in model.named_modules():
#                 if isinstance(module, nn.Linear):
#                     # Get activations for this layer (would need hooks in practice)
#                     # This is a simplified version
#                     scores = pruner.compute_scores(
#                         module.weight.data,
#                         calibration_data,
#                     )
#                     layer_scores[name] = {
#                         "scores": scores,
#                         "mask": pruner.get_pruning_mask(scores, sparsity),
#                     }
            
#             results[method][sparsity] = {
#                 "layer_scores": layer_scores,
#                 "sparsity": sparsity,
#             }
    
#     return results


# Registry for easy access
PRUNING_METHODS = {
    "magnitude": MagnitudePruning,
    "wanda": WandaPruning,
    "sparsegpt": SparseGPTStylePruning,
}


def get_pruning_method(name: str, **kwargs) -> Any:
    """Get a pruning method by name."""
    if name not in PRUNING_METHODS:
        raise ValueError(f"Unknown pruning method: {name}. Available: {list(PRUNING_METHODS.keys())}")
    return PRUNING_METHODS[name](**kwargs)

