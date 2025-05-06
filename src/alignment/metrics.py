"""
Alignment metrics for neural network analysis.

This module provides metrics for measuring alignment between weight matrices 
and input activations, with various metrics to quantify the degree of alignment.
It also supports node-wise scoring for pruning and other experiments.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

from alignment.alignment_metrics import (
    AlignmentMetricBase, 
    RQMetric, 
    MIMetric, 
    WeightSimilarityMetric, 
    NodeRedundancyMetric,
    AlignmentMetrics,
    alignment
)


class AlignmentMetric:
    """
    Base class for alignment metrics.
    
    This class provides a standardized interface for alignment metrics,
    including support for node-wise scoring for pruning experiments.
    """
    
    def __init__(self, name: str = "RQ", scale_by_norm: bool = False):
        """
        Initialize alignment metric.
        
        Args:
            name: Name of the alignment metric to use
            scale_by_norm: Whether to scale the covariance or final measure by norm
        """
        self.name = name
        self.scale_by_norm = scale_by_norm
        
    def measure(self, 
                activations: List[torch.Tensor], 
                targets: torch.Tensor, 
                num_classes: int = 10) -> List[float]:
        """
        Measure alignment between activations and weights.
        
        Args:
            activations: List of hidden layer activations
            targets: Target labels
            num_classes: Number of classes in the dataset
            
        Returns:
            List of alignment values per layer
        """
        results = []
        
        for layer_activations in activations:
            # Handle different types of metrics
            if self.name == "RQ":
                # For RQ, we typically compute alignment to weight matrices
                # For demonstration, let's compute activation correlation as a proxy
                if layer_activations.dim() > 2:
                    layer_activations = layer_activations.reshape(layer_activations.size(0), -1)
                
                # Center the activations
                centered = layer_activations - layer_activations.mean(dim=0, keepdim=True)
                
                # Compute correlation matrix of activations
                cov = torch.matmul(centered.t(), centered) / (centered.size(0) - 1)
                
                # Normalize to get correlation matrix
                diag = torch.diag(cov)
                std_dev = torch.sqrt(diag)
                corr = cov / (torch.outer(std_dev, std_dev) + 1e-8)
                
                # Take mean of absolute correlations as a measure of alignment
                mask = torch.ones_like(corr) - torch.eye(corr.size(0), device=corr.device)
                result = (torch.abs(corr) * mask).sum() / (mask.sum() + 1e-8)
                
            elif self.name == "class_separation":
                # Measure separation between class activations
                if layer_activations.dim() > 2:
                    layer_activations = layer_activations.reshape(layer_activations.size(0), -1)
                
                # Compute class means
                class_means = []
                for c in range(num_classes):
                    class_mask = (targets == c)
                    if class_mask.sum() > 0:
                        class_mean = layer_activations[class_mask].mean(dim=0)
                        class_means.append(class_mean)
                
                if len(class_means) > 1:
                    # Compute mean distance between class centers
                    class_means = torch.stack(class_means)
                    dists = torch.cdist(class_means, class_means)
                    # Mask the diagonal
                    mask = torch.ones_like(dists) - torch.eye(dists.size(0), device=dists.device)
                    result = (dists * mask).sum() / (mask.sum() + 1e-8)
                else:
                    result = torch.tensor(0.0, device=layer_activations.device)
            
            else:
                # Default to using the AlignmentMetrics class from alignment_metrics
                # We would need weights to compute proper alignment, so this is a placeholder
                result = torch.tensor(0.5, device=layer_activations.device)
            
            results.append(result.item())
        
        return results
        
    def compute_per_node_scores(
        self,
        layer_input: torch.Tensor,   # shape (N, input_dim) or with conv flattening
        layer_weights: torch.Tensor, # shape (num_nodes, input_dim)
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute a per-node alignment score for each node's weight vector w_i.

        Return shape: (num_nodes,).

        For RQ:
            RQ_i = (w_i^T Cov(X) w_i) / (w_i^T w_i), optionally scaled or adjusted.
        For MI_x:
            Possibly an alternate measure.

        Args:
            layer_input: 2D tensor of shape (N, input_dim) with the data
            layer_weights: 2D tensor (#nodes, input_dim)
            device: PyTorch device to ensure everything is on the correct device

        Returns:
            A 1D tensor (#nodes,) of alignment scores
        """
        if device is None:
            device = layer_input.device

        # Basic checks
        if layer_input.dim() != 2:
            # For CNN, you might have already flattened each patch or done an "unfold"
            # If not, do so here or raise an error:
            raise ValueError(f"layer_input must be 2D, got shape {layer_input.shape}")

        # Move weights if not on the same device
        layer_weights = layer_weights.to(device)

        # Depending on the metric name, do your computation:
        if self.name.upper() == "RQ":
            return self._compute_rq(layer_input, layer_weights, device)
        elif self.name.upper().startswith("MI"):
            return self._compute_mi_placeholder(layer_input, layer_weights, device)
        else:
            # Default fallback => just do RQ
            return self._compute_rq(layer_input, layer_weights, device)
            
    def _compute_rq(
        self,
        X: torch.Tensor,            # shape (N, input_dim)
        W: torch.Tensor,            # shape (num_nodes, input_dim)
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient per node.

        RQ_i = (w_i^T Cov(X) w_i) / (w_i^T w_i)
        where Cov(X) is the sample covariance of X. If scale_by_norm is True,
        we might also scale Cov(X) or the final RQ_i.

        Args:
            X: input data
            W: weight matrix
        Returns:
            shape (#nodes,)
        """
        # Center the input
        N = X.size(0)
        if N < 2:
            # fallback if dataset is extremely small
            return torch.zeros(W.size(0), device=device)

        X_centered = X - X.mean(dim=0, keepdim=True)
        # sample covariance => (input_dim x input_dim)
        cov_x = (X_centered.t() @ X_centered) / (N - 1)

        # Possibly scale
        if self.scale_by_norm:
            norm_val = cov_x.norm(p=2)
            if norm_val > 1e-12:
                cov_x = cov_x / norm_val

        # W shape (#nodes, input_dim)
        # Let's do a row-by-row RQ:
        w_norm_sq = (W * W).sum(dim=1)  # (#nodes,)
        wCov = W @ cov_x               # (#nodes, input_dim)
        numerator = (wCov * W).sum(dim=1)  # dot with W row wise => (#nodes,)

        eps = 1e-12
        rq_scores = numerator / (w_norm_sq + eps)
        return rq_scores

    def _compute_mi_placeholder(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Placeholder for a mutual-information-like measure, or some custom measure.
        For demonstration, let's just do a log(1 + abs(RQ)) or something.

        In practice, you'd implement the actual per-node MI measure.
        """
        # We'll reuse the RQ as a base, then transform:
        rq_scores = self._compute_rq(X, W, device)
        # Suppose we define MI_i = log(1 + |RQ_i|)
        mi_scores = torch.log(1.0 + rq_scores.abs())
        return mi_scores


class RankAlignmentMetric(AlignmentMetric):
    """
    Rank-based alignment metric.
    
    This metric assesses alignment based on the correlation of
    principal components of activations and weight matrices.
    """
    
    def __init__(self):
        """Initialize rank alignment metric."""
        super().__init__(name="rank")
    
    def measure(self, 
                activations: List[torch.Tensor], 
                targets: torch.Tensor, 
                num_classes: int = 10) -> List[float]:
        """
        Measure rank alignment for each layer.
        
        Args:
            activations: List of hidden layer activations
            targets: Target labels
            num_classes: Number of classes in the dataset
            
        Returns:
            List of rank alignment values per layer
        """
        # Invoke AlignmentMetrics.measure with 'RQ' method
        # For now, we'll delegate to the parent class
        return super().measure(activations, targets, num_classes)


class NullSpaceAlignmentMetric(AlignmentMetric):
    """
    Null space alignment metric.
    
    This metric assesses alignment based on the projection of
    weights onto the null space of the input covariance matrix.
    """
    
    def __init__(self):
        """Initialize null space alignment metric."""
        super().__init__(name="null_space")
    
    def measure(self, 
                activations: List[torch.Tensor], 
                targets: torch.Tensor, 
                num_classes: int = 10) -> List[float]:
        """
        Measure null space alignment for each layer.
        
        Args:
            activations: List of hidden layer activations
            targets: Target labels
            num_classes: Number of classes in the dataset
            
        Returns:
            List of null space alignment values per layer
        """
        results = []
        
        for layer_activations in activations:
            if layer_activations.dim() > 2:
                layer_activations = layer_activations.reshape(layer_activations.size(0), -1)
            
            # Compute eigenvalues and eigenvectors of input covariance
            # Center the activations
            centered = layer_activations - layer_activations.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = torch.matmul(centered.t(), centered) / (centered.size(0) - 1)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Compute the effective rank (number of eigenvalues above a threshold)
            threshold = torch.max(eigenvalues) * 1e-3
            effective_rank = (eigenvalues > threshold).sum().item()
            
            # Use effective rank as a proxy for null space alignment
            result = effective_rank / cov.size(0)
            results.append(result)
        
        return results


def get_metric(name: str, scale_by_norm: bool = False) -> AlignmentMetric:
    """
    Get alignment metric by name.
    
    Args:
        name: Name of the metric ('RQ', 'MI', 'rank', 'null_space', etc.)
        scale_by_norm: Whether to scale the covariance or final measure by norm
        
    Returns:
        AlignmentMetric instance
    """
    name = name.lower()
    
    if name == 'rank':
        return RankAlignmentMetric()
    elif name == 'null_space':
        return NullSpaceAlignmentMetric()
    else:
        return AlignmentMetric(name=name, scale_by_norm=scale_by_norm) 