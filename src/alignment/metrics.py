"""
Alignment metrics for neural network analysis.

This module provides metrics for measuring alignment between weight matrices 
and input activations, with various metrics to quantify the degree of alignment.
It supports node-wise scoring for pruning experiments and various utility
functions for alignment-based analysis.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Import core implementations from utils.metrics_utils
from alignment.utils.metrics_utils import (
    AlignmentMetricBase,
    RQMetric,
    MIMetric,
    WeightSimilarityMetric,
    NodeRedundancyMetric,
    AlignmentMetricsFactory as AlignmentMetrics,
    alignment
)


class RankAlignmentMetric(AlignmentMetricBase):
    """
    Rank-based alignment metric.
    
    This metric assesses alignment based on the correlation of
    principal components of activations and weight matrices.
    """
    
    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure rank alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            
        Returns:
            Tensor containing rank alignment values
        """
        # Ensure inputs have at least 2 dimensions
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
            
        # Move weights to same device as inputs
        weights = weights.to(inputs.device)
        
        # Center the inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute SVD of inputs
        try:
            U, S, V = torch.linalg.svd(X, full_matrices=False)
            
            # Compute alignment between singular vectors and weight vectors
            alignment_scores = []
            for i in range(min(10, V.size(0))):  # Use top 10 singular vectors
                singular_vector = V[i, :]
                
                # Compute alignment with each weight vector
                for j in range(weights.size(0)):
                    weight_vector = weights[j, :]
                    
                    # Compute cosine similarity
                    cos_sim = torch.dot(singular_vector, weight_vector) / (
                        torch.norm(singular_vector) * torch.norm(weight_vector) + 1e-8
                    )
                    alignment_scores.append(cos_sim.abs().item())
            
            # Return mean alignment
            return torch.tensor(alignment_scores).mean()
            
        except Exception:
            # Fallback if SVD fails
            return torch.tensor(0.5, device=inputs.device)
    
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric."""
        return "RankAlignment"


class NullSpaceAlignmentMetric(AlignmentMetricBase):
    """
    Null space alignment metric.
    
    This metric assesses alignment based on the projection of
    weights onto the null space of the input covariance matrix.
    """
    
    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure null space alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            
        Returns:
            Tensor containing null space alignment values
        """
        # Ensure inputs have at least 2 dimensions
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
            
        # Move weights to same device as inputs
        weights = weights.to(inputs.device)
        
        # Center the inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute eigendecomposition of input covariance
        try:
            cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Calculate threshold for null space
            threshold = torch.max(eigenvalues) * 1e-3
            
            # Identify null space eigenvectors
            null_indices = (eigenvalues < threshold).nonzero().flatten()
            
            if null_indices.size(0) == 0:
                # No clear null space
                return torch.tensor(0.0, device=inputs.device)
                
            null_eigenvectors = eigenvectors[:, null_indices]
            
            # Compute projection of weights onto null space
            projection_scores = []
            for i in range(weights.size(0)):
                weight_vector = weights[i, :]
                
                # Project weight vector onto null space
                projection = torch.matmul(
                    torch.matmul(null_eigenvectors.t(), weight_vector),
                    null_eigenvectors
                )
                
                # Compute fraction of weight vector in null space
                projection_norm = torch.norm(projection)
                weight_norm = torch.norm(weight_vector)
                
                projection_scores.append((projection_norm / weight_norm).item())
                
            return torch.tensor(projection_scores).mean()
                
        except Exception:
            # Fallback if eigendecomposition fails
            return torch.tensor(0.0, device=inputs.device)
    
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric."""
        return "NullSpaceAlignment"


# Register additional metrics with the registry
AlignmentMetrics.register("rank", RankAlignmentMetric)
AlignmentMetrics.register("null_space", NullSpaceAlignmentMetric)


class AlignmentMetric:
    """
    Interface for computing alignment metrics on neural networks.
    
    This class provides a standardized interface for alignment metrics,
    including support for node-wise scoring for pruning experiments.
    It serves as a higher-level wrapper around the lower-level AlignmentMetrics
    factory and individual metric implementations.
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
        
        # Validate that the metric exists
        self._validate_metric_name()
        
    def _validate_metric_name(self):
        """Validate that the metric name is valid."""
        normalized_name = self.name.lower()
        
        valid_metrics = [
            "rq", "mi", "rank", "null_space", 
            "weight_similarity", "redundancy", "class_separation"
        ]
        
        if normalized_name not in valid_metrics:
            raise ValueError(f"Unknown alignment metric: {self.name}. "
                           f"Valid options are: {', '.join(valid_metrics)}")
        
    def measure(self, 
                activations: Union[List[torch.Tensor], torch.Tensor], 
                targets: Optional[torch.Tensor] = None, 
                num_classes: int = 10) -> Union[List[float], torch.Tensor]:
        """
        Measure alignment between activations and targets.
        
        This method handles both:
        1. List of activations from different layers (returns a list of scores)
        2. Single activation tensor (returns a single tensor of scores)
        
        Args:
            activations: Layer activations (list for multiple layers or tensor for single layer)
            targets: Target labels (optional, used for class-based metrics)
            num_classes: Number of classes in the dataset
            
        Returns:
            List of alignment values per layer or tensor of values for a single layer
        """
        if isinstance(activations, list):
            # Multiple layers - compute for each and return list
            results = []
            
            for layer_activations in activations:
                # Handle class-specific metrics
                if self.name.lower() == "class_separation" and targets is not None:
                    result = self._compute_class_separation(
                        layer_activations, targets, num_classes
                    )
                else:
                    # For metrics that don't use targets, we pass a dummy weight matrix
                    # TODO: Implement proper handling for all metrics
                    if layer_activations.dim() > 2:
                        layer_activations = layer_activations.reshape(layer_activations.size(0), -1)
                    
                    # Use a dummy identity matrix as weights
                    dummy_weights = torch.eye(
                        layer_activations.size(1), 
                        device=layer_activations.device
                    )
                    
                    # Compute using the lower-level API
                    result = AlignmentMetrics.measure(
                        layer_activations, dummy_weights, 
                        method=self.name
                    ).mean().item()
                
                results.append(result)
                
            return results
        else:
            # Single layer - return tensor of scores
            return self.compute_per_node_scores(activations, None)
    
    def _compute_class_separation(self, 
                                activations: torch.Tensor, 
                                targets: torch.Tensor, 
                                num_classes: int) -> float:
        """
        Compute separation between class activations.
        
        Args:
            activations: Layer activations
            targets: Target labels
            num_classes: Number of classes in the dataset
            
        Returns:
            Class separation score
        """
        if activations.dim() > 2:
            activations = activations.reshape(activations.size(0), -1)
        
        # Compute class means
        class_means = []
        for c in range(num_classes):
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_mean = activations[class_mask].mean(dim=0)
                class_means.append(class_mean)
        
        if len(class_means) > 1:
            # Compute mean distance between class centers
            class_means = torch.stack(class_means)
            dists = torch.cdist(class_means, class_means)
            # Mask the diagonal
            mask = torch.ones_like(dists) - torch.eye(dists.size(0), device=dists.device)
            return (dists * mask).sum().item() / (mask.sum().item() + 1e-8)
        else:
            return 0.0
    
    def compute_per_node_scores(
        self,
        layer_input: torch.Tensor,
        layer_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute a per-node alignment score for each node's weight vector w_i.
        
        Args:
            layer_input: 2D tensor of shape (N, input_dim) with the data
            layer_weights: 2D tensor (#nodes, input_dim), or None to use identity
            device: PyTorch device to ensure everything is on the correct device
            
        Returns:
            A 1D tensor (#nodes,) of alignment scores
        """
        if device is None:
            device = layer_input.device if isinstance(layer_input, torch.Tensor) else torch.device('cpu')
        
        # Move input to device if needed
        if isinstance(layer_input, torch.Tensor):
            layer_input = layer_input.to(device)
        
        # Basic checks
        if layer_input.dim() != 2:
            # For CNN, flatten the input
            orig_shape = layer_input.shape
            layer_input = layer_input.reshape(orig_shape[0], -1)
        
        # If weights are not provided, use identity matrix
        if layer_weights is None:
            # For RQ and similar metrics, weights must have compatible shape
            layer_weights = torch.eye(layer_input.size(1), device=device)
        else:
            # Move weights to device
            layer_weights = layer_weights.to(device)
        
        # Dispatch to appropriate method based on metric name
        if self.name.lower() == "rq":
            return self._compute_rq(layer_input, layer_weights, device)
        elif self.name.lower().startswith("mi"):
            return self._compute_mi(layer_input, layer_weights, device)
        elif self.name.lower() == "rank":
            metric = RankAlignmentMetric()
            return metric.measure(layer_input, layer_weights)
        elif self.name.lower() == "null_space":
            metric = NullSpaceAlignmentMetric()
            return metric.measure(layer_input, layer_weights)
        else:
            # Default to RQ as a fallback
            return self._compute_rq(layer_input, layer_weights, device)
    
    def _compute_rq(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient per node.
        
        Args:
            X: Input data (N, input_dim)
            W: Weight matrix (num_nodes, input_dim)
            device: Compute device
            
        Returns:
            Per-node RQ scores (num_nodes,)
        """
        # Center the input
        N = X.size(0)
        if N < 2:
            # fallback if dataset is extremely small
            return torch.zeros(W.size(0), device=device)

        X_centered = X - X.mean(dim=0, keepdim=True)
        # sample covariance => (input_dim x input_dim)
        cov_x = (X_centered.t() @ X_centered) / (N - 1)

        # Possibly scale by norm
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
        
        # Scale by sqrt(d) if using relative RQ
        if False:  # TODO: Add a relative parameter
            d = W.size(1)
            rq_scores = rq_scores * np.sqrt(d)
            
        return rq_scores

    def _compute_mi(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute mutual information-based scores.
        
        Args:
            X: Input data (N, input_dim)
            W: Weight matrix (num_nodes, input_dim)
            device: Compute device
            
        Returns:
            Per-node MI scores (num_nodes,)
        """
        # For now, we delegate to the AlignmentMetrics class
        return AlignmentMetrics.measure(X, W, method="MI")


# Function to get an alignment metric instance
def get_metric(name: str, scale_by_norm: bool = False) -> AlignmentMetric:
    """
    Get alignment metric by name.
    
    Args:
        name: Name of the metric ('RQ', 'MI', 'rank', 'null_space', etc.)
        scale_by_norm: Whether to scale the covariance or final measure by norm
        
    Returns:
        AlignmentMetric instance
    """
    return AlignmentMetric(name=name, scale_by_norm=scale_by_norm)


# Register a new metric with the registry
def register_metric(name: str, metric_class: AlignmentMetricBase) -> None:
    """
    Register a new alignment metric.
    
    Args:
        name: Name to register the metric under
        metric_class: Class implementing the AlignmentMetricBase interface
    """
    AlignmentMetrics.register(name, metric_class)


# Legacy function for backward compatibility
def alignment(inputs: torch.Tensor, weights: torch.Tensor, 
             method: str = "RQ", relative: bool = True, **kwargs) -> torch.Tensor:
    """
    Compute alignment between inputs and weights.
    
    Args:
        inputs: Input activations tensor
        weights: Weight tensor
        method: Alignment method
        relative: Whether to use relative alignment (for RQ)
        
    Returns:
        Tensor containing alignment values
    """
    kwargs["relative"] = relative
    return AlignmentMetrics.measure(inputs, weights, method=method, **kwargs) 