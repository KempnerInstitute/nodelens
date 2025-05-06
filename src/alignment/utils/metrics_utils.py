"""
Metrics utility functions for the alignment package.

This module provides the core implementations of alignment metrics that
are used by both metrics.py and alignment_metrics.py to avoid circular dependencies.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Union, Optional


class AlignmentMetricBase:
    """Base class for all alignment metrics."""

    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure the alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            **kwargs: Additional parameters specific to metric
            
        Returns:
            Tensor containing alignment values
        """
        raise NotImplementedError("Subclasses must implement measure")
    
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric."""
        return cls.__name__


class RQMetric(AlignmentMetricBase):
    """Representation Quality (RQ) alignment metric."""

    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, 
                relative: bool = True, epsilon: float = 1e-8, **kwargs) -> torch.Tensor:
        """
        Measure the representation quality (RQ) alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor (batch, features)
            weights: Weight tensor (output, features)
            relative: Whether to use relative alignment
            epsilon: Small value to prevent division by zero
            
        Returns:
            Tensor containing RQ values per weight vector
        """
        # Ensure inputs have at least 2 dimensions
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
            
        # Move weights to same device as inputs
        weights = weights.to(inputs.device)
        
        # Center the inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
        
        # Add small value to diagonal for stability
        cov = cov + torch.eye(cov.size(0), device=cov.device) * epsilon
        
        # Compute the RQ values
        numerator = torch.sum(weights * torch.matmul(weights, cov), dim=1)
        denominator = (torch.norm(weights, dim=1) ** 2) * (torch.norm(weights @ cov, dim=1) + epsilon)
        
        # Calculate RQ as cosine similarity between weight vectors and weight @ covariance
        rq = numerator / denominator
        
        if relative:
            # Make RQ values relative to random vectors in high dimensions (expected value is 1/sqrt(d))
            d = weights.size(1)
            rq = rq * np.sqrt(d)
            
        return rq


class MIMetric(AlignmentMetricBase):
    """Mutual Information (MI) alignment metric."""

    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, 
               bins: int = 50, epsilon: float = 1e-8, **kwargs) -> torch.Tensor:
        """
        Measure the mutual information between input dimensions and weight vectors.
        
        Args:
            inputs: Input activations tensor (batch, features)
            weights: Weight tensor (output, features)
            bins: Number of bins for histogram
            epsilon: Small value to prevent division by zero
            
        Returns:
            Tensor containing MI values per weight vector
        """
        # Implementation of MI calculation
        # Normalize inputs and weights
        X = (inputs - inputs.min()) / (inputs.max() - inputs.min() + epsilon)
        W_norm = weights / (torch.norm(weights, dim=1, keepdim=True) + epsilon)
        
        # Calculate projections
        projections = torch.matmul(X, W_norm.t())
        
        # Calculate MI for each weight vector
        mi_values = []
        for i in range(projections.size(1)):
            proj = projections[:, i]
            hist_x, _ = torch.histogram(proj, bins=bins, density=True)
            # Convert histogram to probability distribution
            hist_x = hist_x / torch.sum(hist_x)
            
            # Calculate entropy
            entropy = -torch.sum(hist_x * torch.log2(hist_x + epsilon))
            mi_values.append(entropy)
            
        return torch.tensor(mi_values, device=weights.device)


class WeightSimilarityMetric(AlignmentMetricBase):
    """Measures similarity between weight vectors."""

    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, 
               metric: str = "cosine", **kwargs) -> torch.Tensor:
        """
        Measure similarity between weight vectors.
        
        Args:
            inputs: Not used for this metric
            weights: Weight tensor (output, features)
            metric: Similarity metric to use (cosine, dot, euclidean)
            
        Returns:
            Tensor containing pairwise similarity values
        """
        # Normalize weights for cosine similarity
        if metric == "cosine":
            w_norm = weights / torch.norm(weights, dim=1, keepdim=True)
            similarity = torch.mm(w_norm, w_norm.t())
        elif metric == "dot":
            similarity = torch.mm(weights, weights.t())
        elif metric == "euclidean":
            similarity = torch.cdist(weights, weights)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
            
        return similarity


class NodeRedundancyMetric(AlignmentMetricBase):
    """Measures redundancy between nodes."""

    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure redundancy between nodes based on activations and weights.
        
        Args:
            inputs: Input activations tensor (batch, features)
            weights: Weight tensor (output, features)
            
        Returns:
            Tensor containing redundancy values per node
        """
        # Center inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute correlation matrix of activations
        corr_matrix = torch.corrcoef(X.t())
        
        # Replace NaN values with zeros
        corr_matrix = torch.nan_to_num(corr_matrix)
        
        # For each weight vector, compute average absolute correlation with other nodes
        redundancy = torch.mean(torch.abs(corr_matrix), dim=1)
        
        return redundancy


class AlignmentMetricsFactory:
    """Factory class for accessing different alignment metrics."""
    
    # Registry of available metrics
    _registry = {
        "RQ": RQMetric,
        "MI": MIMetric,
        "weight_similarity": WeightSimilarityMetric,
        "redundancy": NodeRedundancyMetric,
    }
    
    @classmethod
    def register(cls, name: str, metric_class: AlignmentMetricBase) -> None:
        """Register a new alignment metric."""
        cls._registry[name] = metric_class
    
    @classmethod
    def measure(cls, inputs: torch.Tensor, weights: torch.Tensor, 
               method: str = "RQ", **kwargs) -> torch.Tensor:
        """
        Measure alignment using specified method.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            method: Name of alignment metric to use
            **kwargs: Additional parameters for the metric
            
        Returns:
            Tensor containing alignment values
        """
        method = method.lower()
        
        # Map method names for backward compatibility
        method_map = {
            "rq": "RQ",
            "mi": "MI",
            "weight_similarity": "weight_similarity",
            "redundancy": "redundancy",
        }
        
        normalized_method = method_map.get(method, method)
        normalized_method = normalized_method.upper() if normalized_method in ["rq", "mi"] else normalized_method
        
        if normalized_method not in cls._registry:
            raise ValueError(f"Unknown alignment metric: {method}")
            
        return cls._registry[normalized_method].measure(inputs, weights, **kwargs)


# Standalone function for backward compatibility
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
    return AlignmentMetricsFactory.measure(inputs, weights, method=method, **kwargs) 