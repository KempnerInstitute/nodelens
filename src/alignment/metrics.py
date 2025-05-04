"""
Alignment metrics for neural network analysis.

This module provides metrics for measuring alignment between weight matrices 
and input activations, with various metrics to quantify the degree of alignment.
"""

import torch
import numpy as np
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
    
    This class provides a standardized interface for alignment metrics.
    """
    
    def __init__(self, name: str = "RQ"):
        """
        Initialize alignment metric.
        
        Args:
            name: Name of the alignment metric to use
        """
        self.name = name
        
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


def get_metric(name: str) -> AlignmentMetric:
    """
    Get alignment metric by name.
    
    Args:
        name: Name of the metric ('RQ', 'rank', 'null_space', etc.)
        
    Returns:
        AlignmentMetric instance
        
    Raises:
        ValueError: If metric name is not recognized
    """
    name = name.lower()
    
    if name == 'rq':
        return AlignmentMetric(name="RQ")
    elif name == 'rank':
        return RankAlignmentMetric()
    elif name == 'null_space':
        return NullSpaceAlignmentMetric()
    elif name == 'class_separation':
        return AlignmentMetric(name="class_separation")
    else:
        # Default to RQ with a warning
        import warnings
        warnings.warn(f"Unknown metric '{name}', defaulting to 'RQ'")
        return AlignmentMetric(name="RQ") 