"""
Alignment metrics for neural network analysis.

DEPRECATED: This module is deprecated and will be removed in a future version.
Please import directly from `alignment.metrics` instead.

This module provides various metrics for measuring alignment between weight vectors
and activation vectors, including RQ (representation quality), MI (mutual information),
and other metrics for analyzing neural network representations.
"""

import warnings
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable

# Show deprecation warning when the module is imported
warnings.warn(
    "The alignment.alignment_metrics module is deprecated and will be removed in a future version. "
    "Please import directly from alignment.metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

# Original classes for backward compatibility
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


class AlignmentMetrics:
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
        if method not in cls._registry:
            raise ValueError(f"Unknown alignment metric: {method}")
            
        return cls._registry[method].measure(inputs, weights, **kwargs)
    
    @classmethod
    def measure_methods(cls, network, inputs: torch.Tensor, 
                      methods: List[str], precomputed: bool = False) -> List[Dict[str, torch.Tensor]]:
        """
        Measure multiple alignment metrics for a network.
        
        Args:
            network: Neural network to analyze
            inputs: Input tensor to the network
            methods: List of alignment metrics to compute
            precomputed: Whether inputs have been precomputed
            
        Returns:
            List of dictionaries containing alignment values for each layer and method
        """
        layer_inputs = network.get_layer_inputs(inputs, precomputed=precomputed)
        preprocessed = network._preprocess_inputs(layer_inputs, compress_convolutional=True)
        weights = network.get_alignment_weights(flatten=True)
        
        all_layer_results = []
        for idx, (inp, w) in enumerate(zip(preprocessed, weights)):
            metrics_dict = {}
            for m in methods:
                if network.cnn_mode == "patchwise" and inp.ndim == 3:
                    val = cls.patchwise_alignment(inp, w, method=m, weigh_by_var=True)
                elif m == "delta_alignment":
                    val = cls.delta_alignment(network, idx, inp)
                else:
                    val = cls.measure(inp, w, method=m)
                metrics_dict[m] = val
            all_layer_results.append(metrics_dict)
        
        return all_layer_results
    
    @classmethod
    def patchwise_alignment(cls, inputs: torch.Tensor, weights: torch.Tensor, 
                          method: str = "RQ", weigh_by_var: bool = True) -> torch.Tensor:
        """
        Compute alignment metric for convolutional layers patch by patch.
        
        Args:
            inputs: Input tensor with shape (batch, patches, features)
            weights: Weight tensor
            method: Alignment method to use
            weigh_by_var: Whether to weight by variance of each patch
            
        Returns:
            Tensor containing alignment values
        """
        if inputs.dim() != 3:
            raise ValueError(f"Expected 3D tensor for patchwise alignment, got {inputs.dim()}D")
            
        batch_size, num_patches, num_features = inputs.shape
        
        # Compute alignment for each patch separately
        patch_alignments = []
        patch_vars = []
        
        for p in range(num_patches):
            patch_data = inputs[:, p, :]
            alignment = cls.measure(patch_data, weights, method=method)
            patch_alignments.append(alignment)
            
            if weigh_by_var:
                # Calculate variance of this patch for weighting
                patch_vars.append(torch.var(patch_data))
        
        # Stack patch alignments
        patch_alignments = torch.stack(patch_alignments, dim=0)
        
        if weigh_by_var:
            # Weight by variance
            patch_vars = torch.stack(patch_vars)
            weights = patch_vars / torch.sum(patch_vars)
            weighted_alignment = torch.sum(patch_alignments * weights.unsqueeze(1), dim=0)
            return weighted_alignment
        else:
            # Simple average across patches
            return torch.mean(patch_alignments, dim=0)
    
    @classmethod
    def delta_alignment(cls, network, layer_idx: int, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute delta alignment (change in alignment from previous checkpoint).
        
        Args:
            network: Neural network to analyze
            layer_idx: Index of layer to analyze
            inputs: Input tensor
            
        Returns:
            Tensor containing delta alignment values
        """
        # Implementation depends on having access to previous weights
        # This would need to be adapted based on how previous weights are stored
        current_weights = network.get_alignment_weights()[layer_idx]
        prev_weights = network.previous_weights[layer_idx] if hasattr(network, 'previous_weights') else current_weights
        
        # Calculate alignment with current and previous weights
        current_align = cls.measure(inputs, current_weights, method="RQ")
        prev_align = cls.measure(inputs, prev_weights, method="RQ")
        
        # Return delta
        return current_align - prev_align
    
    @classmethod
    def compute_eigenvalues(cls, inputs: torch.Tensor, centered: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalues and eigenvectors of input covariance matrix.
        
        Args:
            inputs: Input tensor
            centered: Whether to center the data before computing covariance
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if centered:
            X = inputs - inputs.mean(dim=0, keepdim=True)
        else:
            X = inputs
            
        # Compute covariance matrix
        cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    @classmethod
    def measure_expected_distribution(cls, method: str, eigenvalues: torch.Tensor, 
                                    bins: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Measure the expected distribution of an alignment metric.
        
        Args:
            method: Alignment method name
            eigenvalues: Eigenvalues of input covariance matrix
            bins: Number of bins for histogram
            
        Returns:
            Tuple of (counts, edges) for histogram
        """
        # Compute expected distribution based on eigenvalues
        if method == "RQ":
            # For RQ, sample random vectors and compute alignment
            n_samples = 10000
            d = eigenvalues.size(0)
            
            # Generate random vectors
            random_vectors = torch.randn(n_samples, d, device=eigenvalues.device)
            random_vectors = random_vectors / torch.norm(random_vectors, dim=1, keepdim=True)
            
            # Compute expected RQ values
            rq_values = []
            for vec in random_vectors:
                # Compute RQ for random vector with eigenvalues
                rq = torch.sum(vec**2 * eigenvalues) / (torch.norm(vec)**2 * torch.norm(vec * eigenvalues))
                rq_values.append(rq.item())
                
            # Compute histogram
            counts, edges = torch.histogram(
                torch.tensor(rq_values, device=eigenvalues.device),
                bins=bins,
                density=True
            )
            
            return counts, edges
        else:
            # Default implementation for other metrics
            # This would need to be customized based on what's expected
            # For now, return a uniform distribution
            counts = torch.ones(bins, device=eigenvalues.device) / bins
            edges = torch.linspace(0, 1, bins+1, device=eigenvalues.device)
            return counts, edges


# For backward compatibility
def alignment(inputs: torch.Tensor, weights: torch.Tensor, 
             method: str = "RQ", relative: bool = True, **kwargs) -> torch.Tensor:
    """
    Compute alignment between inputs and weights.
    Backward-compatible function with the original API.
    
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