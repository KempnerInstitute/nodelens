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
    def measure(inputs: torch.Tensor, weights: torch.Tensor, bins: int = 30, eps: float = 1e-9, force_cpu_for_large_metric_ops: bool = True, **kwargs) -> torch.Tensor:
        """
        Measure the mutual information between input dimensions and weight vectors.
        
        Args:
            inputs: Input activations tensor (batch, features)
            weights: Weight tensor (output, features)
            bins: Number of bins for histogram
            eps: Small value to prevent division by zero
            force_cpu_for_large_metric_ops: Whether to force CPU for large metric operations
            
        Returns:
            Tensor containing MI values per weight vector
        """
        # `inputs` is X_processed_one_batch (potentially (N_patches, D_filter_features))
        # `weights` is w_flat (potentially (Out_filters, D_filter_features))
        # All metric-specific computations should ideally happen on CPU if inputs are large to avoid OOM.

        original_device = inputs.device # Device of the incoming processed activations
        mi_scores_device = weights.device # Final scores should be on original weights device

        # Determine if processing should be forced to CPU based on input size
        perform_on_cpu = False
        if force_cpu_for_large_metric_ops and inputs.is_cuda and ((inputs.shape[0] > 200_000 and inputs.shape[1] > 50) or (inputs.numel() > 10_000_000)):
            # logger.info(f"MI: Large input tensor X ({inputs.shape}) detected on CUDA. Offloading to CPU.") # Ensure logger is available
            perform_on_cpu = True

        current_inputs = inputs.cpu() if perform_on_cpu else inputs
        current_weights = weights.cpu() if perform_on_cpu else weights # Weights for projections
        
        # Normalization of inputs (on CPU if offloaded)
        min_in = current_inputs.min()
        max_in = current_inputs.max()
        # Handle cases where max_in == min_in to avoid division by zero or NaN
        if max_in == min_in:
            X_normalized = torch.zeros_like(current_inputs)
        else:
            X_normalized = (current_inputs - min_in) / (max_in - min_in + eps)
        
        # Normalize weights (on CPU if offloaded)
        norm_W = torch.norm(current_weights, dim=1, keepdim=True)
        W_norm = current_weights / (norm_W + eps)
        
        projections = torch.matmul(X_normalized, W_norm.t()) # Result is on CPU if offloaded
        
        mi_scores = torch.zeros(weights.shape[0], device=mi_scores_device) # Final scores on original device

        for i in range(projections.shape[1]):  
            x_node_projection = projections[:, i].contiguous().cpu() # Ensure it's on CPU for histogram
            # y_variable_for_mi needs to be on CPU for np.histogram2d
            y_variable_for_mi = current_inputs.mean(dim=1).cpu() # current_inputs is already on CPU if offloaded

            if x_node_projection.numel() == 0 or y_variable_for_mi.numel() == 0:
                # Score already 0, correctly placed on mi_scores_device by initialization
                continue 
            
            min_val_x, max_val_x = x_node_projection.min().item(), x_node_projection.max().item()
            min_val_y, max_val_y = y_variable_for_mi.min().item(), y_variable_for_mi.max().item()

            range_x_tuple = (min_val_x, max_val_x) if max_val_x > min_val_x + eps else None
            range_y_tuple = (min_val_y, max_val_y) if max_val_y > min_val_y + eps else None

            # torch.histogram expects CPU inputs if not implemented for CUDA for certain args
            if range_x_tuple is not None:
                hist_x, _ = torch.histogram(input=x_node_projection, bins=bins, range=range_x_tuple, density=True)
            else:
                hist_x, _ = torch.histogram(input=x_node_projection, bins=bins, density=True)
            
            if range_y_tuple is not None:
                hist_y, _ = torch.histogram(input=y_variable_for_mi, bins=bins, range=range_y_tuple, density=True)
            else:
                hist_y, _ = torch.histogram(input=y_variable_for_mi, bins=bins, density=True)

            if x_node_projection.shape[0] != y_variable_for_mi.shape[0] or range_x_tuple is None or range_y_tuple is None:
                continue # Score already 0
            
            try:
                hist_xy_np, _, _ = np.histogram2d(
                    x_node_projection.numpy().flatten(), 
                    y_variable_for_mi.numpy().flatten(), 
                    bins=bins, 
                    range=[list(range_x_tuple), list(range_y_tuple)], 
                    density=True
                )
                hist_xy = torch.from_numpy(hist_xy_np).float() # CPU tensor
            except Exception as e_hist2d:
                # logger.warning(f"MI node {i}: Error in np.histogram2d: {e_hist2d}")
                continue # Score already 0
            
            bin_width_x = (range_x_tuple[1] - range_x_tuple[0]) / bins if range_x_tuple else 1.0 
            bin_width_y = (range_y_tuple[1] - range_y_tuple[0]) / bins if range_y_tuple else 1.0
            
            px = hist_x * bin_width_x 
            py = hist_y * bin_width_y 
            pxy = hist_xy * bin_width_x * bin_width_y 

            px = torch.clamp(px, min=eps); py = torch.clamp(py, min=eps); pxy = torch.clamp(pxy, min=eps)
            
            h_x = -torch.sum(px[px>0] * torch.log2(px[px>0]))
            h_y = -torch.sum(py[py>0] * torch.log2(py[py>0]))
            h_xy = -torch.sum(pxy[pxy>0] * torch.log2(pxy[pxy>0]))
            
            mi_val = h_x + h_y - h_xy
            mi_scores[i] = torch.clamp(mi_val, min=0.0).to(mi_scores_device) # Move final scalar score to original device

        return mi_scores


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
        cls._registry[name.lower()] = metric_class
    
    @classmethod
    def measure(cls, inputs: torch.Tensor, weights: torch.Tensor, 
               method: str = "RQ", force_cpu_for_large_metric_ops: bool = True, **kwargs) -> torch.Tensor:
        """
        Measure alignment using specified method.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            method: Name of alignment metric to use
            force_cpu_for_large_metric_ops: Whether to force CPU for large metric operations
            **kwargs: Additional parameters for the metric
            
        Returns:
            Tensor containing alignment values
        """
        normalized_method_name = method.lower()
        # Map legacy/alternative names to current registered names if necessary
        # This simple mapping assumes registered names are consistent (e.g., "rq", "mi")
        
        metric_class_to_call = cls._registry.get(normalized_method_name)
        if metric_class_to_call is None:
            # Try uppercase for RQ/MI as they were before
            metric_class_to_call = cls._registry.get(method.upper())
        
        if metric_class_to_call is None:
            raise ValueError(f"Unknown alignment metric: {method}. Registered: {list(cls._registry.keys())}")
        
        # Pass the flag through kwargs if the specific metric's measure method expects it
        kwargs['force_cpu_for_large_metric_ops'] = force_cpu_for_large_metric_ops
        return metric_class_to_call.measure(inputs, weights, **kwargs)


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