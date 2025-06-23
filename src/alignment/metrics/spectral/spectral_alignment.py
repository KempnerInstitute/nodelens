"""Spectral alignment metrics based on eigenvalue analysis."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import numpy as np
from ..base import AlignmentMetric


class SpectralGapMetric(AlignmentMetric):
    """
    Measures the spectral gap of weight matrices.
    
    The spectral gap is the difference between the largest and second-largest
    eigenvalues, normalized by the largest eigenvalue. A larger spectral gap
    indicates a more dominant principal component.
    """
    
    requires_inputs = False
    requires_weights = True
    requires_outputs = False
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize by the largest eigenvalue
        """
        super().__init__()
        self.normalize = normalize
    
    def compute(self, 
                inputs: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                outputs: Optional[torch.Tensor] = None) -> float:
        """Compute spectral gap of weight matrix."""
        if weights is None:
            raise ValueError("Weights required for spectral gap metric")
        
        # Handle different weight shapes
        if weights.dim() == 4:  # Conv weights
            # Reshape to 2D: (out_channels, in_channels * kernel_size)
            weights = weights.view(weights.size(0), -1)
        elif weights.dim() == 3:  # e.g., attention weights
            # Take mean over batch dimension
            weights = weights.mean(dim=0)
        elif weights.dim() != 2:
            raise ValueError(f"Unsupported weight dimension: {weights.dim()}")
        
        # Compute SVD (more stable than eigendecomposition)
        try:
            U, S, V = torch.svd(weights)
            
            if S.numel() < 2:
                return 0.0
            
            # Spectral gap is difference between top 2 singular values
            gap = (S[0] - S[1]).item()
            
            if self.normalize and S[0] > 1e-8:
                gap = gap / S[0].item()
            
            return float(gap)
            
        except Exception as e:
            # Return 0 if SVD fails
            return 0.0


class EigenvalueAlignmentMetric(AlignmentMetric):
    """
    Measures alignment between eigenvalue distributions of two weight matrices.
    
    Uses Wasserstein distance between eigenvalue distributions as a measure
    of spectral similarity.
    """
    
    requires_inputs = False
    requires_weights = True
    requires_outputs = False
    
    def __init__(self, p: float = 2.0, top_k: Optional[int] = None):
        """
        Args:
            p: Order of Wasserstein distance (default: 2.0)
            top_k: If specified, only compare top-k eigenvalues
        """
        super().__init__()
        self.p = p
        self.top_k = top_k
        self._reference_eigenvalues = None
    
    def set_reference(self, weights: torch.Tensor):
        """Set reference weight matrix for comparison."""
        self._reference_eigenvalues = self._compute_eigenvalues(weights)
    
    def _compute_eigenvalues(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues of weight matrix."""
        # Handle different weight shapes
        if weights.dim() == 4:  # Conv weights
            weights = weights.view(weights.size(0), -1)
        elif weights.dim() == 3:  # e.g., attention weights
            weights = weights.mean(dim=0)
        elif weights.dim() != 2:
            raise ValueError(f"Unsupported weight dimension: {weights.dim()}")
        
        # Compute singular values (eigenvalues of W^T W)
        _, S, _ = torch.svd(weights)
        
        if self.top_k is not None and S.numel() > self.top_k:
            S = S[:self.top_k]
        
        return S.sort(descending=True)[0]
    
    def compute(self,
                inputs: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                outputs: Optional[torch.Tensor] = None) -> float:
        """Compute eigenvalue alignment."""
        if weights is None:
            raise ValueError("Weights required for eigenvalue alignment")
        
        if self._reference_eigenvalues is None:
            # If no reference, return 0 (perfect alignment with self)
            return 0.0
        
        current_eigenvalues = self._compute_eigenvalues(weights)
        
        # Compute Wasserstein distance
        # For 1D distributions, this simplifies to sorted L^p distance
        n_ref = len(self._reference_eigenvalues)
        n_curr = len(current_eigenvalues)
        
        if n_ref != n_curr:
            # Pad shorter sequence with zeros
            max_len = max(n_ref, n_curr)
            ref_padded = torch.zeros(max_len, device=weights.device)
            curr_padded = torch.zeros(max_len, device=weights.device)
            ref_padded[:n_ref] = self._reference_eigenvalues
            curr_padded[:n_curr] = current_eigenvalues
        else:
            ref_padded = self._reference_eigenvalues
            curr_padded = current_eigenvalues
        
        # Wasserstein distance
        distance = torch.norm(ref_padded - curr_padded, p=self.p).item()
        
        return float(distance)


class SpectralClusteringAlignment(AlignmentMetric):
    """
    Measures how well weight matrix eigenspaces align with data clustering.
    
    This metric computes the alignment between the top eigenvectors of the
    weight matrix and the cluster structure in the output space.
    """
    
    requires_inputs = False
    requires_weights = True
    requires_outputs = True
    
    def __init__(self, n_components: int = 5, n_clusters: int = 10):
        """
        Args:
            n_components: Number of top eigenvectors to consider
            n_clusters: Number of clusters to form in output space
        """
        super().__init__()
        self.n_components = n_components
        self.n_clusters = n_clusters
    
    def compute(self,
                inputs: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                outputs: Optional[torch.Tensor] = None) -> float:
        """Compute spectral clustering alignment."""
        if weights is None or outputs is None:
            raise ValueError("Both weights and outputs required")
        
        # Handle different weight shapes
        if weights.dim() == 4:  # Conv weights
            weights = weights.view(weights.size(0), -1)
        elif weights.dim() == 3:
            weights = weights.mean(dim=0)
        elif weights.dim() != 2:
            raise ValueError(f"Unsupported weight dimension: {weights.dim()}")
        
        # Get top eigenvectors of weight matrix
        try:
            U, S, V = torch.svd(weights)
            top_components = V[:, :self.n_components]  # Right singular vectors
            
            # Project outputs using eigenvectors
            if outputs.size(-1) != top_components.size(0):
                # Dimension mismatch - return low alignment
                return 0.0
            
            projected = outputs @ top_components
            
            # Compute clustering quality in projected space
            # Use variance ratio as a simple measure
            total_var = outputs.var(dim=0).sum().item()
            projected_var = projected.var(dim=0).sum().item()
            
            if total_var > 1e-8:
                alignment = projected_var / total_var
            else:
                alignment = 0.0
            
            return float(alignment)
            
        except Exception as e:
            return 0.0


class PowerIterationAlignment(AlignmentMetric):
    """
    Measures alignment using power iteration convergence properties.
    
    This metric analyzes how quickly power iteration converges to the dominant
    eigenvector, which indicates the spectral structure of the weight matrix.
    """
    
    requires_inputs = False
    requires_weights = True
    requires_outputs = False
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Args:
            max_iterations: Maximum power iterations
            tolerance: Convergence tolerance
        """
        super().__init__()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def compute(self,
                inputs: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                outputs: Optional[torch.Tensor] = None) -> float:
        """Compute power iteration convergence rate."""
        if weights is None:
            raise ValueError("Weights required for power iteration alignment")
        
        # Handle different weight shapes
        if weights.dim() == 4:  # Conv weights
            weights = weights.view(weights.size(0), -1)
        elif weights.dim() == 3:
            weights = weights.mean(dim=0)
        elif weights.dim() != 2:
            raise ValueError(f"Unsupported weight dimension: {weights.dim()}")
        
        # Initialize random vector
        n = weights.size(0)
        v = torch.randn(n, 1, device=weights.device)
        v = v / torch.norm(v)
        
        # Power iteration
        convergence_rate = 0.0
        prev_v = v.clone()
        
        for i in range(self.max_iterations):
            # One power iteration step
            v = weights @ v
            v = v / torch.norm(v)
            
            # Check convergence
            diff = torch.norm(v - prev_v).item()
            if diff < self.tolerance:
                # Converged - compute convergence rate
                convergence_rate = 1.0 / (i + 1)
                break
            
            prev_v = v.clone()
        
        return float(convergence_rate) 