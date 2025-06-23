"""
GPU-accelerated utilities for fast computation of alignment metrics.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


def gpu_histogram1d(
    data: torch.Tensor,
    bins: int = 10,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    density: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated 1D histogram computation.
    
    Args:
        data: Input tensor to bin
        bins: Number of bins
        min_val: Minimum value for binning range
        max_val: Maximum value for binning range
        density: If True, normalize to density
        
    Returns:
        hist: Histogram counts
        bin_edges: Bin edge values
    """
    if min_val is None:
        min_val = data.min().item()
    if max_val is None:
        max_val = data.max().item()
    
    # Create bin edges
    bin_edges = torch.linspace(min_val, max_val, bins + 1, device=data.device)
    
    # Compute bin indices for each data point
    bin_width = (max_val - min_val) / bins
    indices = ((data - min_val) / bin_width).long()
    indices = indices.clamp(0, bins - 1)
    
    # Count occurrences using scatter_add
    hist = torch.zeros(bins, dtype=torch.float32, device=data.device)
    ones = torch.ones_like(data, dtype=torch.float32)
    hist.scatter_add_(0, indices.flatten(), ones.flatten())
    
    if density:
        # Normalize to density
        hist = hist / (hist.sum() * bin_width)
    
    return hist, bin_edges


def gpu_histogram2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: Union[int, Tuple[int, int]] = 10,
    range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    density: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated 2D histogram computation.
    
    Args:
        x: First variable
        y: Second variable
        bins: Number of bins (can be different for x and y)
        range: Range for binning ((xmin, xmax), (ymin, ymax))
        density: If True, normalize to density
        
    Returns:
        hist: 2D histogram
        x_edges: Bin edges for x
        y_edges: Bin edges for y
    """
    if isinstance(bins, int):
        x_bins = y_bins = bins
    else:
        x_bins, y_bins = bins
    
    if range is None:
        x_min, x_max = x.min().item(), x.max().item()
        y_min, y_max = y.min().item(), y.max().item()
    else:
        (x_min, x_max), (y_min, y_max) = range
    
    # Create bin edges
    x_edges = torch.linspace(x_min, x_max, x_bins + 1, device=x.device)
    y_edges = torch.linspace(y_min, y_max, y_bins + 1, device=y.device)
    
    # Compute bin indices
    x_width = (x_max - x_min) / x_bins
    y_width = (y_max - y_min) / y_bins
    
    x_indices = ((x - x_min) / x_width).long().clamp(0, x_bins - 1)
    y_indices = ((y - y_min) / y_width).long().clamp(0, y_bins - 1)
    
    # Convert to linear indices
    linear_indices = y_indices * x_bins + x_indices
    
    # Count occurrences
    hist = torch.zeros(y_bins * x_bins, dtype=torch.float32, device=x.device)
    ones = torch.ones_like(x, dtype=torch.float32)
    hist.scatter_add_(0, linear_indices.flatten(), ones.flatten())
    
    # Reshape to 2D
    hist = hist.reshape(y_bins, x_bins)
    
    if density:
        hist = hist / (hist.sum() * x_width * y_width)
    
    return hist, x_edges, y_edges


def gpu_mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int = 10,
    method: str = "histogram"
) -> torch.Tensor:
    """
    GPU-accelerated mutual information computation.
    
    Args:
        x: First variable
        y: Second variable
        bins: Number of bins for discretization
        method: Method to use ('histogram' or 'kraskov')
        
    Returns:
        Mutual information value
    """
    if method == "histogram":
        # Compute 2D histogram
        hist_xy, _, _ = gpu_histogram2d(x, y, bins=bins)
        hist_xy = hist_xy + 1e-10  # Add small constant
        
        # Normalize to joint probability
        p_xy = hist_xy / hist_xy.sum()
        
        # Compute marginals
        p_x = p_xy.sum(dim=0)
        p_y = p_xy.sum(dim=1)
        
        # Compute MI: sum(p_xy * log(p_xy / (p_x * p_y)))
        p_x_p_y = p_x.unsqueeze(0) * p_y.unsqueeze(1)
        
        # Use log trick to avoid log(0)
        log_term = torch.where(
            p_xy > 1e-10,
            torch.log(p_xy / (p_x_p_y + 1e-10)),
            torch.tensor(0.0, device=x.device)
        )
        
        mi = (p_xy * log_term).sum()
        
        return mi
    
    elif method == "kraskov":
        # Simplified Kraskov estimator
        # This is a basic version - full implementation would be more complex
        
        n = x.shape[0]
        k = min(3, n // 10)  # Number of neighbors
        
        # Standardize
        x_std = (x - x.mean()) / (x.std() + 1e-8)
        y_std = (y - y.mean()) / (y.std() + 1e-8)
        
        # Stack for joint space
        xy = torch.stack([x_std, y_std], dim=1)
        
        # Compute distances in joint space
        distances = torch.cdist(xy, xy)
        
        # Get k-th nearest neighbor distances
        kth_distances, _ = distances.topk(k + 1, largest=False, dim=1)
        epsilon = kth_distances[:, -1]  # k-th neighbor distance
        
        # Count neighbors in marginal spaces
        x_distances = torch.abs(x_std.unsqueeze(1) - x_std.unsqueeze(0))
        y_distances = torch.abs(y_std.unsqueeze(1) - y_std.unsqueeze(0))
        
        n_x = (x_distances < epsilon.unsqueeze(1)).sum(dim=1).float()
        n_y = (y_distances < epsilon.unsqueeze(1)).sum(dim=1).float()
        
        # Digamma function approximation
        def digamma_approx(x):
            return x.log() - 1.0 / (2.0 * x)
        
        # MI estimation
        mi = digamma_approx(torch.tensor(k, dtype=torch.float32)) + \
             digamma_approx(torch.tensor(n, dtype=torch.float32)) - \
             digamma_approx(n_x).mean() - digamma_approx(n_y).mean()
        
        return mi.clamp(min=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def gpu_entropy(
    data: torch.Tensor,
    bins: int = 10,
    method: str = "histogram"
) -> torch.Tensor:
    """
    GPU-accelerated entropy computation.
    
    Args:
        data: Input data
        bins: Number of bins
        method: Method to use ('histogram' or 'gaussian')
        
    Returns:
        Entropy value
    """
    if method == "histogram":
        hist, _ = gpu_histogram1d(data, bins=bins)
        
        # Normalize to probability
        p = hist / hist.sum()
        p = p + 1e-10  # Add small constant
        
        # Compute entropy
        entropy = -(p * p.log()).sum()
        
        return entropy
    
    elif method == "gaussian":
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * var)
        var = data.var()
        entropy = 0.5 * torch.log(2 * math.pi * math.e * var)
        
        return entropy
    
    else:
        raise ValueError(f"Unknown method: {method}")


def gpu_conditional_entropy(
    x: torch.Tensor,
    y: torch.Tensor,
    condition: torch.Tensor,
    bins: int = 10
) -> torch.Tensor:
    """
    GPU-accelerated conditional entropy H(X,Y|Z).
    
    Args:
        x: First variable
        y: Second variable
        condition: Conditioning variable
        bins: Number of bins
        
    Returns:
        Conditional entropy
    """
    # Create 3D histogram
    n = x.shape[0]
    
    # Compute ranges
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()
    z_min, z_max = condition.min().item(), condition.max().item()
    
    # Compute bin indices
    x_idx = ((x - x_min) / ((x_max - x_min) / bins)).long().clamp(0, bins - 1)
    y_idx = ((y - y_min) / ((y_max - y_min) / bins)).long().clamp(0, bins - 1)
    z_idx = ((condition - z_min) / ((z_max - z_min) / bins)).long().clamp(0, bins - 1)
    
    # Linear indices for 3D histogram
    indices = z_idx * bins * bins + y_idx * bins + x_idx
    
    # Count occurrences
    hist_xyz = torch.zeros(bins ** 3, device=x.device)
    ones = torch.ones(n, device=x.device)
    hist_xyz.scatter_add_(0, indices, ones)
    hist_xyz = hist_xyz.reshape(bins, bins, bins) + 1e-10
    
    # Normalize to probability
    p_xyz = hist_xyz / hist_xyz.sum()
    
    # Marginal p(z)
    p_z = p_xyz.sum(dim=(1, 2))
    
    # Conditional entropy
    h_xy_given_z = 0
    
    for z in range(bins):
        if p_z[z] > 1e-10:
            # p(x,y|z) = p(x,y,z) / p(z)
            p_xy_given_z = p_xyz[z] / p_z[z]
            
            # H(X,Y|Z=z)
            log_term = torch.where(
                p_xy_given_z > 1e-10,
                p_xy_given_z.log(),
                torch.tensor(0.0, device=x.device)
            )
            h_xy_z = -(p_xy_given_z * log_term).sum()
            
            # Weight by p(z)
            h_xy_given_z += p_z[z] * h_xy_z
    
    return h_xy_given_z


class GPUAcceleratedMetrics:
    """
    Collection of GPU-accelerated metric computations.
    """
    
    @staticmethod
    @torch.jit.script
    def fast_covariance(X: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled fast covariance computation.
        
        Args:
            X: Input tensor (n_samples, n_features)
            
        Returns:
            Covariance matrix
        """
        X_centered = X - X.mean(dim=0, keepdim=True)
        cov = X_centered.T @ X_centered / (X.shape[0] - 1)
        return cov
    
    @staticmethod  
    @torch.jit.script
    def fast_correlation(X: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled fast correlation computation.
        
        Args:
            X: Input tensor (n_samples, n_features)
            
        Returns:
            Correlation matrix
        """
        # Standardize
        X_std = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)
        corr = X_std.T @ X_std / (X.shape[0] - 1)
        return corr 