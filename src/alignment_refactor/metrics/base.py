"""Base module for metrics."""

from typing import Optional
import torch
from alignment_refactor.core.base import BaseMetric

@torch.no_grad()
def compute_covariance(X: torch.Tensor, force_cpu: bool = False) -> torch.Tensor:
    """Compute covariance matrix."""
    if force_cpu and X.is_cuda:
        X = X.cpu()
    if X.ndim == 1:
        X = X.unsqueeze(0)
    if X.shape[0] < 2:
        return torch.zeros((X.shape[1], X.shape[1]), device=X.device, dtype=X.dtype)
    X_centered = X - X.mean(dim=0, keepdim=True)
    return torch.matmul(X_centered.T, X_centered) / (X.shape[0] - 1) 