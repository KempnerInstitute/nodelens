# src/alignment/metrics.py

"""
Alignment metrics for neural network analysis.

This module provides classes and functions for measuring alignment between
weight matrices and input (or output) activations, enabling node-wise scoring
for pruning and other experiments.

By default, we include:
- RQ (Rayleigh Quotient) as a canonical example
- MI_0 (placeholder for a mutual information approach)
- Possibly more advanced or custom metrics.

Each metric can implement a method `compute_per_node_scores(layer_input, layer_weights, device=...)`
returning a 1D tensor (#nodes_in_layer,).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union

################################################################################
# Example base metric class
################################################################################

class AlignmentMetric:
    """
    A base (or generic) alignment metric class.

    Subclasses or usage can specify the internal method (e.g. "rq", "mi", etc.).
    Or we can keep it all in one class with an if/else in compute_per_node_scores.

    Usage:
        metric = AlignmentMetric(name="rq", scale_by_norm=False)
        node_scores = metric.compute_per_node_scores(layer_input, layer_weights, device=device)
    """

    def __init__(self, name: str = "rq", scale_by_norm: bool = False):
        """
        Initialize with a metric name and optional scaling.

        Args:
            name: The metric to compute. ("rq", "mi", or any custom string).
            scale_by_norm: Whether to scale the covariance or final measure by norm
                           (some variants of RQ do that).
        """
        self.name = name.lower()
        self.scale_by_norm = scale_by_norm

    def compute_per_node_scores(
        self,
        layer_input: torch.Tensor,   # shape (N, input_dim) or with conv flattening
        layer_weights: torch.Tensor, # shape (num_nodes, input_dim)
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute a per-node alignment score for each node's weight vector w_i.

        Return shape: (num_nodes,).

        This method can implement RQ, MI, or any other approach. You can do a 
        big if/else or separate classes. Here, we do if/else for simplicity.

        For RQ:
            RQ_i = (w_i^T Cov(X) w_i) / (w_i^T w_i), optionally scaled or adjusted.
        For MI_x:
            Possibly an alternate measure.

        Args:
            layer_input: 2D tensor of shape (N, input_dim) with the (centered) data if needed.
            layer_weights: 2D tensor (#nodes, input_dim).
            device: PyTorch device to ensure everything is on the correct device.

        Returns:
            A 1D tensor (#nodes,) of alignment scores.
        """
        if device is None:
            device = layer_input.device

        # Basic checks
        if layer_input.dim() != 2:
            # For CNN, you might have already flattened each patch or done an "unfold".
            # If not, do so here or raise an error:
            raise ValueError(f"layer_input must be 2D, got shape {layer_input.shape}")

        # Move weights if not on the same device
        layer_weights = layer_weights.to(device)

        # Depending on the metric name, do your computation:
        if self.name == "rq":
            return self._compute_rq(layer_input, layer_weights, device)
        elif self.name.startswith("mi"):
            return self._compute_mi_placeholder(layer_input, layer_weights, device)
        else:
            # Default fallback => just do RQ or throw
            # return self._compute_rq(layer_input, layer_weights, device)
            raise ValueError(f"Unknown metric name '{self.name}' in AlignmentMetric.")

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


################################################################################
# If you want a quick helper for "get_metric" style usage:
################################################################################

def get_metric(metric_name: str, scale_by_norm: bool = False) -> AlignmentMetric:
    """
    Factory method to return an AlignmentMetric instance with the requested metric.

    Args:
        metric_name: e.g. 'rq', 'mi_0', 'mi_1', etc.
        scale_by_norm: boolean for scaling covariance or the final measure.

    Returns:
        AlignmentMetric instance
    """
    metric_name = metric_name.lower()
    return AlignmentMetric(name=metric_name, scale_by_norm=scale_by_norm)