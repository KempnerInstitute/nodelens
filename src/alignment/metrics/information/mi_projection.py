"""
Mutual information between neuron projections and mean input.
"""

import torch
import numpy as np
import logging
from typing import Optional
from ...core.base import BaseMetric

logger = logging.getLogger(__name__)


class MIProjectionVsMeanInput(BaseMetric):
    """
    Compute mutual information between each neuron's projection and the mean input.
    
    For each neuron with weights w, computes MI(w^T x, mean(x)) where x is the input.
    This measures how much information the neuron's projection preserves about the
    average input pattern.
    """
    
    name = "mi_projection_vs_mean_input"
    requires_weights = True
    requires_inputs = True
    requires_outputs = False
    
    def __init__(self, bins: int = 30, eps: float = 1e-9, force_cpu: bool = False):
        """
        Initialize the MI projection metric.
        
        Args:
            bins: Number of bins for discretization
            eps: Small value for numerical stability
            force_cpu: Whether to force CPU computation for large operations
        """
        self.bins = bins
        self.eps = eps
        self.force_cpu = force_cpu
    
    @torch.no_grad()
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute MI scores for each neuron.
        
        Args:
            inputs: Input activations [batch_size, num_features]
            weights: Weight matrix [num_neurons, num_features]
            outputs: Not used
            **kwargs: Additional arguments
            
        Returns:
            MI scores per neuron [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("MI projection requires both inputs and weights")
        
        # Handle dimensions
        if inputs.ndim != 2:
            if inputs.ndim > 2:
                inputs = inputs.flatten(start_dim=1)
            else:
                logger.warning(f"Inputs have unexpected shape: {inputs.shape}")
                return torch.zeros(1, device=weights.device)
        
        if weights.ndim != 2:
            if weights.ndim > 2:
                weights = weights.reshape(weights.shape[0], -1)
            else:
                logger.warning(f"Weights have unexpected shape: {weights.shape}")
                return torch.zeros(1, device=weights.device)
        
        batch_size, num_features = inputs.shape
        num_neurons = weights.shape[0]
        
        # Check dimension compatibility
        if weights.shape[1] != num_features:
            min_dim = min(weights.shape[1], num_features)
            weights = weights[:, :min_dim]
            inputs = inputs[:, :min_dim]
            logger.warning(f"Dimension mismatch, truncating to {min_dim} features")
        
        # Need enough samples for MI estimation
        if batch_size < 10:
            logger.warning(f"Too few samples for MI estimation: {batch_size}")
            return torch.zeros(num_neurons, device=weights.device)
        
        # Compute mean input across features
        mean_input = inputs.mean(dim=1)  # [batch_size]
        
        # Initialize scores
        mi_scores = torch.zeros(num_neurons, device=weights.device)
        
        # Move to CPU if requested and tensors are large
        if self.force_cpu and inputs.is_cuda and inputs.numel() > 1e6:
            inputs_cpu = inputs.cpu()
            weights_cpu = weights.cpu()
            mean_input_cpu = mean_input.cpu()
        else:
            inputs_cpu = inputs
            weights_cpu = weights
            mean_input_cpu = mean_input
        
        # Convert to numpy for binning
        mean_input_np = mean_input_cpu.numpy()
        
        # Discretize mean input
        mean_min, mean_max = np.min(mean_input_np), np.max(mean_input_np)
        if mean_max - mean_min < self.eps:
            logger.warning("Mean input has no variation")
            return torch.zeros(num_neurons, device=weights.device)
        
        mean_bins = np.linspace(mean_min, mean_max, self.bins + 1)
        mean_digitized = np.digitize(mean_input_np, mean_bins[:-1]) - 1
        mean_digitized = np.clip(mean_digitized, 0, self.bins - 1)
        
        # Compute MI for each neuron
        for i in range(num_neurons):
            try:
                # Compute projection: w^T x
                projection = torch.matmul(inputs_cpu, weights_cpu[i])  # [batch_size]
                projection_np = projection.numpy()
                
                # Discretize projection
                proj_min, proj_max = np.min(projection_np), np.max(projection_np)
                if proj_max - proj_min < self.eps:
                    continue
                
                proj_bins = np.linspace(proj_min, proj_max, self.bins + 1)
                proj_digitized = np.digitize(projection_np, proj_bins[:-1]) - 1
                proj_digitized = np.clip(proj_digitized, 0, self.bins - 1)
                
                # Compute joint histogram
                joint_hist = np.zeros((self.bins, self.bins))
                for j in range(batch_size):
                    joint_hist[proj_digitized[j], mean_digitized[j]] += 1
                
                # Convert to probabilities
                joint_prob = joint_hist / batch_size
                
                # Marginal probabilities
                p_proj = joint_prob.sum(axis=1)
                p_mean = joint_prob.sum(axis=0)
                
                # Compute MI
                mi = 0.0
                for pi in range(self.bins):
                    for mi_idx in range(self.bins):
                        if (joint_prob[pi, mi_idx] > self.eps and 
                            p_proj[pi] > self.eps and 
                            p_mean[mi_idx] > self.eps):
                            mi += joint_prob[pi, mi_idx] * np.log2(
                                joint_prob[pi, mi_idx] / (p_proj[pi] * p_mean[mi_idx])
                            )
                
                mi_scores[i] = mi
                
            except Exception as e:
                logger.debug(f"Error computing MI for neuron {i}: {e}")
                continue
        
        # Convert back to torch tensor on original device
        if mi_scores.device != weights.device:
            mi_scores = mi_scores.to(weights.device)
        
        return torch.nan_to_num(mi_scores, nan=0.0) 