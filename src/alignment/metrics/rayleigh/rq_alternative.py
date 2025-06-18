"""
Alternative Rayleigh Quotient metric with different normalization.
"""

import torch
import logging
from typing import Optional
from alignment.core.base import BaseMetric

logger = logging.getLogger(__name__)


class RayleighQuotientAlternative(BaseMetric):
    """
    Compute Rayleigh Quotient with alternative denominator.
    
    Instead of using w^T w in the denominator, this uses trace(C) where C is the
    covariance matrix of inputs. This provides a different normalization that
    can be more stable in some cases.
    """
    
    name = "rayleigh_quotient_alternative"
    requires_weights = True
    requires_inputs = True
    requires_outputs = False
    
    def __init__(self, relative: bool = True, epsilon: float = 1e-8):
        """
        Initialize the alternative RQ metric.
        
        Args:
            relative: Whether to normalize by trace(C)
            epsilon: Small value for numerical stability
        """
        self.relative = relative
        self.epsilon = epsilon
    
    @torch.no_grad()
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute alternative RQ scores for each neuron.
        
        Args:
            inputs: Input activations [batch_size, num_features]
            weights: Weight matrix [num_neurons, num_features]
            outputs: Not used
            **kwargs: Additional arguments
            
        Returns:
            RQ scores per neuron [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("Alternative RQ requires both inputs and weights")
        
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
        
        # Need at least 2 samples for covariance
        if batch_size < 2:
            logger.warning(f"Need at least 2 samples, got {batch_size}")
            return torch.zeros(num_neurons, device=weights.device)
        
        try:
            # Compute input covariance
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            C = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
            
            # Check for numerical issues
            if torch.isnan(C).any() or torch.isinf(C).any():
                logger.warning("NaN or Inf in covariance matrix")
                return torch.zeros(num_neurons, device=weights.device)
            
            # Compute w^T C w for each neuron
            # Efficient computation: WC = W @ C, then sum(WC * W, dim=1)
            WC = torch.matmul(weights, C)
            numerators = torch.sum(WC * weights, dim=1)
            
            # Alternative denominator: trace(C)
            trace_C = torch.trace(C)
            
            # Initialize scores
            rq_scores = torch.zeros(num_neurons, device=weights.device)
            
            if self.relative and trace_C > self.epsilon:
                # Normalize by trace(C)
                rq_scores = numerators / trace_C
            else:
                # Just use raw numerators
                rq_scores = numerators
            
            # Additional normalization by number of features
            # This helps make scores comparable across layers
            rq_scores = rq_scores / num_features
            
        except Exception as e:
            logger.error(f"Error computing alternative RQ: {e}")
            return torch.zeros(num_neurons, device=weights.device)
        
        return torch.nan_to_num(rq_scores, nan=0.0, posinf=0.0, neginf=0.0) 