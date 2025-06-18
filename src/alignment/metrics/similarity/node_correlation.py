"""
Node correlation metric for measuring output correlations.
"""

import torch
import logging
from typing import Optional
from ..base import AlignmentMetric

logger = logging.getLogger(__name__)


class NodeCorrelation(AlignmentMetric):
    """
    Compute average correlation between each neuron's output and all other neurons.
    
    This measures how correlated a neuron's activations are with other neurons
    in the same layer, which can indicate redundancy or specialization.
    """
    
    name = "node_correlation"
    requires_weights = False
    requires_inputs = False
    requires_outputs = True
    
    def __init__(self, absolute: bool = True, force_cpu: bool = False):
        """
        Initialize the node correlation metric.
        
        Args:
            absolute: Whether to use absolute correlation values
            force_cpu: Whether to force CPU computation for large operations
        """
        self.absolute = absolute
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
        Compute node correlation scores.
        
        Args:
            inputs: Not used
            weights: Not used
            outputs: Output activations [batch_size, num_neurons]
            **kwargs: Additional arguments
            
        Returns:
            Average correlation per neuron [num_neurons]
        """
        if outputs is None:
            raise ValueError("Node correlation requires outputs")
        
        # Handle different output dimensions
        if outputs.ndim != 2:
            if outputs.ndim > 2:
                # Flatten spatial dimensions
                outputs = outputs.flatten(start_dim=1)
            else:
                logger.warning(f"Outputs have unexpected shape: {outputs.shape}")
                return torch.zeros(1, device=outputs.device)
        
        batch_size, num_neurons = outputs.shape
        
        # Need at least 2 samples to compute correlation
        if batch_size < 2:
            logger.warning(f"Need at least 2 samples for correlation, got {batch_size}")
            return torch.zeros(num_neurons, device=outputs.device)
        
        # Single neuron case
        if num_neurons == 1:
            return torch.zeros(1, device=outputs.device)
        
        correlation_scores = torch.zeros(num_neurons, device=outputs.device)
        
        try:
            # Compute correlation matrix
            corr_matrix = self._compute_correlation(outputs)
            
            # Use absolute values if requested
            if self.absolute:
                corr_matrix = torch.abs(corr_matrix)
            
            # For each neuron, compute average correlation with others
            for i in range(num_neurons):
                # Exclude self-correlation
                mask = torch.ones(num_neurons, dtype=torch.bool, device=outputs.device)
                mask[i] = False
                
                if mask.sum() > 0:
                    correlation_scores[i] = corr_matrix[i, mask].mean()
        
        except Exception as e:
            logger.error(f"Error computing node correlation: {e}")
            return torch.zeros(num_neurons, device=outputs.device)
        
        return torch.nan_to_num(correlation_scores, nan=0.0)
    
    def _compute_correlation(self, X: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix."""
        device = X.device
        
        # Move to CPU if requested and tensor is large
        if self.force_cpu and X.is_cuda and X.numel() > 1e6:
            X = X.cpu()
        
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov = torch.matmul(X_centered.T, X_centered) / (X.size(0) - 1)
        
        # Compute standard deviations
        std = torch.sqrt(torch.diag(cov) + 1e-10)
        
        # Handle zero variance neurons
        valid_mask = std > 1e-10
        
        # Compute correlation
        corr = torch.zeros_like(cov)
        if valid_mask.any():
            # Only compute correlation for neurons with non-zero variance
            valid_indices = torch.where(valid_mask)[0]
            outer_std = torch.outer(std[valid_mask], std[valid_mask])
            corr[valid_indices[:, None], valid_indices] = (
                cov[valid_indices[:, None], valid_indices] / outer_std
            )
        
        # Set diagonal to 1 for valid neurons
        corr.diagonal().copy_(valid_mask.float())
        
        # Move back to original device
        if corr.device != device:
            corr = corr.to(device)
        
        return corr 