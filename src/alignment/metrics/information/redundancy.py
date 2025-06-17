"""
Redundancy metrics for neural network alignment analysis.

These metrics measure the redundancy between neurons or features,
indicating how much overlapping information they capture.
"""

from typing import Optional, Any
import torch
import logging

from alignment_refactor.core.base import BaseMetric
from alignment_refactor.core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("average_redundancy", aliases=["redundancy_gaussian"])
class AverageRedundancy(BaseMetric):
    """
    Average redundancy between neurons using Gaussian approximation.
    
    For each neuron, computes the average mutual information with all
    other neurons, indicating how much its information is redundant
    with the rest of the layer.
    """
    
    def __init__(
        self,
        min_samples: int = 2,
        use_correlation: bool = True,
        **config: Any
    ):
        """
        Initialize the redundancy metric.
        
        Args:
            min_samples: Minimum samples for computation
            use_correlation: If True, use correlation; if False, use covariance
            **config: Additional configuration
        """
        super().__init__(**config)
        self.min_samples = min_samples
        self.use_correlation = use_correlation
    
    @property
    def requires_inputs(self) -> bool:
        return True
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return False
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute average redundancy for each neuron.
        
        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [num_neurons, input_features]
            outputs: Not used (computed from inputs and weights)
            
        Returns:
            Average redundancy scores [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("AverageRedundancy requires inputs and weights")
        
        # Flatten if needed
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        batch_size = inputs.shape[0]
        num_neurons = weights.shape[0]
        
        if batch_size < self.min_samples:
            logger.warning(f"Redundancy: Only {batch_size} samples, returning zeros")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)
        
        if num_neurons <= 1:
            logger.warning("Redundancy: Need at least 2 neurons")
            return torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)
        
        # Compute projected outputs
        projected = torch.matmul(inputs, weights.T)  # [batch_size, num_neurons]
        
        # Move to CPU for large correlation computations if needed
        compute_device = projected.device
        if self._should_use_cpu(projected):
            projected = projected.cpu()
        
        # Compute correlation or covariance matrix
        if self.use_correlation:
            # Normalize each neuron's output
            proj_mean = projected.mean(dim=0, keepdim=True)
            proj_std = projected.std(dim=0, keepdim=True)
            proj_std = torch.where(proj_std > 1e-12, proj_std, torch.ones_like(proj_std))
            projected_norm = (projected - proj_mean) / proj_std
            corr_matrix = torch.matmul(projected_norm.T, projected_norm) / (batch_size - 1)
        else:
            # Use covariance
            corr_matrix = torch.cov(projected.T)
        
        # Compute average redundancy for each neuron
        redundancy_scores = torch.zeros(num_neurons, device=weights.device)
        
        for i in range(num_neurons):
            # Average MI with other neurons using Gaussian approximation
            sum_redundancy = 0.0
            num_pairs = 0
            
            for j in range(num_neurons):
                if i == j:
                    continue
                
                # Get correlation/covariance value
                if self.use_correlation:
                    rho_sq = corr_matrix[i, j] ** 2
                else:
                    # Normalize by variances for MI computation
                    var_i = corr_matrix[i, i]
                    var_j = corr_matrix[j, j]
                    if var_i > 1e-12 and var_j > 1e-12:
                        rho_sq = (corr_matrix[i, j] ** 2) / (var_i * var_j)
                    else:
                        rho_sq = 0.0
                
                # MI approximation
                rho_sq = torch.clamp(rho_sq, 0, 0.999999)
                mi_ij = -0.5 * torch.log(1.0 - rho_sq)
                
                sum_redundancy += mi_ij
                num_pairs += 1
            
            if num_pairs > 0:
                redundancy_scores[i] = sum_redundancy / num_pairs
        
        # Move back to original device if needed
        if compute_device != weights.device:
            redundancy_scores = redundancy_scores.to(weights.device)
        
        return torch.nan_to_num(redundancy_scores)


@register_metric("node_redundancy", aliases=["input_redundancy"])
class NodeRedundancy(BaseMetric):
    """
    Redundancy between input features based on correlation.
    
    This metric measures how redundant each input feature is with
    respect to other input features, useful for identifying
    correlated inputs.
    """
    
    def __init__(
        self,
        min_samples: int = 2,
        exclude_self: bool = True,
        **config: Any
    ):
        """
        Initialize node redundancy metric.
        
        Args:
            min_samples: Minimum samples for correlation
            exclude_self: Whether to exclude self-correlation
            **config: Additional configuration
        """
        super().__init__(**config)
        self.min_samples = min_samples
        self.exclude_self = exclude_self
    
    @property
    def requires_inputs(self) -> bool:
        return True
    
    @property
    def requires_weights(self) -> bool:
        return False
    
    @property
    def requires_outputs(self) -> bool:
        return False
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute redundancy for each input feature.
        
        Args:
            inputs: Input activations [batch_size, num_features]
            weights: Not used
            outputs: Not used
            
        Returns:
            Redundancy scores for each input feature [num_features]
        """
        if inputs is None:
            raise ValueError("NodeRedundancy requires inputs")
        
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        
        batch_size, num_features = inputs.shape
        
        if batch_size < self.min_samples:
            logger.warning(f"NodeRedundancy: Only {batch_size} samples")
            return torch.zeros(num_features, device=inputs.device, dtype=inputs.dtype)
        
        # Compute correlation matrix
        compute_device = inputs.device
        if self._should_use_cpu(inputs):
            inputs = inputs.cpu()
        
        # Standardize features
        inputs_mean = inputs.mean(dim=0, keepdim=True)
        inputs_std = inputs.std(dim=0, keepdim=True)
        inputs_std = torch.where(inputs_std > 1e-10, inputs_std, torch.ones_like(inputs_std))
        inputs_norm = (inputs - inputs_mean) / inputs_std
        
        # Correlation matrix
        corr_matrix = torch.matmul(inputs_norm.T, inputs_norm) / (batch_size - 1)
        
        # Take absolute correlations (strength matters, not direction)
        abs_corr = torch.abs(corr_matrix)
        
        # Compute average correlation with other features
        redundancy_scores = torch.zeros(num_features, device=inputs.device)
        
        for i in range(num_features):
            if self.exclude_self:
                # Average correlation with other features
                mask = torch.ones(num_features, dtype=torch.bool, device=corr_matrix.device)
                mask[i] = False
                if mask.sum() > 0:
                    redundancy_scores[i] = abs_corr[i, mask].mean()
            else:
                # Include self-correlation
                redundancy_scores[i] = abs_corr[i].mean()
        
        # Move back to original device
        if compute_device != inputs.device:
            redundancy_scores = redundancy_scores.to(compute_device)
        
        return torch.nan_to_num(redundancy_scores)


@register_metric("layer_redundancy")
class LayerRedundancy(BaseMetric):
    """
    Overall redundancy measure for an entire layer.
    
    Computes a single redundancy score for the layer based on
    the average pairwise mutual information between neurons.
    """
    
    def __init__(
        self,
        return_matrix: bool = False,
        **config: Any
    ):
        """
        Initialize layer redundancy metric.
        
        Args:
            return_matrix: If True, return full redundancy matrix
            **config: Additional configuration
        """
        super().__init__(**config)
        self.return_matrix = return_matrix
        self._avg_redundancy = AverageRedundancy(**config)
    
    @property
    def requires_inputs(self) -> bool:
        return True
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return False
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute overall layer redundancy.
        
        Returns:
            If return_matrix=False: scalar redundancy score
            If return_matrix=True: redundancy matrix [num_neurons, num_neurons]
        """
        # Get per-neuron redundancy scores
        neuron_redundancies = self._avg_redundancy.compute(
            inputs=inputs,
            weights=weights,
            outputs=outputs,
            **kwargs
        )
        
        if self.return_matrix:
            # Return full redundancy matrix (would need to modify avg_redundancy)
            logger.warning("Full matrix return not yet implemented, returning average")
        
        # Return average redundancy across all neurons
        return neuron_redundancies.mean().unsqueeze(0) 