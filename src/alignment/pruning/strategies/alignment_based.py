"""
Alignment-based pruning strategies.

This module implements pruning based on alignment metrics like Rayleigh quotient,
allowing pruning decisions to be guided by neuron-input alignment measures.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import logging

from ..base import BasePruningStrategy
from ...metrics import get_metric

logger = logging.getLogger(__name__)


class AlignmentPruning(BasePruningStrategy):
    """
    Alignment-based pruning strategy.
    
    This strategy prunes weights based on alignment metrics between neurons
    and their inputs, such as Rayleigh quotient or mutual information.
    
    Examples:
        >>> from alignment.pruning.strategies import AlignmentPruning
        >>> from alignment.pruning import PruningConfig
        >>>
        >>> # Prune based on Rayleigh quotient
        >>> config = PruningConfig(amount=0.5, pruning_mode='low')
        >>> strategy = AlignmentPruning(metric='rayleigh_quotient', config=config)
        >>> 
        >>> # Need to provide inputs for alignment computation
        >>> inputs = torch.randn(batch_size, input_dim)
        >>> mask = strategy.prune(layer, inputs=inputs)
    """
    
    def __init__(
        self,
        metric: str = 'rayleigh_quotient',
        config=None,
        aggregate: Literal['mean', 'max', 'sum'] = 'mean',
        **metric_kwargs
    ):
        """
        Initialize alignment-based pruning strategy.
        
        Args:
            metric: Name of alignment metric to use
                Options: 'rayleigh_quotient', 'mutual_information', 'cka', etc.
            config: Pruning configuration
            aggregate: How to aggregate neuron-wise scores to weight-wise scores
                - 'mean': Average score across output neurons (default)
                - 'max': Maximum score across output neurons
                - 'sum': Sum of scores across output neurons
            **metric_kwargs: Additional arguments for the metric
        """
        super().__init__(config)
        self.metric_name = metric
        self.aggregate = aggregate
        self.metric_kwargs = metric_kwargs
        
        # Initialize the metric
        try:
            self.metric = get_metric(metric, **metric_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize metric {metric}: {e}")
            raise
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores based on alignment metrics.
        
        Args:
            module: Module to compute scores for
            inputs: Input activations to the module (required)
            **kwargs: Additional arguments
            
        Returns:
            Tensor of importance scores with shape matching module weights
            
        Raises:
            ValueError: If inputs are not provided or module has no weights
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        if inputs is None:
            raise ValueError(
                "AlignmentPruning requires inputs to compute alignment. "
                "Pass inputs to the prune() method."
            )
        
        weights = module.weight.data
        
        # Compute alignment scores (neuron-wise)
        # Shape: [num_output_neurons]
        alignment_scores = self.metric.compute(inputs=inputs, weights=weights)
        
        # Ensure alignment_scores is on the same device as weights
        if alignment_scores.device != weights.device:
            alignment_scores = alignment_scores.to(weights.device)
        
        # Now we need to convert neuron-wise scores to weight-wise scores
        # weights shape: [out_features, in_features] for Linear
        #                [out_channels, in_channels, ...] for Conv
        
        if len(weights.shape) == 2:  # Linear layer
            # Expand neuron scores to weight scores
            # Each row (output neuron) gets the same score for all its weights
            importance = alignment_scores.unsqueeze(1).expand_as(weights)
            
        elif len(weights.shape) >= 3:  # Conv layer
            # For conv layers, alignment_scores has one value per output channel
            # Expand to all weights in that channel
            out_channels = weights.shape[0]
            importance = alignment_scores.view(out_channels, 1, 1, 1)
            # Expand to match weight dimensions
            importance = importance.expand_as(weights)
            
        else:
            raise ValueError(f"Unsupported weight shape: {weights.shape}")
        
        # Option to aggregate across output dimension for true weight-wise importance
        if self.aggregate == 'none':
            return importance
        
        # For more fine-grained pruning, we might want to consider
        # the importance of individual weights rather than whole neurons
        # This would require a different approach or metric
        
        return importance


class HybridPruning(BasePruningStrategy):
    """
    Hybrid pruning strategy combining magnitude and alignment information.
    
    This strategy combines traditional magnitude-based importance with
    alignment metrics for more informed pruning decisions.
    
    Examples:
        >>> from alignment.pruning.strategies import HybridPruning
        >>>
        >>> # Combine magnitude and Rayleigh quotient
        >>> strategy = HybridPruning(
        ...     alignment_metric='rayleigh_quotient',
        ...     alpha=0.5  # Equal weighting
        ... )
        >>> mask = strategy.prune(layer, inputs=inputs, amount=0.5)
    """
    
    def __init__(
        self,
        alignment_metric: str = 'rayleigh_quotient',
        alpha: float = 0.5,
        config=None,
        **metric_kwargs
    ):
        """
        Initialize hybrid pruning strategy.
        
        Args:
            alignment_metric: Name of alignment metric to use
            alpha: Weight for alignment score (1-alpha for magnitude)
                0 = pure magnitude, 1 = pure alignment
            config: Pruning configuration
            **metric_kwargs: Additional arguments for the metric
        """
        super().__init__(config)
        self.alignment_metric_name = alignment_metric
        self.alpha = alpha
        self.metric_kwargs = metric_kwargs
        
        # Initialize the alignment metric
        try:
            from ...metrics import get_metric
            self.alignment_metric = get_metric(alignment_metric, **metric_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize metric {alignment_metric}: {e}")
            raise
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores combining magnitude and alignment.
        
        Args:
            module: Module to compute scores for
            inputs: Input activations (required for alignment)
            **kwargs: Additional arguments
            
        Returns:
            Combined importance scores
        """
        if not hasattr(module, 'weight'):
            raise ValueError(f"Module {module} does not have weights")
        
        weights = module.weight.data
        
        # Magnitude-based importance
        magnitude_importance = weights.abs()
        
        # Normalize magnitude scores
        mag_min = magnitude_importance.min()
        mag_max = magnitude_importance.max()
        if mag_max > mag_min:
            magnitude_importance = (magnitude_importance - mag_min) / (mag_max - mag_min)
        
        if inputs is None or self.alpha == 0:
            # No inputs provided or pure magnitude
            return magnitude_importance
        
        # Alignment-based importance
        alignment_scores = self.alignment_metric.compute(inputs=inputs, weights=weights)
        
        # Ensure alignment_scores is on the same device
        if alignment_scores.device != weights.device:
            alignment_scores = alignment_scores.to(weights.device)
        
        # Expand alignment scores to match weight dimensions
        if len(weights.shape) == 2:  # Linear
            alignment_importance = alignment_scores.unsqueeze(1).expand_as(weights)
        elif len(weights.shape) >= 3:  # Conv
            out_channels = weights.shape[0]
            alignment_importance = alignment_scores.view(out_channels, 1, 1, 1)
            alignment_importance = alignment_importance.expand_as(weights)
        else:
            raise ValueError(f"Unsupported weight shape: {weights.shape}")
        
        # Normalize alignment scores
        align_min = alignment_importance.min()
        align_max = alignment_importance.max()
        if align_max > align_min:
            alignment_importance = (alignment_importance - align_min) / (align_max - align_min)
        
        # Combine scores
        combined_importance = (
            self.alpha * alignment_importance + 
            (1 - self.alpha) * magnitude_importance
        )
        
        return combined_importance 