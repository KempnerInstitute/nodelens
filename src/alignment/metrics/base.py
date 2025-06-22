"""Base metrics module."""

from alignment.core.base import BaseMetric
from typing import Dict, Any, Optional
import torch


class MetricComputer:
    """
    Helper class for computing multiple metrics at once.
    
    This class provides a convenient way to compute multiple metrics
    on the same data efficiently.
    """
    
    def __init__(self, metrics: Dict[str, BaseMetric]):
        """
        Initialize with a dictionary of metrics.
        
        Args:
            metrics: Dictionary mapping metric names to metric instances
        """
        self.metrics = metrics
    
    def compute_all(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all metrics on the given data.
        
        Args:
            inputs: Input activations
            weights: Layer weights
            outputs: Output activations
            **kwargs: Additional arguments for metrics
            
        Returns:
            Dictionary mapping metric names to computed scores
        """
        results = {}
        
        for name, metric in self.metrics.items():
            try:
                # Only pass data that the metric requires
                compute_kwargs = {}
                if metric.requires_inputs and inputs is not None:
                    compute_kwargs['inputs'] = inputs
                if metric.requires_weights and weights is not None:
                    compute_kwargs['weights'] = weights
                if metric.requires_outputs and outputs is not None:
                    compute_kwargs['outputs'] = outputs
                
                # Add any additional kwargs
                compute_kwargs.update(kwargs)
                
                # Compute metric
                results[name] = metric.compute(**compute_kwargs)
            except Exception as e:
                print(f"Error computing {name}: {e}")
                # Return zeros as fallback
                if weights is not None:
                    results[name] = torch.zeros(weights.shape[0])
                elif outputs is not None:
                    results[name] = torch.zeros(outputs.shape[1])
                else:
                    results[name] = torch.zeros(1)
        
        return results


class BaseInformationMetric(BaseMetric):
    """Base class for information-theoretic metrics."""
    
    def __init__(self, **kwargs):
        """Initialize information metric."""
        super().__init__(**kwargs)
    
    # Information metrics typically need both inputs and outputs
    requires_outputs = True


__all__ = ['BaseMetric', 'MetricComputer', 'BaseInformationMetric']
