"""
Delta Alignment metric implementation.

This metric measures the alignment of weight changes (W_current - W_init)
with the input covariance structure.
"""

from typing import Optional, Any, Dict
import torch
import logging

from ...core.base import BaseMetric
from ...core.registry import register_metric
from .rayleigh_quotient import RayleighQuotient

logger = logging.getLogger(__name__)


@register_metric("delta_alignment")
class DeltaAlignment(RayleighQuotient):
    """
    Delta Alignment metric.
    
    Computes the Rayleigh Quotient of the weight changes (W_current - W_initial)
    with respect to the input covariance. This measures how the learned weight
    changes align with the input data structure.
    
    This metric requires the model to store initial weights for comparison.
    """
    
    def __init__(
        self,
        relative: bool = True,
        min_samples: int = 2,
        scale_by_norm: bool = False,
        **config: Any
    ):
        """
        Initialize the Delta Alignment metric.
        
        Args:
            relative: Whether to normalize by trace(C) for relative alignment
            min_samples: Minimum samples required for covariance computation
            scale_by_norm: Whether to scale covariance by its Frobenius norm
            **config: Additional configuration parameters
        """
        super().__init__(relative, min_samples, scale_by_norm, **config)
        self._initial_weights: Dict[str, torch.Tensor] = {}
    
    def set_initial_weights(self, layer_name: str, weights: torch.Tensor) -> None:
        """
        Store initial weights for a layer.
        
        Args:
            layer_name: Name of the layer
            weights: Initial weight tensor
        """
        self._initial_weights[layer_name] = weights.clone().detach()
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        initial_weights: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute Delta Alignment values for each neuron.
        
        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Current layer weights [output_features, input_features]
            outputs: Not used for this metric
            layer_name: Name of the layer (for retrieving stored initial weights)
            initial_weights: Initial weights (if provided directly)
            **kwargs: Additional parameters
            
        Returns:
            Delta alignment values for each output neuron [output_features]
        """
        if inputs is None or weights is None:
            raise ValueError("DeltaAlignment requires both inputs and weights")
        
        # Get initial weights
        if initial_weights is None:
            if layer_name is None or layer_name not in self._initial_weights:
                logger.warning(
                    f"DeltaAlignment: No initial weights found for layer '{layer_name}'. "
                    "Using zeros as initial weights."
                )
                initial_weights = torch.zeros_like(weights)
            else:
                initial_weights = self._initial_weights[layer_name]
        
        # Ensure shapes match
        if initial_weights.shape != weights.shape:
            logger.warning(
                f"DeltaAlignment: Shape mismatch between current weights {weights.shape} "
                f"and initial weights {initial_weights.shape}. Reshaping."
            )
            if initial_weights.numel() == weights.numel():
                initial_weights = initial_weights.reshape(weights.shape)
            else:
                # If sizes don't match, use zeros
                logger.warning("DeltaAlignment: Cannot reshape initial weights. Using zeros.")
                initial_weights = torch.zeros_like(weights)
        
        # Compute weight difference
        weight_diff = weights - initial_weights
        
        # Use parent class RQ computation on the weight differences
        return super().compute(inputs=inputs, weights=weight_diff, outputs=None, **kwargs)


@register_metric("normalized_delta_alignment")
class NormalizedDeltaAlignment(DeltaAlignment):
    """
    Normalized Delta Alignment metric.
    
    Normalizes the delta alignment by the magnitude of weight changes,
    providing a scale-invariant measure of alignment.
    """
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        initial_weights: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute normalized delta alignment values.
        
        Returns:
            Normalized delta alignment values [output_features]
        """
        # Get regular delta alignment
        delta_rq = super().compute(
            inputs=inputs,
            weights=weights,
            outputs=outputs,
            layer_name=layer_name,
            initial_weights=initial_weights,
            **kwargs
        )
        
        # Get initial weights for normalization
        if initial_weights is None:
            if layer_name is None or layer_name not in self._initial_weights:
                initial_weights = torch.zeros_like(weights)
            else:
                initial_weights = self._initial_weights[layer_name]
        
        # Compute weight change magnitudes
        weight_diff = weights - initial_weights
        weight_change_norm = torch.norm(weight_diff, dim=1)
        
        # Normalize by weight change magnitude
        eps = 1e-12
        normalized_delta = torch.zeros_like(delta_rq)
        valid_mask = weight_change_norm > eps
        normalized_delta[valid_mask] = delta_rq[valid_mask] / weight_change_norm[valid_mask]
        
        return normalized_delta 