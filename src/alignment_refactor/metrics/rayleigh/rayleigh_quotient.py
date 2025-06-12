"""
Rayleigh Quotient alignment metric implementation.

This metric measures how well neural network weights align with the 
principal components of their input activations.
"""

from typing import Optional, Any
import torch
import logging

from alignment_refactor.core.base import BaseMetric
from alignment_refactor.core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("rayleigh_quotient", aliases=["rq", "RQ"])
class RayleighQuotient(BaseMetric):
    """
    Rayleigh Quotient alignment metric.
    
    Computes the proportion of variance in input activations that is
    captured by each neuron's weight vector. This measures how well
    the weights align with the covariance structure of the inputs.
    
    For weight vector w and input covariance C:
        RQ(w) = (w^T C w) / (w^T w)
    
    When relative=True (default), normalizes by trace(C) to get a
    proportion of total variance.
    """
    
    def __init__(
        self,
        relative: bool = True,
        min_samples: int = 2,
        scale_by_norm: bool = False,
        **config: Any
    ):
        """
        Initialize the Rayleigh Quotient metric.
        
        Args:
            relative: Whether to normalize by trace(C) for relative alignment
            min_samples: Minimum samples required for covariance computation
            scale_by_norm: Whether to scale covariance by its Frobenius norm
            **config: Additional configuration parameters
        """
        super().__init__(**config)
        self.relative = relative
        self.min_samples = min_samples
        self.scale_by_norm = scale_by_norm
    
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
        Compute Rayleigh Quotient values for each neuron.
        
        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            outputs: Not used for this metric
            **kwargs: Additional parameters
            
        Returns:
            RQ values for each output neuron [output_features]
        """
        if inputs is None or weights is None:
            raise ValueError("RayleighQuotient requires both inputs and weights")
        
        # Validate shapes
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        batch_size, input_features = inputs.shape
        output_features, weight_features = weights.shape
        
        # Check sample size
        if batch_size < self.min_samples:
            logger.warning(
                f"RQ: Only {batch_size} samples, minimum {self.min_samples} recommended. "
                "Returning zeros."
            )
            return torch.zeros(output_features, device=weights.device, dtype=weights.dtype)
        
        # Check dimension compatibility
        if input_features != weight_features:
            logger.warning(
                f"RQ: Dimension mismatch - inputs: {input_features}, weights: {weight_features}. "
                "Truncating to common dimensions."
            )
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
        
        # Move to appropriate device for computation
        compute_device = weights.device
        if self._should_use_cpu(inputs, weights):
            logger.debug("RQ: Moving computation to CPU for large tensors")
            compute_device = torch.device('cpu')
            inputs = inputs.cpu()
            weights = weights.cpu()
        
        # Compute covariance matrix
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
        
        # Scale covariance by norm if requested
        if self.scale_by_norm:
            cov_norm = torch.norm(cov, p='fro')
            if cov_norm > 0:
                cov = cov / cov_norm
        
        # Compute RQ for each neuron efficiently
        # w^T C w = sum((w @ C) * w, dim=1)
        wc = torch.matmul(weights, cov)  # [output_features, input_features]
        numerator = torch.sum(wc * weights, dim=1)  # [output_features]
        
        # w^T w
        denominator = torch.sum(weights * weights, dim=1)  # [output_features]
        
        # Compute RQ with numerical stability
        eps = 1e-12
        rq_values = torch.zeros_like(numerator)
        valid_mask = denominator > eps
        rq_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        # Normalize by trace if relative
        if self.relative:
            trace_cov = torch.trace(cov)
            if trace_cov > eps:
                rq_values = rq_values / trace_cov
            else:
                logger.warning("RQ: Covariance trace near zero, cannot compute relative RQ")
        
        # Move back to original device
        if compute_device != weights.device:
            rq_values = rq_values.to(weights.device)
        
        # Handle any numerical issues
        rq_values = torch.nan_to_num(rq_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        return rq_values


@register_metric("rq_patchwise")
class PatchWiseRayleighQuotient(RayleighQuotient):
    """
    Patch-wise variant of Rayleigh Quotient for convolutional layers.
    
    This variant computes RQ separately for each spatial location (patch)
    and then aggregates the results, weighted by patch variance.
    """
    
    def __init__(
        self,
        relative: bool = True,
        min_samples: int = 2,
        scale_by_norm: bool = False,
        weight_by_variance: bool = True,
        **config: Any
    ):
        """
        Initialize patch-wise RQ metric.
        
        Args:
            relative: Whether to normalize by trace(C)
            min_samples: Minimum samples for covariance
            scale_by_norm: Whether to scale covariance by norm
            weight_by_variance: Whether to weight patches by their variance
            **config: Additional configuration
        """
        super().__init__(relative, min_samples, scale_by_norm, **config)
        self.weight_by_variance = weight_by_variance
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute patch-wise RQ for convolutional inputs.
        
        Args:
            inputs: Input patches [batch_size, features, num_patches]
            weights: Convolutional weights [output_channels, input_features]
            outputs: Not used
            
        Returns:
            RQ values for each output channel [output_channels]
        """
        if inputs is None or weights is None:
            raise ValueError("PatchWiseRQ requires both inputs and weights")
        
        # Handle different input formats
        if inputs.ndim == 2:
            # Regular RQ if not patch format
            return super().compute(inputs, weights, outputs, **kwargs)
        
        if inputs.ndim != 3:
            raise ValueError(f"PatchWiseRQ expects 3D input, got {inputs.ndim}D")
        
        batch_size, features, num_patches = inputs.shape
        output_channels = weights.shape[0]
        
        if batch_size < self.min_samples:
            return torch.zeros(output_channels, device=weights.device, dtype=weights.dtype)
        
        # Compute variance for each patch
        patch_var = torch.var(inputs, dim=0, keepdim=False)  # [features, num_patches]
        patch_total_var = patch_var.sum(dim=0)  # [num_patches]
        
        # Initialize results
        all_patch_rq = []
        all_patch_weights = []
        
        # Compute RQ for each patch
        for p in range(num_patches):
            patch_data = inputs[:, :, p]  # [batch_size, features]
            
            # Compute RQ for this patch
            patch_rq = super().compute(patch_data, weights, None)
            
            # Weight by patch variance if requested
            if self.weight_by_variance:
                patch_weight = patch_total_var[p]
            else:
                patch_weight = 1.0
            
            all_patch_rq.append(patch_rq * patch_weight)
            all_patch_weights.append(patch_weight)
        
        # Aggregate across patches
        total_weight = sum(all_patch_weights)
        if total_weight > 0:
            final_rq = torch.stack(all_patch_rq).sum(dim=0) / total_weight
        else:
            final_rq = torch.zeros(output_channels, device=weights.device, dtype=weights.dtype)
        
        return final_rq 