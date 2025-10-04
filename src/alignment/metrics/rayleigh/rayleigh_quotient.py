"""
Rayleigh Quotient alignment metric implementation.

This metric measures how well neural network weights align with the 
principal components of their input activations.
"""

from typing import Optional, Any, Union, Dict
import torch
import logging

from ...core.base import BaseMetric
from ...core.registry import register_metric

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
        class_conditioned_targets: Optional[torch.Tensor] = None,
        regularization: float = 1e-6,
        **config: Any
    ):
        """
        Initialize the Rayleigh Quotient metric.
        
        Args:
            relative: Whether to normalize by trace(C) for relative alignment
            min_samples: Minimum samples required for covariance computation
            scale_by_norm: Whether to scale covariance by its Frobenius norm
            regularization: Small value added to diagonal for numerical stability (default: 1e-6)
            **config: Additional configuration parameters
        """
        super().__init__(**config)
        self.relative = relative
        self.min_samples = min_samples
        self.scale_by_norm = scale_by_norm
        self.regularization = regularization
        # Optional: class-conditioned covariance support (targets provided at compute time preferred)
        self._cc_targets = class_conditioned_targets
    
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
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient values for each neuron.
        
        Args:
            inputs: Input activations [batch_size, input_features] or [batch_size, features, patches]
            weights: Layer weights [output_features, input_features]
            outputs: Not used for this metric
            **kwargs: Additional parameters
            
        Returns:
            RQ values for each output neuron [output_features]
        """
        if inputs is None or weights is None:
            raise ValueError("RayleighQuotient requires both inputs and weights")
        
        # Handle patchwise inputs (3D tensors)
        if inputs.ndim == 3:
            # Compute patchwise RQ
            return self._compute_patchwise(inputs, weights, **kwargs)
        
        # Validate shapes for standard 2D computation
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
        
        # Optionally compute class-conditioned covariance and average
        use_class_cond = targets is not None or self._cc_targets is not None
        if use_class_cond:
            tgt = targets if targets is not None else self._cc_targets
            if tgt.ndim > 1:
                tgt = tgt.squeeze()
            classes = torch.unique(tgt)
            cov = torch.zeros(input_features, input_features, device=inputs.device, dtype=inputs.dtype)
            total_weight = 0.0
            for c in classes:
                mask = (tgt == c)
                if mask.sum() < self.min_samples:
                    continue
                Xc = inputs[mask]
                Xc_centered = Xc - Xc.mean(dim=0, keepdim=True)
                cov_c = (Xc_centered.T @ Xc_centered) / max(1, (Xc.shape[0] - 1))
                weight_c = float(mask.sum())
                cov += cov_c * weight_c
                total_weight += weight_c
            if total_weight > 0:
                cov = cov / total_weight
            else:
                inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
        else:
            # Compute covariance matrix
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
        
        # Add regularization to diagonal for numerical stability
        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(
                input_features, device=cov.device, dtype=cov.dtype
            )
        
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
    
    def compute_class_conditioned(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        targets: torch.Tensor,
        return_delta_rq: bool = False,
        **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute class-conditioned Rayleigh Quotient.
        
        For each class c, computes RQ using class-specific covariance Σ_{X|y=c},
        then returns the average across classes weighted by class frequency.
        
        Optionally also computes ΔRQ = RQ(unconditional) - E[RQ(class-conditioned)],
        which measures how much alignment varies across classes.
        
        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            targets: Class labels [batch_size]
            return_delta_rq: If True, also return ΔRQ
            **kwargs: Additional parameters
            
        Returns:
            If return_delta_rq=False: class-conditioned RQ [output_features]
            If return_delta_rq=True: dict with keys 'rq_uncond', 'rq_cond', 'delta_rq'
        """
        # Flatten if needed
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        batch_size, input_features = inputs.shape
        output_features, weight_features = weights.shape
        
        # Check compatibility
        if input_features != weight_features:
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_features = min_dim
        
        device = weights.device
        
        # Get unique classes
        classes = torch.unique(targets)
        
        # Compute class-conditioned RQ (weighted average)
        rq_cond_sum = torch.zeros(output_features, device=device)
        total_weight = 0.0
        
        for c in classes:
            mask = (targets == c)
            n_c = mask.sum()
            
            if n_c < self.min_samples:
                logger.warning(f"Class {c}: only {n_c} samples, skipping")
                continue
            
            # Extract class data
            inputs_c = inputs[mask]
            
            # Compute class-specific covariance
            inputs_c_centered = inputs_c - inputs_c.mean(dim=0, keepdim=True)
            cov_c = (inputs_c_centered.T @ inputs_c_centered) / max(1, n_c - 1)
            
            # Add regularization
            if self.regularization > 0:
                cov_c = cov_c + self.regularization * torch.eye(
                    input_features, device=device, dtype=cov_c.dtype
                )
            
            # Compute RQ for this class
            wc = torch.matmul(weights, cov_c)
            numerator_c = torch.sum(wc * weights, dim=1)
            denominator_c = torch.sum(weights * weights, dim=1)
            
            eps = 1e-12
            rq_c = torch.zeros_like(numerator_c)
            valid_mask = denominator_c > eps
            rq_c[valid_mask] = numerator_c[valid_mask] / denominator_c[valid_mask]
            
            # Normalize by trace if relative
            if self.relative:
                trace_c = torch.trace(cov_c)
                if trace_c > eps:
                    rq_c = rq_c / trace_c
            
            # Weighted sum
            weight_c = n_c.float()
            rq_cond_sum += rq_c * weight_c
            total_weight += weight_c
        
        # Average across classes
        if total_weight > 0:
            rq_cond = rq_cond_sum / total_weight
        else:
            logger.warning("No valid classes found, returning zeros")
            rq_cond = torch.zeros(output_features, device=device)
        
        # Clean up numerical issues
        rq_cond = torch.nan_to_num(rq_cond, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not return_delta_rq:
            return rq_cond
        
        # Also compute unconditional RQ
        rq_uncond = self.compute(inputs=inputs, weights=weights, **kwargs)
        
        # Compute ΔRQ
        delta_rq = rq_uncond - rq_cond
        
        return {
            'rq_uncond': rq_uncond,
            'rq_cond': rq_cond,
            'delta_rq': delta_rq
        }
    
    def _compute_patchwise(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        weight_by_variance: bool = True,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute patch-wise RQ for CNN layers.
        
        Args:
            inputs: Input patches [batch_size, features, num_patches]
            weights: Flattened weights [output_features, features]
            weight_by_variance: Whether to weight patches by their variance
            
        Returns:
            RQ values [output_features]
        """
        batch_size, features, num_patches = inputs.shape
        output_features = weights.shape[0]
        
        if batch_size < self.min_samples:
            logger.warning(f"Only {batch_size} samples, minimum {self.min_samples} recommended")
            return torch.zeros(output_features, device=weights.device, dtype=weights.dtype)
        
        # Ensure weight dimensions match
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        # Compute variance for each patch
        patch_var = torch.var(inputs, dim=0, keepdim=False)  # [features, num_patches]
        patch_total_var = patch_var.sum(dim=0)  # [num_patches]
        
        # Initialize accumulators
        weighted_rq_sum = torch.zeros(output_features, device=weights.device)
        total_weight = 0.0
        
        # Compute RQ for each patch
        for p in range(num_patches):
            patch_data = inputs[:, :, p]  # [batch_size, features]
            
            # Center the data
            patch_data_centered = patch_data - patch_data.mean(dim=0, keepdim=True)
            
            # Compute covariance for this patch
            patch_cov = torch.matmul(patch_data_centered.T, patch_data_centered) / (batch_size - 1)
            
            # Scale by norm if requested
            if self.scale_by_norm:
                cov_norm = torch.norm(patch_cov, p='fro')
                if cov_norm > 0:
                    patch_cov = patch_cov / cov_norm
            
            # Handle dimension mismatch
            min_dim = min(features, weights.shape[1])
            if features != weights.shape[1]:
                patch_cov = patch_cov[:min_dim, :min_dim]
                weights_adj = weights[:, :min_dim]
            else:
                weights_adj = weights
            
            # Compute RQ for this patch
            wc = torch.matmul(weights_adj, patch_cov)
            numerator = torch.sum(wc * weights_adj, dim=1)
            denominator = torch.sum(weights_adj * weights_adj, dim=1)
            
            eps = 1e-12
            patch_rq = torch.zeros_like(numerator)
            valid_mask = denominator > eps
            patch_rq[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
            
            # Normalize by trace if relative
            if self.relative:
                trace = torch.trace(patch_cov)
                if trace > eps:
                    patch_rq = patch_rq / trace
            
            # Weight by patch variance if requested
            if weight_by_variance:
                patch_weight = patch_total_var[p].item()
            else:
                patch_weight = 1.0
            
            weighted_rq_sum += patch_rq * patch_weight
            total_weight += patch_weight
        
        # Average across patches
        if total_weight > 0:
            final_rq = weighted_rq_sum / total_weight
        else:
            final_rq = weighted_rq_sum
        
        return torch.nan_to_num(final_rq, nan=0.0, posinf=0.0, neginf=0.0)


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