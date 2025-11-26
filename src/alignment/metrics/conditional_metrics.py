"""
Conditional metrics for class-conditioned analysis.

This module provides metrics that compute importance scores conditioned on class labels:
- Conditional Rayleigh Quotient (class-conditioned covariance)
- Conditional Mutual Information (MI between activations and targets given class)
- Conditional Activation Norm (class-conditioned activation statistics)
- MI About Class (direct MI between activations and class labels)
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("conditional_rayleigh_quotient", aliases=["conditional_rq", "cond_rq"])
class ConditionalRayleighQuotient(BaseMetric):
    """
    Class-conditioned Rayleigh Quotient.
    
    Computes RQ using class-specific covariance matrices, then averages
    weighted by class frequency. This measures alignment with within-class
    variance rather than total variance.
    
    For each class c:
        RQ_c(w) = (w^T Σ_{X|Y=c} w) / (w^T w)
    
    Final score = Σ_c P(Y=c) * RQ_c(w)
    """
    
    def __init__(
        self,
        relative: bool = True,
        min_samples: int = 2,
        regularization: float = 1e-6,
        return_delta: bool = False,
        **config: Any,
    ):
        """
        Initialize conditional RQ metric.
        
        Args:
            relative: Whether to normalize by trace(Σ_c) for relative alignment
            min_samples: Minimum samples per class for covariance computation
            regularization: Regularization for numerical stability
            return_delta: If True, return ΔRQ = RQ_uncond - RQ_cond (measures class dependency)
        """
        super().__init__(**config)
        self.relative = relative
        self.min_samples = min_samples
        self.regularization = regularization
        self.return_delta = return_delta
    
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
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute class-conditioned RQ scores.
        
        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            targets: Class labels [batch_size]
            
        Returns:
            Conditional RQ scores [output_features]
        """
        if inputs is None or weights is None:
            raise ValueError("Conditional RQ requires both inputs and weights")
        
        # Flatten if needed
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)
        
        batch_size, input_features = inputs.shape
        output_features, weight_features = weights.shape
        
        # Handle dimension mismatch
        if input_features != weight_features:
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_features = min_dim
        
        device = weights.device
        
        # If no targets, fall back to unconditional RQ
        if targets is None:
            logger.warning("Conditional RQ: No targets provided, computing unconditional RQ")
            return self._compute_unconditional(inputs, weights)
        
        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        # Get unique classes
        classes = torch.unique(targets)
        
        # Compute class-conditioned RQ (weighted average)
        rq_cond_sum = torch.zeros(output_features, device=device, dtype=inputs.dtype)
        total_weight = 0.0
        
        for c in classes:
            mask = targets == c
            n_c = mask.sum()
            
            if n_c < self.min_samples:
                continue
            
            # Extract class data
            inputs_c = inputs[mask]
            
            # Compute class-specific covariance
            inputs_c_centered = inputs_c - inputs_c.mean(dim=0, keepdim=True)
            cov_c = (inputs_c_centered.T @ inputs_c_centered) / max(1, n_c - 1)
            
            # Add regularization
            if self.regularization > 0:
                cov_c = cov_c + self.regularization * torch.eye(input_features, device=device, dtype=cov_c.dtype)
            
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
        
        if self.return_delta:
            rq_uncond = self._compute_unconditional(inputs, weights)
            return rq_uncond - rq_cond  # ΔRQ measures class-dependent alignment
        
        return rq_cond
    
    def _compute_unconditional(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute standard unconditional RQ."""
        batch_size = inputs.shape[0]
        
        # Compute covariance
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        cov = (inputs_centered.T @ inputs_centered) / max(1, batch_size - 1)
        
        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        
        # Compute RQ
        wc = torch.matmul(weights, cov)
        numerator = torch.sum(wc * weights, dim=1)
        denominator = torch.sum(weights * weights, dim=1)
        
        eps = 1e-12
        rq = torch.zeros_like(numerator)
        valid_mask = denominator > eps
        rq[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        if self.relative:
            trace = torch.trace(cov)
            if trace > eps:
                rq = rq / trace
        
        return torch.nan_to_num(rq, nan=0.0, posinf=0.0, neginf=0.0)


@register_metric("mi_about_class", aliases=["mi_class", "class_mi"])
class MIAboutClass(BaseMetric):
    """
    Mutual Information between neuron activations and class labels.
    
    I(Z; Y) where Z is the neuron activation and Y is the class label.
    
    This measures how much information each neuron carries about the class.
    High MI means the neuron is informative about class identity.
    
    Uses Gaussian approximation for continuous activations with discrete labels.
    """
    
    def __init__(
        self,
        method: str = "gaussian",
        bins: int = 10,
        min_samples_per_class: int = 5,
        **config: Any,
    ):
        """
        Initialize MI about class metric.
        
        Args:
            method: "gaussian" (default) or "binning"
            bins: Number of bins for binning method
            min_samples_per_class: Minimum samples per class
        """
        super().__init__(**config)
        self.method = method
        self.bins = bins
        self.min_samples_per_class = min_samples_per_class
    
    @property
    def requires_inputs(self) -> bool:
        return False
    
    @property
    def requires_weights(self) -> bool:
        return False
    
    @property
    def requires_outputs(self) -> bool:
        return True
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute MI between activations and class labels.
        
        Args:
            outputs: Neuron activations [batch_size, num_neurons]
            targets: Class labels [batch_size]
            
        Returns:
            MI scores [num_neurons]
        """
        if outputs is None:
            raise ValueError("MI about class requires outputs (activations)")
        
        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        if targets is None:
            logger.warning("MI about class: No targets provided, returning zeros")
            return torch.zeros(num_neurons, device=device)
        
        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        if self.method == "gaussian":
            return self._compute_gaussian(outputs, targets)
        else:
            return self._compute_binning(outputs, targets)
    
    def _compute_gaussian(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MI using Gaussian approximation.
        
        For continuous Z and discrete Y:
        I(Z; Y) = H(Z) - H(Z|Y)
                = 0.5 * log(Var(Z)) - E_y[0.5 * log(Var(Z|Y=y))]
        """
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        classes = torch.unique(targets)
        mi_scores = torch.zeros(num_neurons, device=device, dtype=outputs.dtype)
        
        eps = 1e-12
        
        for i in range(num_neurons):
            z = outputs[:, i]
            
            # Total variance H(Z)
            var_z = z.var()
            if var_z < eps:
                continue
            
            h_z = 0.5 * torch.log(var_z + eps)
            
            # Conditional variance H(Z|Y)
            h_z_given_y = 0.0
            total_weight = 0.0
            
            for c in classes:
                mask = targets == c
                n_c = mask.sum()
                
                if n_c < self.min_samples_per_class:
                    continue
                
                z_c = z[mask]
                var_z_c = z_c.var()
                
                if var_z_c > eps:
                    h_z_given_y += (n_c.float() / batch_size) * 0.5 * torch.log(var_z_c + eps)
                    total_weight += n_c.float() / batch_size
            
            if total_weight > 0:
                h_z_given_y = h_z_given_y / total_weight * total_weight  # Weighted average
                mi_scores[i] = torch.clamp(h_z - h_z_given_y, min=0.0)
        
        return torch.nan_to_num(mi_scores, nan=0.0)
    
    def _compute_binning(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MI using binning method."""
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        # Convert to numpy
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        classes = np.unique(targets_np)
        n_classes = len(classes)
        
        mi_scores = np.zeros(num_neurons)
        
        for i in range(num_neurons):
            z = outputs_np[:, i]
            
            # Discretize z
            z_min, z_max = z.min(), z.max()
            if z_max - z_min < 1e-10:
                continue
            
            z_bins = np.linspace(z_min, z_max, self.bins + 1)
            z_discrete = np.digitize(z, z_bins[:-1]) - 1
            z_discrete = np.clip(z_discrete, 0, self.bins - 1)
            
            # Compute joint distribution P(Z, Y)
            joint = np.zeros((self.bins, n_classes))
            for j, c in enumerate(classes):
                mask = targets_np == c
                for k in range(self.bins):
                    joint[k, j] = np.sum((z_discrete == k) & mask)
            
            joint = joint / batch_size
            
            # Marginals
            p_z = joint.sum(axis=1)
            p_y = joint.sum(axis=0)
            
            # MI
            mi = 0.0
            for k in range(self.bins):
                for j in range(n_classes):
                    if joint[k, j] > 1e-12 and p_z[k] > 1e-12 and p_y[j] > 1e-12:
                        mi += joint[k, j] * np.log(joint[k, j] / (p_z[k] * p_y[j]))
            
            mi_scores[i] = max(0, mi / np.log(2))  # Convert to bits
        
        return torch.tensor(mi_scores, device=device, dtype=outputs.dtype)


@register_metric("conditional_activation_norm", aliases=["cond_act_norm"])
class ConditionalActivationNorm(BaseMetric):
    """
    Class-conditioned activation L2 norm.
    
    Computes the average L2 norm of activations within each class,
    then aggregates across classes. This measures how strongly neurons
    respond to inputs from each class.
    
    Can also compute variance of norms across classes to identify
    class-selective neurons.
    """
    
    def __init__(
        self,
        aggregation: str = "mean",
        normalize: bool = True,
        min_samples_per_class: int = 2,
        **config: Any,
    ):
        """
        Initialize conditional activation norm metric.
        
        Args:
            aggregation: How to aggregate across classes:
                - "mean": Average norm across classes
                - "max": Max norm across classes (most responsive class)
                - "variance": Variance of norms (class selectivity)
                - "ratio": Max/min ratio (class discrimination)
            normalize: Whether to normalize by total activation
            min_samples_per_class: Minimum samples per class
        """
        super().__init__(**config)
        self.aggregation = aggregation
        self.normalize = normalize
        self.min_samples_per_class = min_samples_per_class
    
    @property
    def requires_inputs(self) -> bool:
        return False
    
    @property
    def requires_weights(self) -> bool:
        return False
    
    @property
    def requires_outputs(self) -> bool:
        return True
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute class-conditioned activation norm.
        
        Args:
            outputs: Neuron activations [batch_size, num_neurons]
            targets: Class labels [batch_size]
            
        Returns:
            Activation norm scores [num_neurons]
        """
        if outputs is None:
            raise ValueError("Conditional activation norm requires outputs")
        
        # Handle CNN outputs
        if outputs.ndim == 4:
            # [B, C, H, W] -> [B, C] (average over spatial dims)
            outputs = outputs.mean(dim=(2, 3))
        elif outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        if targets is None:
            # Fall back to unconditional norm
            return torch.norm(outputs, p=2, dim=0) / np.sqrt(batch_size)
        
        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        classes = torch.unique(targets)
        
        # Compute per-class norms for each neuron
        class_norms = []
        
        for c in classes:
            mask = targets == c
            n_c = mask.sum()
            
            if n_c < self.min_samples_per_class:
                continue
            
            outputs_c = outputs[mask]
            
            # L2 norm per neuron within this class
            norm_c = torch.norm(outputs_c, p=2, dim=0) / np.sqrt(n_c.float())
            class_norms.append(norm_c)
        
        if not class_norms:
            return torch.zeros(num_neurons, device=device)
        
        class_norms = torch.stack(class_norms, dim=0)  # [num_classes, num_neurons]
        
        # Aggregate across classes
        if self.aggregation == "mean":
            scores = class_norms.mean(dim=0)
        elif self.aggregation == "max":
            scores = class_norms.max(dim=0)[0]
        elif self.aggregation == "variance":
            scores = class_norms.var(dim=0)
        elif self.aggregation == "ratio":
            max_norm = class_norms.max(dim=0)[0]
            min_norm = class_norms.min(dim=0)[0]
            scores = max_norm / (min_norm + 1e-12)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        if self.normalize:
            total_norm = torch.norm(outputs, p=2, dim=0) / np.sqrt(batch_size)
            scores = scores / (total_norm + 1e-12)
        
        return torch.nan_to_num(scores, nan=0.0)


@register_metric("delta_rq", aliases=["delta_rayleigh_quotient"])
class DeltaRQ(BaseMetric):
    """
    Delta Rayleigh Quotient: RQ_unconditional - RQ_conditional.
    
    Measures how much the alignment changes when conditioning on class.
    High ΔRQ means the neuron aligns differently with different classes,
    indicating class-specific processing.
    """
    
    def __init__(self, relative: bool = True, min_samples: int = 2, **config: Any):
        super().__init__(**config)
        self.relative = relative
        self.min_samples = min_samples
        self._cond_rq = ConditionalRayleighQuotient(
            relative=relative,
            min_samples=min_samples,
            return_delta=True,
        )
    
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
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute ΔRQ = RQ_uncond - RQ_cond."""
        return self._cond_rq.compute(
            inputs=inputs,
            weights=weights,
            outputs=outputs,
            targets=targets,
            **kwargs,
        )


@register_metric("conditional_mi", aliases=["cond_mi"])
class ConditionalMIGaussian(BaseMetric):
    """
    Class-conditioned Mutual Information using Gaussian approximation.
    
    Computes MI between neuron activations and a reference signal,
    conditioned on class labels. This measures information flow
    that is specific to each class.
    """
    
    def __init__(
        self,
        use_pc_reference: bool = True,
        min_samples: int = 5,
        **config: Any,
    ):
        super().__init__(**config)
        self.use_pc_reference = use_pc_reference
        self.min_samples = min_samples
    
    @property
    def requires_inputs(self) -> bool:
        return self.use_pc_reference
    
    @property
    def requires_weights(self) -> bool:
        return False
    
    @property
    def requires_outputs(self) -> bool:
        return True
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute class-conditioned MI.
        
        Args:
            outputs: Neuron activations [batch_size, num_neurons]
            targets: Class labels [batch_size]
            inputs: Input activations (for PC reference)
            target_outputs: Reference signal (optional)
            
        Returns:
            Conditional MI scores [num_neurons]
        """
        if outputs is None:
            raise ValueError("Conditional MI requires outputs")
        
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        if targets is None:
            logger.warning("Conditional MI: No targets, computing unconditional MI")
            return self._compute_unconditional_mi(outputs, inputs, target_outputs)
        
        if targets.ndim > 1:
            targets = targets.squeeze()
        
        classes = torch.unique(targets)
        
        # Compute MI within each class and average
        mi_sum = torch.zeros(num_neurons, device=device, dtype=outputs.dtype)
        total_weight = 0.0
        
        for c in classes:
            mask = targets == c
            n_c = mask.sum()
            
            if n_c < self.min_samples:
                continue
            
            outputs_c = outputs[mask]
            inputs_c = inputs[mask] if inputs is not None else None
            target_c = target_outputs[mask] if target_outputs is not None else None
            
            mi_c = self._compute_unconditional_mi(outputs_c, inputs_c, target_c)
            
            weight_c = n_c.float() / batch_size
            mi_sum += mi_c * weight_c
            total_weight += weight_c
        
        if total_weight > 0:
            return mi_sum / total_weight
        
        return torch.zeros(num_neurons, device=device)
    
    def _compute_unconditional_mi(
        self,
        outputs: torch.Tensor,
        inputs: Optional[torch.Tensor],
        target_outputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute standard MI."""
        batch_size, num_neurons = outputs.shape
        device = outputs.device
        
        # Determine reference
        if target_outputs is not None:
            ref = target_outputs
            if ref.ndim == 1:
                ref = ref.unsqueeze(1)
        elif self.use_pc_reference and inputs is not None:
            if inputs.ndim > 2:
                inputs = inputs.reshape(inputs.shape[0], -1)
            try:
                inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                _, _, V = torch.linalg.svd(inputs_centered, full_matrices=False)
                ref = (inputs_centered @ V[0:1, :].T)
            except Exception:
                ref = outputs[:, :1]
        else:
            ref = outputs[:, :1]
        
        if ref.shape[0] != batch_size:
            return torch.zeros(num_neurons, device=device)
        
        # Compute correlation-based MI
        eps = 1e-12
        out_std = outputs.std(dim=0, keepdim=True)
        out_std = torch.where(out_std > eps, out_std, torch.ones_like(out_std))
        z_out = (outputs - outputs.mean(dim=0, keepdim=True)) / out_std
        
        ref_std = ref.std(dim=0, keepdim=True)
        ref_std = torch.where(ref_std > eps, ref_std, torch.ones_like(ref_std))
        z_ref = (ref - ref.mean(dim=0, keepdim=True)) / ref_std
        
        corr = (z_out.T @ z_ref) / max(1, batch_size - 1)
        rho_sq = torch.clamp(corr.pow(2), 0.0, 0.999999)
        mi = -0.5 * torch.log(1.0 - rho_sq)
        
        return torch.nan_to_num(mi.mean(dim=1))

