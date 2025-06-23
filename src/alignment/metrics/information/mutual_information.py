"""
Mutual Information metrics for measuring dependencies between neural representations.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np
import logging
import warnings
from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("mutual_information_gaussian", aliases=["mi_gaussian", "mi_0"])
class MutualInformationGaussian(BaseMetric):
    """
    Mutual Information metric assuming Gaussian distributions.
    
    Computes MI between layer outputs and a reference signal (target outputs
    or first principal component of inputs) using the Gaussian approximation:
    MI = -0.5 * log(1 - ρ²)
    
    where ρ is the correlation coefficient.
    """
    
    def __init__(
        self,
        use_pc_reference: bool = True,
        min_samples: int = 2,
        **config: Any
    ):
        """
        Initialize the Gaussian MI metric.
        
        Args:
            use_pc_reference: If True and no target provided, use PC1 of inputs as reference
            min_samples: Minimum samples required for computation
            **config: Additional configuration
        """
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
        target_outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute Gaussian MI for each output neuron.
        
        Args:
            inputs: Input activations (used for PC reference if needed)
            weights: Not used
            outputs: Layer output activations [batch_size, num_neurons]
            target_outputs: Target reference signal [batch_size, num_targets]
            
        Returns:
            MI values for each neuron [num_neurons]
        """
        if outputs is None:
            raise ValueError("MutualInformationGaussian requires outputs")
        
        if outputs.ndim != 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        batch_size, num_neurons = outputs.shape
        
        if batch_size < self.min_samples:
            logger.warning(f"MI_gaussian: Only {batch_size} samples, returning zeros")
            return torch.zeros(num_neurons, device=outputs.device, dtype=outputs.dtype)
        
        # Determine reference signal
        if target_outputs is None:
            if self.use_pc_reference and inputs is not None:
                # Use first PC of inputs as reference
                if inputs.ndim != 2:
                    inputs = inputs.reshape(inputs.shape[0], -1)
                
                try:
                    # Compute covariance and get first PC
                    inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                    cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
                    _, eigvecs = torch.linalg.eigh(cov)
                    ref_data = torch.matmul(inputs, eigvecs[:, -1:])  # PC1
                except Exception as e:
                    logger.warning(f"MI_gaussian: PC computation failed: {e}, using first neuron")
                    ref_data = outputs[:, :1]
            else:
                # Use first neuron as reference if no other option
                ref_data = outputs[:, :1]
        else:
            ref_data = target_outputs
            if ref_data.ndim == 1:
                ref_data = ref_data.unsqueeze(1)
        
        # Ensure ref_data has correct batch size
        if ref_data.shape[0] != batch_size:
            logger.warning(f"MI_gaussian: Reference batch size mismatch")
            return torch.zeros(num_neurons, device=outputs.device, dtype=outputs.dtype)
        
        # Compute MI for each neuron
        mi_scores = torch.zeros(num_neurons, device=outputs.device)
        
        for i in range(num_neurons):
            neuron_out = outputs[:, i]
            
            # Average MI across all reference dimensions
            mi_sum = 0.0
            valid_refs = 0
            
            for k in range(ref_data.shape[1]):
                ref_k = ref_data[:, k]
                
                # Compute correlation
                combined = torch.stack([neuron_out, ref_k], dim=1)
                if self._should_use_cpu(combined):
                    combined = combined.cpu()
                
                cov_matrix = torch.cov(combined.T)
                var_neuron = cov_matrix[0, 0]
                var_ref = cov_matrix[1, 1]
                
                if var_neuron > 1e-12 and var_ref > 1e-12:
                    rho_sq = (cov_matrix[0, 1] ** 2) / (var_neuron * var_ref)
                    rho_sq = torch.clamp(rho_sq, 0, 0.999999)
                    mi_sum += -0.5 * torch.log(1.0 - rho_sq)
                    valid_refs += 1
            
            if valid_refs > 0:
                mi_scores[i] = mi_sum / valid_refs
        
        return torch.nan_to_num(mi_scores)


@register_metric("mutual_information_binning", aliases=["mi_binning", "mi_1"])
class MutualInformationBinning(BaseMetric):
    """
    Mutual Information using histogram binning method.
    
    Computes MI using discrete probability distributions estimated
    through histogram binning of continuous values.
    """
    
    def __init__(
        self,
        bins: int = 10,
        min_samples: int = 50,
        **config: Any
    ):
        """
        Initialize the binning MI metric.
        
        Args:
            bins: Number of bins for histogram
            min_samples: Minimum samples (need more for binning)
            **config: Additional configuration
        """
        super().__init__(**config)
        self.bins = bins
        self.min_samples = min_samples
    
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
        target_outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute binning-based MI for each output neuron.
        
        Args:
            inputs: Not used
            weights: Not used
            outputs: Layer output activations [batch_size, num_neurons]
            target_outputs: Target reference signal [batch_size, num_targets]
            
        Returns:
            MI values for each neuron [num_neurons]
        """
        if outputs is None:
            raise ValueError("MutualInformationBinning requires outputs")
        
        if outputs.ndim != 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        batch_size, num_neurons = outputs.shape
        
        if batch_size < self.min_samples:
            logger.warning(f"MI_binning: Only {batch_size} samples, need {self.min_samples}")
            return torch.zeros(num_neurons, device=outputs.device, dtype=outputs.dtype)
        
        # Convert to numpy for binning operations
        outputs_np = outputs.cpu().numpy()
        
        # Determine reference
        if target_outputs is not None:
            ref_np = target_outputs.cpu().numpy()
            if ref_np.ndim == 1:
                ref_np = ref_np.reshape(-1, 1)
        else:
            # Use mean of other neurons as reference
            ref_np = None
        
        mi_scores = np.zeros(num_neurons)
        
        for i in range(num_neurons):
            neuron_i_np = outputs_np[:, i]
            
            # Get reference for this neuron
            if ref_np is None and num_neurons > 1:
                # Use mean of other neurons
                other_indices = [j for j in range(num_neurons) if j != i]
                if other_indices:
                    current_ref_np = np.mean(outputs_np[:, other_indices], axis=1, keepdims=True)
                else:
                    continue
            else:
                current_ref_np = ref_np
            
            if current_ref_np is None:
                continue
            
            # Compute MI with each reference dimension
            mi_sum = 0.0
            valid_refs = 0
            
            for k in range(current_ref_np.shape[1]):
                ref_k_np = current_ref_np[:, k]
                
                # Bin the data
                hist_2d, x_edges, y_edges = np.histogram2d(
                    neuron_i_np, ref_k_np, bins=self.bins
                )
                
                # Convert to probabilities
                joint_p = hist_2d / batch_size
                p_x = np.sum(joint_p, axis=1)
                p_y = np.sum(joint_p, axis=0)
                
                # Compute MI
                mi_val = 0.0
                for xi in range(self.bins):
                    for yi in range(self.bins):
                        if joint_p[xi, yi] > 1e-12 and p_x[xi] > 1e-12 and p_y[yi] > 1e-12:
                            mi_val += joint_p[xi, yi] * np.log2(
                                joint_p[xi, yi] / (p_x[xi] * p_y[yi])
                            )
                
                mi_sum += mi_val
                valid_refs += 1
            
            if valid_refs > 0:
                mi_scores[i] = mi_sum / valid_refs
        
        return torch.tensor(mi_scores, device=outputs.device, dtype=outputs.dtype)


@register_metric("conditional_mutual_information")
class ConditionalMutualInformation(MutualInformationGaussian):
    """
    Conditional Mutual Information I(X;Y|Z).
    
    Measures information between X and Y that is not explained by Z.
    Uses Gaussian approximation.
    """
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        target_outputs: Optional[torch.Tensor] = None,
        condition_on: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute conditional MI: I(outputs; target | condition).
        
        Args:
            outputs: X variable [batch_size, num_neurons]
            target_outputs: Y variable [batch_size, num_targets]
            condition_on: Z variable to condition on [batch_size, num_conditions]
            
        Returns:
            Conditional MI values [num_neurons]
        """
        if outputs is None or target_outputs is None or condition_on is None:
            raise ValueError("ConditionalMI requires outputs, target_outputs, and condition_on")
        
        # Compute I(X;Y) - I(X;Y;Z) using Gaussian approximation
        # This is a simplified implementation
        
        # Regular MI between outputs and targets
        mi_xy = super().compute(
            outputs=outputs,
            target_outputs=target_outputs,
            **kwargs
        )
        
        # MI between outputs and condition
        mi_xz = super().compute(
            outputs=outputs,
            target_outputs=condition_on,
            **kwargs
        )
        
        # MI between targets and condition  
        mi_yz = super().compute(
            outputs=target_outputs,
            target_outputs=condition_on,
            **kwargs
        )
        
        # Approximate CMI (this is simplified - proper implementation would use
        # partial correlations or multivariate Gaussian formulas)
        cmi = mi_xy - torch.min(mi_xz, mi_yz.mean() * torch.ones_like(mi_xz))
        cmi = torch.clamp(cmi, min=0.0)  # CMI is non-negative
        
        return cmi 