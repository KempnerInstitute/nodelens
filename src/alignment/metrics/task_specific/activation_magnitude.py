"""
Activation magnitude-based importance metrics.

These metrics compute neuron importance based on activation magnitudes,
commonly used in pruning literature including TensorRT and NeMo.
"""

import logging
from typing import Any, Optional

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("activation_l2_norm")
class ActivationL2Norm(BaseMetric):
    """
    Compute neuron importance as L2 norm of activations.

    This metric is commonly used in LLM pruning (e.g., TensorRT-LLM, NeMo).
    For each neuron, computes: sqrt(sum_over_batch(mean_over_seq(|activations|)^2))

    This is equivalent to the metric used in:
    - NVIDIA NeMo pruning
    - TensorRT-LLM pruning
    - Various LLM compression papers

    Args:
        aggregate_method: How to aggregate activations ('l2', 'mean', 'max')
        use_absolute: Whether to take absolute value before aggregation
    """

    name = "activation_l2_norm"
    requires_inputs = True
    requires_weights = False
    requires_outputs = True

    def __init__(self, aggregate_method: str = "l2", use_absolute: bool = True):
        super().__init__()
        self.aggregate_method = aggregate_method
        self.use_absolute = use_absolute

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute activation-based importance scores.

        Args:
            inputs: Input activations [batch_size, input_dim] or [seq_len, batch_size, input_dim]
            weights: Weight matrix [out_features, in_features] or [out_channels, in_channels, kH, kW]
            outputs: Output activations [batch_size, num_neurons] or [seq_len, batch_size, num_neurons]

        Returns:
            Importance scores [num_neurons] - one score per OUTPUT neuron/channel
        """
        # Use outputs if available, otherwise compute from inputs and weights
        if outputs is not None:
            activations = outputs
        elif inputs is not None and weights is not None:
            # Compute activations - need to produce [batch, out_features] shape
            if inputs.ndim == 2:
                # Linear: [batch, in_features] @ [out_features, in_features].T -> [batch, out_features]
                activations = torch.matmul(inputs, weights.T)
            elif inputs.ndim == 3:
                # Transformer: [seq_len, batch_size, input_dim] @ [num_neurons, input_dim].T
                activations = torch.matmul(inputs, weights.T)
            elif inputs.ndim == 4:
                # CNN: [batch, in_channels, height, width]
                # weights: [out_channels, in_channels, kH, kW]
                # We need to compute per-output-channel importance
                # Use weight magnitude combined with input activation as proxy
                batch_size = inputs.shape[0]
                out_channels = weights.shape[0]
                
                # Compute input activation magnitude per input channel
                input_mag = inputs.abs().mean(dim=(0, 2, 3))  # [in_channels]
                
                # Weight magnitude per output channel (sum over in_channels and kernel)
                weight_mag = weights.abs().sum(dim=(1, 2, 3))  # [out_channels]
                
                # Combine: scale weight magnitude by mean input magnitude
                # This gives a proxy for output activation magnitude per output channel
                mean_input_mag = input_mag.mean()
                importance = weight_mag * mean_input_mag
                return importance
            else:
                raise ValueError(f"Unsupported input shape: {inputs.shape}")
        elif inputs is not None:
            # Only inputs available - compute importance from input activations
            # For CNN, this gives per-input-channel scores, which is wrong for pruning
            # Log a warning
            logger.warning("ActivationL2Norm: No weights provided, using input activations only. "
                          "Scores will be per input channel, not per output channel.")
            activations = inputs
        else:
            raise ValueError("Must provide either outputs or (inputs + weights)")

        # Handle different input shapes
        if activations.ndim == 4:
            # CNN: [batch, channels, height, width]
            # Compute per-channel importance by averaging over spatial dimensions
            if self.use_absolute:
                activations = activations.abs()
            # Average over spatial dimensions (H, W), keep batch and channels
            activations = activations.mean(dim=(2, 3))  # [batch, channels]
            
        elif activations.ndim == 3:
            # [seq_len, batch_size, num_neurons] - typical for transformers
            # This matches PruneLLM's format

            if self.use_absolute:
                activations = activations.abs()

            # Mean over sequence dimension
            activations = activations.mean(dim=0)  # [batch_size, num_neurons]

        elif activations.ndim == 2:
            # [batch_size, num_neurons] - typical for MLPs/CNNs
            if self.use_absolute:
                activations = activations.abs()
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")

        # Now activations is [batch_size, num_neurons]
        # Compute importance based on method
        if self.aggregate_method == "l2":
            # L2 norm across batch: sqrt(sum(x^2))
            # This matches PruneLLM exactly: activations.pow(2).sum(dim=0).sqrt()
            importance = activations.pow(2).sum(dim=0).sqrt()

        elif self.aggregate_method == "mean":
            # Mean activation magnitude
            importance = activations.mean(dim=0)

        elif self.aggregate_method == "max":
            # Max activation magnitude
            importance = activations.max(dim=0)[0]

        else:
            raise ValueError(f"Unknown aggregate_method: {self.aggregate_method}")

        return importance


@register_metric("activation_mean")
class ActivationMean(ActivationL2Norm):
    """
    Compute neuron importance as mean absolute activation.

    For each neuron, computes the average magnitude of activations across
    all samples in the batch.
    """

    name = "activation_mean"

    def __init__(self):
        super().__init__(aggregate_method="mean", use_absolute=True)


@register_metric("activation_norm")
class ActivationNorm(ActivationL2Norm):
    """
    Compute neuron importance as L2 norm of activations.

    For each neuron, computes: sqrt(sum_over_batch(mean_over_seq(|activations|)^2))

    This is a standard metric used in:
    - LLM pruning (TensorRT-LLM, NeMo)
    - Neural architecture search
    - Channel pruning for CNNs

    Formula:
        importance[n] = ||activations[:, n]||_2

    For transformers with shape [seq_len, batch, neurons]:
        1. Take absolute value: |activations|
        2. Mean over sequence: mean(dim=0) -> [batch, neurons]
        3. Square: activations^2
        4. Sum over batch: sum(dim=0) -> [neurons]
        5. Square root: sqrt() -> [neurons]
    """

    name = "activation_norm"

    def __init__(self):
        super().__init__(aggregate_method="l2", use_absolute=True)


@register_metric("activation_variance")
class ActivationVariance(BaseMetric):
    """
    Compute neuron importance as variance of activations.

    High variance neurons are more selective/informative.
    """

    name = "activation_variance"
    requires_inputs = True
    requires_weights = False
    requires_outputs = True

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """Compute variance of activations per neuron."""
        # Use outputs if available
        if outputs is not None:
            activations = outputs
        elif inputs is not None and weights is not None:
            if inputs.ndim == 2:
                activations = torch.matmul(inputs, weights.T)
            elif inputs.ndim == 4:
                # CNN: use inputs directly
                activations = inputs
            else:
                activations = torch.matmul(inputs, weights.T)
        else:
            raise ValueError("Must provide either outputs or (inputs + weights)")

        # Handle different activation shapes
        if activations.ndim == 4:
            # CNN: [batch, channels, height, width]
            # Compute variance per channel over batch and spatial dimensions
            # Reshape to [batch * height * width, channels]
            b, c, h, w = activations.shape
            activations = activations.permute(0, 2, 3, 1).reshape(-1, c)
        elif activations.ndim == 3:
            # Combine seq and batch dimensions
            activations = activations.reshape(-1, activations.shape[-1])

        # Compute variance per neuron/channel
        variance = activations.var(dim=0)

        return variance


@register_metric("activation_outlier_index")
class ActivationOutlierIndex(BaseMetric):
    """
    Compute an outlier index for each neuron/channel.

    Defined as the ratio between a high-percentile activation magnitude
    (default 99.9th) and the mean absolute activation. This highlights
    "supernode" style channels whose activations spike far above their mean.
    """

    name = "activation_outlier_index"
    requires_inputs = False
    requires_weights = False
    requires_outputs = True

    def __init__(self, quantile: float = 0.999, eps: float = 1e-6):
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")
        self.quantile = quantile
        self.eps = eps

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        if outputs is None:
            if inputs is None or weights is None:
                raise ValueError("activation_outlier_index requires outputs or (inputs + weights)")
            if inputs.ndim == 2:
                outputs = torch.matmul(inputs, weights.T)
            elif inputs.ndim == 4:
                # CNN: use inputs directly
                outputs = inputs
            else:
                outputs = torch.matmul(inputs, weights.T)

        activations = outputs
        if activations.ndim == 4:
            # CNN: [batch, channels, height, width]
            # Reshape to [batch * height * width, channels]
            b, c, h, w = activations.shape
            activations = activations.permute(0, 2, 3, 1).reshape(-1, c)
        elif activations.ndim == 3:
            activations = activations.reshape(-1, activations.shape[-1])
        elif activations.ndim != 2:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")

        # Ensure we use a dtype supported by torch.quantile (float32/float64)
        abs_vals = activations.abs().to(torch.float32)
        high = torch.quantile(abs_vals, self.quantile, dim=0)
        mean = abs_vals.mean(dim=0)

        return high / (mean + self.eps)
