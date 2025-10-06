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

    def __init__(
        self,
        aggregate_method: str = "l2",
        use_absolute: bool = True
    ):
        super().__init__()
        self.aggregate_method = aggregate_method
        self.use_absolute = use_absolute

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute activation-based importance scores.

        Args:
            inputs: Input activations [batch_size, input_dim] or [seq_len, batch_size, input_dim]
            weights: Weight matrix (not used, but kept for interface compatibility)
            outputs: Output activations [batch_size, num_neurons] or [seq_len, batch_size, num_neurons]

        Returns:
            Importance scores [num_neurons]
        """
        # Use outputs if available, otherwise compute from inputs and weights
        if outputs is not None:
            activations = outputs
        elif inputs is not None and weights is not None:
            # Compute activations
            if inputs.ndim == 2:
                activations = torch.matmul(inputs, weights.T)
            elif inputs.ndim == 3:
                # [seq_len, batch_size, input_dim] @ [num_neurons, input_dim].T
                activations = torch.matmul(inputs, weights.T)
            else:
                raise ValueError(f"Unsupported input shape: {inputs.shape}")
        else:
            raise ValueError("Must provide either outputs or (inputs + weights)")

        # Handle different input shapes
        if activations.ndim == 3:
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
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Compute variance of activations per neuron."""
        # Use outputs if available
        if outputs is not None:
            activations = outputs
        elif inputs is not None and weights is not None:
            if inputs.ndim == 2:
                activations = torch.matmul(inputs, weights.T)
            else:
                activations = torch.matmul(inputs, weights.T)
        else:
            raise ValueError("Must provide either outputs or (inputs + weights)")

        # Handle 3D activations (seq_len, batch, neurons)
        if activations.ndim == 3:
            # Combine seq and batch dimensions
            activations = activations.reshape(-1, activations.shape[-1])

        # Compute variance per neuron
        variance = activations.var(dim=0)

        return variance

