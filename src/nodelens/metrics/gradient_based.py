"""
Gradient-based metrics for local learning rules and saliency.

These metrics use gradient information to:
1. Compute node importance (e.g. Taylor expansion saliency)
2. Compute alignment between local signals and backprop gradients
3. Design local learning rules
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("taylor_saliency")
class TaylorSaliency(BaseMetric):
    """
    First-order Taylor expansion saliency metric.

    Approximates the change in loss if a neuron/channel is pruned (masked to 0).
    For a neuron i with output u_i and gradient g_i = dL/du_i:
        Saliency_i = |u_i * g_i|

    This measures the "force" the neuron exerts on the loss.
    Can be computed:
    1. Per sample: |u_i * g_i|
    2. Averaged over batch: mean(|u_i * g_i|)  <-- Standard definition
    3. Squared variant: (u_i * g_i)^2
    """

    def __init__(self, mode: str = "abs_mean", **config: Any):
        """
        Initialize Taylor Saliency metric.

        Args:
            mode: Aggregation mode
                - 'abs_mean': E[|u * g|] (Standard First-Order Taylor)
                - 'mean_abs': |E[u * g]| (Magnitude of average contribution)
                - 'sq_mean': E[(u * g)^2] (Fisher-like)
            **config: Additional configuration
        """
        super().__init__(**config)
        self.mode = mode

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True  # Need activation u

    @property
    def requires_gradients(self) -> bool:
        return True  # Need gradient dL/du

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Taylor Saliency.

        Args:
            outputs: Activations u [batch_size, num_neurons]
            gradients: Gradients dL/du [batch_size, num_neurons]

        Returns:
            Saliency scores [num_neurons]
        """
        if outputs is None or gradients is None:
            raise ValueError("TaylorSaliency requires both outputs and gradients (activations and their grads)")

        # Ensure same shape
        if outputs.shape != gradients.shape:
            # Try to reshape gradients to match outputs if just a batch mismatch or flattened
            if outputs.numel() == gradients.numel():
                gradients = gradients.view_as(outputs)
            else:
                raise ValueError(f"Shape mismatch: outputs {outputs.shape}, gradients {gradients.shape}")

        # CNN conv outputs: [B, C, H, W] (or similar). Here the neuron/channel dimension is 1.
        # Compute per-channel saliency by reducing over batch + spatial dims.
        if outputs.ndim == 4:
            product = outputs * gradients
            if self.mode == "abs_mean":
                return product.abs().mean(dim=(0, 2, 3))
            if self.mode == "mean_abs":
                return product.mean(dim=(0, 2, 3)).abs()
            if self.mode == "sq_mean":
                return (product**2).mean(dim=(0, 2, 3))
            raise ValueError(f"Unknown Taylor Saliency mode: {self.mode}")

        # Flatten batch dimension if needed, keep neuron dimension
        # Assuming [batch, neurons] or [batch, tokens, neurons]
        if outputs.ndim > 2:
            outputs = outputs.reshape(-1, outputs.shape[-1])
            gradients = gradients.reshape(-1, gradients.shape[-1])

        product = outputs * gradients

        if self.mode == "abs_mean":
            # E[|u * g|] - standard magnitude pruning proxy
            scores = product.abs().mean(dim=0)

        elif self.mode == "mean_abs":
            # |E[u * g]| - magnitude of net contribution
            scores = product.mean(dim=0).abs()

        elif self.mode == "sq_mean":
            # E[(u * g)^2]
            scores = (product**2).mean(dim=0)

        else:
            raise ValueError(f"Unknown Taylor Saliency mode: {self.mode}")

        return scores


@register_metric("gradient_alignment")
class GradientAlignment(BaseMetric):
    """
    Measures alignment between local signals and backprop gradient.

    Computes correlation between various local signals (activations, weights, etc.)
    and the true backprop gradient to identify optimal local learning rules.
    """

    def __init__(
        self,
        local_signal: str = "hebbian",  # 'hebbian', 'anti_hebbian', 'output', 'input'
        normalize: bool = True,
        accumulate_over_batches: bool = False,
        **config: Any,
    ):
        super().__init__(**config)
        self.local_signal = local_signal
        self.normalize = normalize
        self.accumulate_over_batches = accumulate_over_batches

        if accumulate_over_batches:
            self.accumulated_gradients = {}
            self.accumulated_signals = {}

    @property
    def requires_inputs(self) -> bool:
        return True

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
        gradients: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs is None or outputs is None:
            raise ValueError("GradientAlignment requires inputs and outputs")

        if gradients is None:
            logger.warning("No gradients provided, cannot compute gradient alignment")
            if outputs.ndim > 1:
                return torch.zeros(outputs.shape[1], device=outputs.device)
            return torch.tensor(0.0, device=outputs.device)

        # Ensure shapes
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        if gradients.ndim > 2:
            gradients = gradients.reshape(gradients.shape[0], -1)

        B, D_in = inputs.shape
        B, N = outputs.shape

        local_signal = self._compute_local_signal(inputs, outputs, targets, gradients.shape)
        alignment = self._compute_correlation(local_signal, gradients)

        return alignment

    def _compute_local_signal(
        self, inputs: torch.Tensor, outputs: torch.Tensor, targets: Optional[torch.Tensor], grad_shape: torch.Size
    ) -> torch.Tensor:
        B, D_in = inputs.shape
        B, N = outputs.shape

        if self.local_signal == "hebbian":
            signal = outputs.T @ inputs / B
        elif self.local_signal == "anti_hebbian":
            if targets is None:
                logger.warning("Anti-Hebbian requires targets, falling back to Hebbian")
                signal = outputs.T @ inputs / B
            else:
                if targets.ndim == 1:
                    target_onehot = F.one_hot(targets, num_classes=N).float()
                else:
                    target_onehot = targets
                error = target_onehot - outputs
                signal = error.T @ inputs / B
        elif self.local_signal == "oja":
            signal = outputs.T @ inputs / B
        elif self.local_signal == "output":
            signal = outputs.mean(dim=0).unsqueeze(1).expand(N, D_in)
        elif self.local_signal == "input":
            signal = inputs.mean(dim=0).unsqueeze(0).expand(N, D_in)
        else:
            raise ValueError(f"Unknown local signal: {self.local_signal}")

        return signal

    def _compute_correlation(self, signal: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        N, D_in = signal.shape
        correlations = torch.zeros(N, device=signal.device)

        for i in range(N):
            signal_i = signal[i]
            grad_i = gradients[i]

            if self.normalize:
                signal_centered = signal_i - signal_i.mean()
                grad_centered = grad_i - grad_i.mean()
                cov = (signal_centered * grad_centered).sum()
                std_signal = signal_centered.std()
                std_grad = grad_centered.std()
                corr = cov / (std_signal * std_grad + 1e-8)
            else:
                corr = (signal_i * grad_i).sum() / (signal_i.norm() * grad_i.norm() + 1e-8)

            correlations[i] = corr.abs()

        return correlations


@register_metric("local_learning_rule_search")
class LocalLearningRuleSearch(BaseMetric):
    def __init__(self, candidate_rules: Optional[List[str]] = None, **config):
        super().__init__(**config)
        self.candidate_rules = candidate_rules or ["hebbian", "anti_hebbian", "oja", "output", "input"]

    @property
    def requires_inputs(self) -> bool:
        return True

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
        gradients: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        return_correlations: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if gradients is None:
            raise ValueError("LocalLearningRuleSearch requires gradients")

        correlations = {}
        for rule in self.candidate_rules:
            metric = GradientAlignment(local_signal=rule)
            corr = metric.compute(inputs=inputs, weights=weights, outputs=outputs, gradients=gradients, targets=targets)
            correlations[rule] = corr

        corr_matrix = torch.stack([correlations[rule] for rule in self.candidate_rules], dim=1)

        if return_correlations:
            return corr_matrix

        best_rule_indices = corr_matrix.argmax(dim=1)
        return best_rule_indices

    def get_learning_rule_for_neuron(self, neuron_idx: int, best_rule_idx: int) -> str:
        return self.candidate_rules[best_rule_idx]


class GradientStatisticsTracker:
    def __init__(self):
        self.gradient_history: Dict[str, List[torch.Tensor]] = {}
        self.signal_history: Dict[str, List[torch.Tensor]] = {}
        self.correlation_history: Dict[str, List[float]] = {}

    def register_layer(self, layer_name: str):
        self.gradient_history[layer_name] = []
        self.signal_history[layer_name] = []
        self.correlation_history[layer_name] = []

    def update(self, layer_name: str, gradient: torch.Tensor, local_signal: torch.Tensor):
        if layer_name not in self.gradient_history:
            self.register_layer(layer_name)

        self.gradient_history[layer_name].append(gradient.detach().cpu().clone())
        self.signal_history[layer_name].append(local_signal.detach().cpu().clone())

        grad_flat = gradient.flatten()
        signal_flat = local_signal.flatten()
        corr = torch.corrcoef(torch.stack([grad_flat, signal_flat]))[0, 1]
        self.correlation_history[layer_name].append(corr.item())

    def get_average_correlation(self, layer_name: str) -> float:
        if layer_name not in self.correlation_history:
            return 0.0
        return sum(self.correlation_history[layer_name]) / len(self.correlation_history[layer_name])


def compute_direction_consistency(gradient_history: List[torch.Tensor]) -> float:
    if len(gradient_history) < 2:
        return 1.0
    grads_normalized = [g / (g.norm() + 1e-8) for g in gradient_history]
    similarities = []
    for i in range(len(grads_normalized) - 1):
        cos_sim = (grads_normalized[i] * grads_normalized[i + 1]).sum()
        similarities.append(cos_sim.abs().item())
    return sum(similarities) / len(similarities)
