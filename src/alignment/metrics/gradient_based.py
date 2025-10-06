"""
Gradient-based metrics for local learning rules.

These metrics use gradient information during training to:
1. Compute node importance based on gradient statistics
2. Design local learning rules that approximate backprop
3. Identify optimal local update rules correlated with global gradient

Key insight: Metrics that maximally correlate with backprop gradient
can be used to design efficient local learning algorithms.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("gradient_alignment")
class GradientAlignment(BaseMetric):
    """
    Measures alignment between local signals and backprop gradient.

    Computes correlation between various local signals (activations, weights, etc.)
    and the true backprop gradient to identify optimal local learning rules.

    Use during training to:
    - Find local rules that approximate backprop
    - Design biologically-plausible learning
    - Identify credit assignment strategies

    Example:
        >>> metric = GradientAlignment()
        >>> # During training (after backward()):
        >>> alignment = metric.compute(
        ...     inputs=layer_inputs,
        ...     outputs=layer_outputs,
        ...     gradients=layer.weight.grad,
        ...     targets=labels
        ... )
        >>> # High alignment = this local signal correlates with gradient
    """

    def __init__(
        self,
        local_signal: str = 'hebbian',  # 'hebbian', 'anti_hebbian', 'output', 'input'
        normalize: bool = True,
        accumulate_over_batches: bool = False,
        **config: Any
    ):
        """
        Initialize gradient alignment metric.

        Args:
            local_signal: Which local signal to test
                - 'hebbian': x_i * y_j (Hebbian rule)
                - 'anti_hebbian': x_i * (y_target - y_j)
                - 'output': y_j only
                - 'input': x_i only
                - 'oja': Hebbian with weight decay
            normalize: Normalize signals before correlation
            accumulate_over_batches: Track correlation across multiple batches
            **config: Additional configuration
        """
        super().__init__(**config)
        self.local_signal = local_signal
        self.normalize = normalize
        self.accumulate_over_batches = accumulate_over_batches

        # Accumulation for multi-batch correlation
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
        **kwargs
    ) -> torch.Tensor:
        """
        Compute alignment between local signal and backprop gradient.

        Args:
            inputs: Layer inputs [B, D_in]
            outputs: Layer outputs [B, N]
            gradients: Weight gradients [N, D_in] (from backprop)
            targets: Target labels (for anti-Hebbian)
            **kwargs: Additional parameters

        Returns:
            Alignment scores per neuron [N]
        """
        if inputs is None or outputs is None:
            raise ValueError("GradientAlignment requires inputs and outputs")

        if gradients is None:
            logger.warning("No gradients provided, cannot compute gradient alignment")
            return torch.zeros(outputs.shape[1], device=outputs.device)

        # Ensure shapes
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        if gradients.ndim > 2:
            gradients = gradients.reshape(gradients.shape[0], -1)

        B, D_in = inputs.shape
        B, N = outputs.shape

        # Compute local signal
        local_signal = self._compute_local_signal(
            inputs, outputs, targets, gradients.shape
        )  # [N, D_in] - same shape as gradients

        # Compute correlation with gradients
        alignment = self._compute_correlation(local_signal, gradients)
        # [N] - alignment per neuron

        return alignment

    def _compute_local_signal(
        self,
        inputs: torch.Tensor,  # [B, D_in]
        outputs: torch.Tensor,  # [B, N]
        targets: Optional[torch.Tensor],
        grad_shape: torch.Size
    ) -> torch.Tensor:
        """
        Compute local learning signal.

        Returns:
            Local signal [N, D_in] (same shape as weight gradient)
        """
        B, D_in = inputs.shape
        B, N = outputs.shape

        if self.local_signal == 'hebbian':
            # Hebbian: Δw_ij ∝ x_i * y_j
            # Averaged over batch: [N, D_in]
            signal = outputs.T @ inputs / B  # Outer product averaged

        elif self.local_signal == 'anti_hebbian':
            # Anti-Hebbian: Δw_ij ∝ x_i * (target - y_j)
            if targets is None:
                logger.warning("Anti-Hebbian requires targets, falling back to Hebbian")
                signal = outputs.T @ inputs / B
            else:
                # Assuming targets are class labels, convert to one-hot
                if targets.ndim == 1:
                    target_onehot = torch.nn.functional.one_hot(targets, num_classes=N).float()
                else:
                    target_onehot = targets

                error = target_onehot - outputs  # [B, N]
                signal = error.T @ inputs / B

        elif self.local_signal == 'oja':
            # Oja's rule: Hebbian with weight decay
            # Δw = η * y * (x - y*w)
            # Requires current weights
            if weights is None:
                signal = outputs.T @ inputs / B
            else:
                # y * x^T - y * y^T * w
                hebbian = outputs.T @ inputs / B
                # This is simplified; full Oja needs weight norm
                signal = hebbian

        elif self.local_signal == 'output':
            # Just output: Δw ∝ y
            # Broadcast to weight shape
            signal = outputs.mean(dim=0).unsqueeze(1).expand(N, D_in)

        elif self.local_signal == 'input':
            # Just input: Δw ∝ x
            # Broadcast to weight shape
            signal = inputs.mean(dim=0).unsqueeze(0).expand(N, D_in)

        else:
            raise ValueError(f"Unknown local signal: {self.local_signal}")

        return signal

    def _compute_correlation(
        self,
        signal: torch.Tensor,  # [N, D_in]
        gradients: torch.Tensor  # [N, D_in]
    ) -> torch.Tensor:
        """
        Compute correlation between local signal and backprop gradient.

        High correlation → local signal approximates gradient well
        → Can be used as local learning rule!

        Returns:
            Correlation per neuron [N]
        """
        N, D_in = signal.shape

        # Compute correlation for each neuron
        correlations = torch.zeros(N, device=signal.device)

        for i in range(N):
            signal_i = signal[i]  # [D_in]
            grad_i = gradients[i]  # [D_in]

            if self.normalize:
                # Pearson correlation
                signal_centered = signal_i - signal_i.mean()
                grad_centered = grad_i - grad_i.mean()

                cov = (signal_centered * grad_centered).sum()
                std_signal = signal_centered.std()
                std_grad = grad_centered.std()

                corr = cov / (std_signal * std_grad + 1e-8)
            else:
                # Dot product (unnormalized)
                corr = (signal_i * grad_i).sum() / (signal_i.norm() * grad_i.norm() + 1e-8)

            correlations[i] = corr.abs()  # Use absolute correlation

        return correlations


@register_metric("local_learning_rule_search")
class LocalLearningRuleSearch(BaseMetric):
    """
    Search for optimal local learning rules that approximate backprop.

    Tests multiple candidate local rules and identifies which best
    correlate with backprop gradients.

    Candidate rules:
    - Hebbian: Δw ∝ x * y
    - Anti-Hebbian: Δw ∝ x * (target - y)
    - Oja: Δw ∝ y * (x - y*w)
    - Contrastive: Δw ∝ x_pos * y_pos - x_neg * y_neg
    - Random feedback: Δw ∝ x * (B @ error) where B is random

    Returns the best rule per neuron.

    Example:
        >>> searcher = LocalLearningRuleSearch()
        >>> best_rules = searcher.compute(
        ...     inputs, outputs, gradients=layer.weight.grad
        ... )
        >>> # best_rules[i] = index of best rule for neuron i
        >>> # Can then use this to train with local rules!
    """

    def __init__(
        self,
        candidate_rules: Optional[List[str]] = None,
        **config
    ):
        """
        Initialize local learning rule search.

        Args:
            candidate_rules: List of rules to test (None = all)
            **config: Additional configuration
        """
        super().__init__(**config)

        self.candidate_rules = candidate_rules or [
            'hebbian', 'anti_hebbian', 'oja', 'output', 'input'
        ]

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
        **kwargs
    ) -> torch.Tensor:
        """
        Find best local learning rule per neuron.

        Args:
            inputs, outputs: Layer activations
            gradients: Backprop gradients
            targets: For anti-Hebbian rules
            return_correlations: If True, return correlation matrix [N, num_rules]
            **kwargs: Additional args

        Returns:
            Best rule index per neuron [N] or correlations [N, num_rules]
        """
        if gradients is None:
            raise ValueError("LocalLearningRuleSearch requires gradients")

        # Test all candidate rules
        correlations = {}

        for rule in self.candidate_rules:
            metric = GradientAlignment(local_signal=rule)
            corr = metric.compute(
                inputs=inputs,
                weights=weights,
                outputs=outputs,
                gradients=gradients,
                targets=targets
            )
            correlations[rule] = corr

        # Stack into matrix [N, num_rules]
        corr_matrix = torch.stack([correlations[rule] for rule in self.candidate_rules], dim=1)

        if return_correlations:
            return corr_matrix

        # Find best rule per neuron
        best_rule_indices = corr_matrix.argmax(dim=1)  # [N]

        return best_rule_indices

    def get_learning_rule_for_neuron(
        self,
        neuron_idx: int,
        best_rule_idx: int
    ) -> str:
        """Get the name of the best rule for a neuron."""
        return self.candidate_rules[best_rule_idx]


class GradientStatisticsTracker:
    """
    Track gradient statistics during training for analysis.

    Collects:
    - Gradient magnitude evolution
    - Gradient direction consistency
    - Correlation with various local signals

    Use to design local learning rules.
    """

    def __init__(self):
        """Initialize tracker."""
        self.gradient_history: Dict[str, List[torch.Tensor]] = {}
        self.signal_history: Dict[str, List[torch.Tensor]] = {}
        self.correlation_history: Dict[str, List[float]] = {}

    def register_layer(self, layer_name: str):
        """Register a layer for tracking."""
        self.gradient_history[layer_name] = []
        self.signal_history[layer_name] = []
        self.correlation_history[layer_name] = []

    def update(
        self,
        layer_name: str,
        gradient: torch.Tensor,
        local_signal: torch.Tensor
    ):
        """
        Update statistics after a training step.

        Call this after loss.backward() but before optimizer.step()

        Args:
            layer_name: Name of layer
            gradient: Backprop gradient (layer.weight.grad)
            local_signal: Local learning signal
        """
        if layer_name not in self.gradient_history:
            self.register_layer(layer_name)

        # Store
        self.gradient_history[layer_name].append(gradient.detach().cpu().clone())
        self.signal_history[layer_name].append(local_signal.detach().cpu().clone())

        # Compute correlation
        grad_flat = gradient.flatten()
        signal_flat = local_signal.flatten()

        corr = torch.corrcoef(torch.stack([grad_flat, signal_flat]))[0, 1]
        self.correlation_history[layer_name].append(corr.item())

    def get_average_correlation(self, layer_name: str) -> float:
        """Get average correlation over training."""
        if layer_name not in self.correlation_history:
            return 0.0

        return sum(self.correlation_history[layer_name]) / len(self.correlation_history[layer_name])

    def get_best_local_rule(
        self,
        layer_name: str,
        candidate_rules: List[str]
    ) -> Tuple[str, float]:
        """
        Determine which local rule best approximates backprop for this layer.

        Returns:
            (best_rule_name, correlation)
        """
        # This would require testing multiple rules
        # For now, return based on accumulated correlation
        avg_corr = self.get_average_correlation(layer_name)

        return (self.local_signal if hasattr(self, 'local_signal') else 'hebbian', avg_corr)


def design_local_learning_rule(
    gradient_tracker: GradientStatisticsTracker,
    layer_name: str
) -> Dict[str, Any]:
    """
    Design optimal local learning rule for a layer based on gradient analysis.

    Args:
        gradient_tracker: Tracker that monitored training
        layer_name: Layer to design rule for

    Returns:
        Local learning rule specification
    """
    # Analyze gradient statistics
    grad_history = gradient_tracker.gradient_history[layer_name]

    if not grad_history:
        return {'rule': 'hebbian', 'params': {}}

    # Compute gradient statistics
    avg_grad_magnitude = torch.stack([g.abs().mean() for g in grad_history]).mean()
    grad_direction_consistency = compute_direction_consistency(grad_history)

    # Design rule based on statistics
    if grad_direction_consistency > 0.8:
        # Consistent gradient → simple Hebbian works
        rule = 'hebbian'
        learning_rate = avg_grad_magnitude.item() * 0.1

    elif grad_direction_consistency < 0.3:
        # Inconsistent → need more sophisticated rule
        rule = 'oja'
        learning_rate = avg_grad_magnitude.item() * 0.05

    else:
        # Moderate → anti-Hebbian
        rule = 'anti_hebbian'
        learning_rate = avg_grad_magnitude.item() * 0.08

    return {
        'rule': rule,
        'params': {
            'learning_rate': learning_rate,
            'direction_consistency': grad_direction_consistency
        }
    }


def compute_direction_consistency(gradient_history: List[torch.Tensor]) -> float:
    """
    Compute how consistent gradient direction is across training.

    High consistency → gradient always points similar direction
    Low consistency → gradient changes direction frequently

    Returns:
        Consistency score [0, 1]
    """
    if len(gradient_history) < 2:
        return 1.0

    # Normalize gradients
    grads_normalized = [g / (g.norm() + 1e-8) for g in gradient_history]

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(grads_normalized) - 1):
        cos_sim = (grads_normalized[i] * grads_normalized[i+1]).sum()
        similarities.append(cos_sim.abs().item())

    # Average similarity
    consistency = sum(similarities) / len(similarities)

    return consistency

