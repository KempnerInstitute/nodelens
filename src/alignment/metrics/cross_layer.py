"""
Cross-Layer Activation Mixing Metrics.

This module provides metrics that analyze cross-layer information flow,
following the SCAR/supernode logic of DOWNSTREAM importance:

1. Downstream importance: How much does layer l+1 depend on neuron i at layer l?
   I(Y^{l+1}; Y_i^l) - High = important, Low = safe to prune

2. Downstream redundancy: Can layer l+1 reconstruct neuron i's info from others?
   If Y_i^l is redundant with other neurons AND next layer depends on those,
   then Y_i^l is safe to prune.

3. Cross-layer importance score: Combines current-layer and next-layer analysis

Theory (cross-layer importance; SCAR-style downstream dependence):
The key insight is that importance flows FORWARD through the network:
- A neuron is important if DOWNSTREAM layers depend on it
- A neuron is redundant if its information is already carried by other neurons
  that downstream layers also use

Score(Y_i^l) = α·RQ(w_i^l) + β·I(Y^{l+1}; Y_i^l) - γ·R(Y_i^l || Y^l)

Where:
- RQ(w_i^l) = Rayleigh Quotient for neuron i at layer l (input alignment)
- I(Y^{l+1}; Y_i^l) = MI with NEXT layer (downstream importance)
- R(Y_i^l || Y^l) = Redundancy with same-layer neurons

High downstream MI = important (like supernode)
High same-layer redundancy = safe to prune (like halo)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@dataclass
class CrossLayerResult:
    """Results from cross-layer analysis."""

    layer_idx: int
    layer_name: str

    # Per-neuron downstream importance (how much next layer depends on each neuron)
    downstream_importance: np.ndarray  # [num_neurons] - HIGHER = more important

    # Per-neuron within-layer redundancy (for comparison)
    within_layer_redundancy: np.ndarray  # [num_neurons] - HIGHER = more redundant

    # Summary statistics
    mean_downstream_importance: float
    mean_within_layer_redundancy: float

    # Derived scores
    # Neurons with high downstream importance and low redundancy are critical
    # Neurons with low downstream importance or high redundancy are prunable
    importance_to_redundancy_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "mean_downstream_importance": self.mean_downstream_importance,
            "mean_within_layer_redundancy": self.mean_within_layer_redundancy,
            "importance_to_redundancy_ratio": self.importance_to_redundancy_ratio,
        }


def compute_downstream_importance(
    current_activations: torch.Tensor,
    next_layer_activations: torch.Tensor,
    max_refs: int = 512,
) -> torch.Tensor:
    """
    Compute downstream importance: how much does the NEXT layer depend on each neuron?

    This follows SCAR logic: neurons are important if downstream layers need them.

    For each neuron i at layer l, computes:
        Importance(Y_i^l) = mean over j of I(Y_i^l; Y_j^{l+1})

    Using Gaussian MI: I(Y_i; Y_j) = -0.5 * log(1 - ρ²)

    Args:
        current_activations: [batch, num_current_neurons] - layer l
        next_layer_activations: [batch, num_next_neurons] - layer l+1
        max_refs: Maximum reference neurons from next layer

    Returns:
        Downstream importance scores [num_current_neurons]
        HIGHER = more important (next layer depends on this neuron)
    """
    batch_size, num_current = current_activations.shape
    num_next = next_layer_activations.shape[1]
    device = current_activations.device

    # Sample reference neurons from next layer if too many
    if num_next > max_refs:
        ref_indices = torch.randperm(num_next, device=device)[:max_refs]
        next_acts = next_layer_activations[:, ref_indices]
    else:
        next_acts = next_layer_activations

    # Normalize both
    curr_centered = current_activations - current_activations.mean(dim=0, keepdim=True)
    curr_std = curr_centered.std(dim=0, keepdim=True)
    curr_std = torch.where(curr_std > 1e-8, curr_std, torch.ones_like(curr_std))
    curr_norm = curr_centered / curr_std

    next_centered = next_acts - next_acts.mean(dim=0, keepdim=True)
    next_std = next_centered.std(dim=0, keepdim=True)
    next_std = torch.where(next_std > 1e-8, next_std, torch.ones_like(next_std))
    next_norm = next_centered / next_std

    # Compute cross-correlation matrix [num_current, num_next_sampled]
    # For each current neuron: how correlated is it with next layer neurons?
    cross_corr = (curr_norm.T @ next_norm) / max(1, batch_size - 1)
    cross_corr = torch.clamp(cross_corr, -0.999, 0.999)

    # Convert to MI (importance measure)
    rho_sq = cross_corr**2
    mi_matrix = -0.5 * torch.log(1 - rho_sq + 1e-8)

    # Average importance (MI with next layer) for each current neuron
    # HIGHER = more important (next layer needs this neuron more)
    downstream_importance = mi_matrix.mean(dim=1)

    return downstream_importance


def compute_within_layer_redundancy(
    activations: torch.Tensor,
    max_refs: int = 512,
) -> torch.Tensor:
    """
    Compute within-layer redundancy for each neuron.

    Args:
        activations: [batch, num_neurons]
        max_refs: Maximum reference neurons

    Returns:
        Within-layer redundancy scores [num_neurons]
    """
    batch_size, num_neurons = activations.shape
    device = activations.device

    # Normalize
    centered = activations - activations.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    normalized = centered / std

    if num_neurons > max_refs:
        # Sample reference neurons
        ref_indices = torch.randperm(num_neurons, device=device)[:max_refs]
        ref_acts = normalized[:, ref_indices]

        # Correlation with references
        corr = (normalized.T @ ref_acts) / max(1, batch_size - 1)
        corr = torch.clamp(corr, -0.999, 0.999)

        # Mask self-correlation (for neurons that are in reference set)
        for idx, ref_idx in enumerate(ref_indices):
            corr[ref_idx, idx] = 0.0

        rho_sq = corr**2
        mi = -0.5 * torch.log(1 - rho_sq + 1e-8)
        redundancy = mi.mean(dim=1)
    else:
        # Full correlation matrix
        corr = (normalized.T @ normalized) / max(1, batch_size - 1)
        corr = torch.clamp(corr, -0.999, 0.999)

        rho_sq = corr**2
        mi = -0.5 * torch.log(1 - rho_sq + 1e-8)
        mi.fill_diagonal_(0)  # Exclude self
        redundancy = mi.sum(dim=1) / max(1, num_neurons - 1)

    return redundancy


@register_metric("cross_layer_redundancy")
class CrossLayerRedundancy(BaseMetric):
    """
    Compute downstream importance: how much does the NEXT layer depend on each neuron?

    This follows SCAR/supernode logic where importance is determined by
    DOWNSTREAM dependencies, not upstream correlations.

    Theory:
        Importance(Y_i^l) = mean_j I(Y_i^l; Y_j^{l+1})

    High downstream importance = neuron is critical (like supernode)
    Low downstream importance = neuron may be safe to prune

    Combined with within-layer redundancy:
    - High importance + Low redundancy = CRITICAL (keep!)
    - Low importance + High redundancy = PRUNABLE (safe to remove)
    - High importance + High redundancy = HALO (redundant with critical neurons)
    """

    def __init__(
        self,
        max_refs: int = 512,
        compute_within: bool = True,
        **config: Any,
    ):
        """
        Initialize cross-layer importance metric.

        Args:
            max_refs: Maximum reference neurons for efficiency
            compute_within: Also compute within-layer redundancy for comparison
        """
        super().__init__(**config)
        self.max_refs = max_refs
        self.compute_within = compute_within
        self._next_layer_activations: Optional[torch.Tensor] = None

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True

    def set_next_layer_activations(self, activations: torch.Tensor) -> None:
        """
        Set activations from NEXT layer for downstream importance computation.

        Args:
            activations: [batch, num_next_neurons]
        """
        self._next_layer_activations = activations.detach()

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        next_layer_activations: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        layer_name: str = "",
        **kwargs: Any,
    ) -> Union[CrossLayerResult, torch.Tensor]:
        """
        Compute downstream importance and within-layer redundancy.

        Args:
            outputs: Current layer activations [batch, neurons]
            next_layer_activations: NEXT layer activations (or use set_next_layer_activations)
            layer_idx: Layer index
            layer_name: Layer name

        Returns:
            CrossLayerResult or just the importance tensor
        """
        if outputs is None:
            raise ValueError("CrossLayerRedundancy requires outputs")

        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        # Get next layer activations
        next_acts = next_layer_activations if next_layer_activations is not None else self._next_layer_activations

        if next_acts is None:
            logger.warning("No next layer activations available, returning zeros")
            return torch.zeros(outputs.shape[1], device=outputs.device)

        if next_acts.ndim > 2:
            next_acts = next_acts.reshape(next_acts.shape[0], -1)

        # Ensure same batch size
        min_batch = min(outputs.shape[0], next_acts.shape[0])
        outputs = outputs[:min_batch].float()
        next_acts = next_acts[:min_batch].float()

        # Compute downstream importance (how much next layer depends on each neuron)
        downstream_imp = compute_downstream_importance(outputs, next_acts, self.max_refs)

        # Optionally compute within-layer redundancy
        if self.compute_within:
            within_redundancy = compute_within_layer_redundancy(outputs, self.max_refs)

            mean_importance = float(downstream_imp.mean())
            mean_within = float(within_redundancy.mean())

            result = CrossLayerResult(
                layer_idx=layer_idx,
                layer_name=layer_name,
                downstream_importance=downstream_imp.cpu().numpy(),
                within_layer_redundancy=within_redundancy.cpu().numpy(),
                mean_downstream_importance=mean_importance,
                mean_within_layer_redundancy=mean_within,
                importance_to_redundancy_ratio=mean_importance / max(mean_within, 1e-8),
            )
            return result

        return downstream_imp


@register_metric("cross_layer_importance")
class CrossLayerImportance(BaseMetric):
    """
    Compute importance scores combining current-layer metrics with downstream importance.

    Following SCAR logic:
    - High RQ/MI (alignment with input) = good
    - High downstream importance (next layer depends on this) = good (like supernode!)
    - Low within-layer redundancy = good (unique information)

    Score = α·RQ + β·Downstream_Importance - γ·R_within

    Key insight: Downstream importance is a POSITIVE term (not penalty)
    because neurons that next layer depends on are important to keep.
    """

    def __init__(
        self,
        rq_weight: float = 0.25,
        downstream_weight: float = 0.35,  # Positive weight for downstream importance
        within_redundancy_weight: float = 0.25,  # Penalty for redundancy
        activation_weight: float = 0.15,  # Activation magnitude
        normalize: bool = True,
        max_refs: int = 512,
        **config: Any,
    ):
        """
        Initialize cross-layer importance metric.

        Args:
            rq_weight: Weight for Rayleigh Quotient (α)
            downstream_weight: Weight for downstream importance (β) - POSITIVE
            within_redundancy_weight: Penalty for within-layer redundancy (γ)
            activation_weight: Weight for activation magnitude
            normalize: Whether to normalize each component to [0, 1]
            max_refs: Maximum reference neurons for efficiency
        """
        super().__init__(**config)
        self.rq_weight = rq_weight
        self.downstream_weight = downstream_weight
        self.within_redundancy_weight = within_redundancy_weight
        self.activation_weight = activation_weight
        self.normalize = normalize
        self.max_refs = max_refs
        self._next_layer_activations: Optional[torch.Tensor] = None

    @property
    def requires_inputs(self) -> bool:
        return True  # For RQ computation

    @property
    def requires_weights(self) -> bool:
        return True  # For RQ computation

    @property
    def requires_outputs(self) -> bool:
        return True

    def set_next_layer_activations(self, activations: torch.Tensor) -> None:
        """Set activations from NEXT layer for downstream importance."""
        self._next_layer_activations = activations.detach()

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1]."""
        min_val, max_val = scores.min(), scores.max()
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return torch.zeros_like(scores)

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        next_layer_activations: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute cross-layer importance scores.

        Args:
            inputs: Layer inputs [batch, input_dim]
            weights: Layer weights [out_dim, in_dim]
            outputs: Layer outputs/activations [batch, out_dim]
            next_layer_activations: NEXT layer activations

        Returns:
            Importance scores [num_neurons]
        """
        if outputs is None or weights is None:
            raise ValueError("CrossLayerImportance requires outputs and weights")

        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        num_neurons = outputs.shape[1]
        device = outputs.device

        # Initialize composite score
        composite = torch.zeros(num_neurons, device=device)

        # 1. Compute RQ if inputs available (alignment with input covariance)
        if inputs is not None and self.rq_weight > 0:
            if inputs.ndim > 2:
                inputs = inputs.reshape(inputs.shape[0], -1)

            # Compute covariance
            inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
            cov = (inputs_centered.T @ inputs_centered) / max(1, inputs.shape[0] - 1)

            # RQ for each neuron: w^T Σ w / w^T w
            rq_scores = torch.zeros(num_neurons, device=device)
            for i in range(min(num_neurons, weights.shape[0])):
                w = weights[i]
                numerator = w @ cov @ w
                denominator = (w @ w) + 1e-8
                rq_scores[i] = numerator / denominator

            if self.normalize:
                rq_scores = self._normalize_scores(rq_scores)

            composite += self.rq_weight * rq_scores

        # 2. Compute downstream importance (POSITIVE - how much next layer needs this neuron)
        next_acts = next_layer_activations if next_layer_activations is not None else self._next_layer_activations

        if next_acts is not None and self.downstream_weight > 0:
            if next_acts.ndim > 2:
                next_acts = next_acts.reshape(next_acts.shape[0], -1)

            min_batch = min(outputs.shape[0], next_acts.shape[0])
            downstream_imp = compute_downstream_importance(
                outputs[:min_batch].float(),
                next_acts[:min_batch].float(),
                self.max_refs,
            )

            if self.normalize:
                downstream_imp = self._normalize_scores(downstream_imp)

            # POSITIVE contribution - more downstream importance = more important!
            composite += self.downstream_weight * downstream_imp

        # 3. Compute within-layer redundancy (PENALTY)
        if self.within_redundancy_weight > 0:
            within_red = compute_within_layer_redundancy(outputs.float(), self.max_refs)
            if self.normalize:
                within_red = self._normalize_scores(within_red)
            # NEGATIVE contribution - more redundancy = less unique = less important
            composite -= self.within_redundancy_weight * within_red

        # 4. Activation magnitude (like supernode identification)
        if self.activation_weight > 0:
            act_mag = outputs.abs().mean(dim=0)
            if self.normalize:
                act_mag = self._normalize_scores(act_mag)
            composite += self.activation_weight * act_mag

        return composite


def compute_activation_uniformity(activations: torch.Tensor) -> float:
    """
    Measure how uniform the activation magnitudes are across neurons.

    Returns a value in [0, 1]:
    - Close to 1 = uniform (no supernodes) -> cross-layer importance won't help
    - Close to 0 = non-uniform (supernodes exist) -> cross-layer importance valuable

    Uses coefficient of variation (CV) of activation magnitudes.
    """
    # Mean activation magnitude per neuron
    act_mag = activations.abs().mean(dim=0)

    mean_mag = act_mag.mean()
    std_mag = act_mag.std()

    if mean_mag < 1e-8:
        return 1.0  # All near zero = uniform

    cv = std_mag / mean_mag  # Coefficient of variation

    # High CV = non-uniform (supernodes), Low CV = uniform
    # Map to uniformity score: exp(-cv) gives ~1 for cv=0, ~0.37 for cv=1, etc.
    uniformity = float(torch.exp(-cv).item())

    return uniformity


@register_metric("adaptive_importance")
class AdaptiveImportance(BaseMetric):
    """
    Adaptive importance metric that switches strategy based on layer structure.

    Key insight: Cross-layer (downstream) importance only provides signal when
    there are non-uniform activations (supernodes). For uniform layers, it's
    better to use within-layer redundancy combined with activation magnitude.

    TWO REGIMES:

    1. Non-uniform (supernodes exist, uniformity < threshold):
       Score = α·Activation + β·Downstream_Importance - γ·Redundancy
       (Cross-layer importance helps identify what next layer needs)

    2. Uniform (no supernodes, uniformity >= threshold):
       Score = α·Activation - γ·Redundancy
       (Only within-layer info matters; downstream importance is flat)

    The metric automatically detects which regime applies.
    """

    def __init__(
        self,
        activation_weight: float = 0.4,
        downstream_weight: float = 0.35,
        redundancy_weight: float = 0.25,
        uniformity_threshold: float = 0.7,  # Above this = uniform layer
        normalize: bool = True,
        max_refs: int = 512,
        **config: Any,
    ):
        """
        Initialize adaptive importance metric.

        Args:
            activation_weight: Weight for activation magnitude
            downstream_weight: Weight for downstream importance (only used if non-uniform)
            redundancy_weight: Penalty for within-layer redundancy
            uniformity_threshold: Threshold for switching regimes (0-1)
            normalize: Whether to normalize components
            max_refs: Maximum reference neurons
        """
        super().__init__(**config)
        self.activation_weight = activation_weight
        self.downstream_weight = downstream_weight
        self.redundancy_weight = redundancy_weight
        self.uniformity_threshold = uniformity_threshold
        self.normalize = normalize
        self.max_refs = max_refs
        self._next_layer_activations: Optional[torch.Tensor] = None

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True

    def set_next_layer_activations(self, activations: torch.Tensor) -> None:
        """Set activations from NEXT layer."""
        self._next_layer_activations = activations.detach()

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1]."""
        min_val, max_val = scores.min(), scores.max()
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return torch.zeros_like(scores)

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        next_layer_activations: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute adaptive importance scores.

        Automatically detects layer uniformity and chooses appropriate strategy.
        """
        if outputs is None:
            raise ValueError("AdaptiveImportance requires outputs")

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        outputs.shape[1]

        # Step 1: Detect uniformity
        uniformity = compute_activation_uniformity(outputs)
        is_uniform = uniformity >= self.uniformity_threshold

        logger.debug(f"Layer uniformity: {uniformity:.3f} ({'uniform' if is_uniform else 'non-uniform'})")

        # Step 2: Compute activation magnitude (always used)
        act_mag = outputs.abs().mean(dim=0)
        if self.normalize:
            act_mag = self._normalize_scores(act_mag)

        # Step 3: Compute within-layer redundancy (always used)
        within_red = compute_within_layer_redundancy(outputs.float(), self.max_refs)
        if self.normalize:
            within_red = self._normalize_scores(within_red)

        # Step 4: Compute score based on regime
        if is_uniform:
            # UNIFORM REGIME: Cross-layer won't help
            # Score = Activation × (1 - redundancy)
            composite = self.activation_weight * act_mag - self.redundancy_weight * within_red

        else:
            # NON-UNIFORM REGIME: Cross-layer importance is valuable
            composite = self.activation_weight * act_mag - self.redundancy_weight * within_red

            # Add downstream importance if available
            next_acts = next_layer_activations if next_layer_activations is not None else self._next_layer_activations

            if next_acts is not None and self.downstream_weight > 0:
                if next_acts.ndim > 2:
                    next_acts = next_acts.reshape(next_acts.shape[0], -1)

                min_batch = min(outputs.shape[0], next_acts.shape[0])
                downstream_imp = compute_downstream_importance(
                    outputs[:min_batch].float(),
                    next_acts[:min_batch].float(),
                    self.max_refs,
                )

                if self.normalize:
                    downstream_imp = self._normalize_scores(downstream_imp)

                composite += self.downstream_weight * downstream_imp

        return composite


@register_metric("within_layer_importance")
class WithinLayerImportance(BaseMetric):
    """
    Importance score using ONLY within-layer information.

    For layers without clear supernode structure, or when cross-layer
    analysis is not available, this metric uses:

    Score = Activation × (1 - α·Redundancy)

    This says: "Keep neurons that are active AND unique within their layer"

    No cross-layer information needed.
    """

    def __init__(
        self,
        activation_weight: float = 0.6,
        redundancy_weight: float = 0.4,
        normalize: bool = True,
        max_refs: int = 512,
        **config: Any,
    ):
        super().__init__(**config)
        self.activation_weight = activation_weight
        self.redundancy_weight = redundancy_weight
        self.normalize = normalize
        self.max_refs = max_refs

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        min_val, max_val = scores.min(), scores.max()
        if max_val > min_val:
            return (scores - min_val) / (max_val - min_val)
        return torch.zeros_like(scores)

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute within-layer importance: activation × (1 - redundancy).
        """
        if outputs is None:
            raise ValueError("WithinLayerImportance requires outputs")

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        # Activation magnitude
        act_mag = outputs.abs().mean(dim=0)
        if self.normalize:
            act_mag = self._normalize_scores(act_mag)

        # Within-layer redundancy
        within_red = compute_within_layer_redundancy(outputs.float(), self.max_refs)
        if self.normalize:
            within_red = self._normalize_scores(within_red)

        # Importance = Activation × (1 - redundancy)
        # Or equivalently: activation_weight × act - redundancy_weight × red
        composite = self.activation_weight * act_mag - self.redundancy_weight * within_red

        return composite


@register_metric("layer_transition_efficiency")
class LayerTransitionEfficiency(BaseMetric):
    """
    Measure how critical each neuron is for downstream information flow.

    This metric captures: "If we remove this neuron, how much does
    the next layer lose?"

    Efficiency = Downstream_Importance / (Within_Layer_Redundancy + ε)

    High efficiency = neuron is critical AND unique
    Low efficiency = neuron is either not needed by next layer OR redundant

    This follows SCAR logic: importance is about downstream impact.
    """

    def __init__(
        self,
        max_refs: int = 512,
        **config: Any,
    ):
        super().__init__(**config)
        self.max_refs = max_refs
        self._next_layer_activations: Optional[torch.Tensor] = None

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return True

    def set_next_layer_activations(self, activations: torch.Tensor) -> None:
        """Set activations from NEXT layer."""
        self._next_layer_activations = activations.detach()

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        next_layer_activations: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute neuron efficiency: downstream importance / redundancy.

        Returns:
            Efficiency scores [num_neurons] where higher = more critical
        """
        if outputs is None:
            raise ValueError("LayerTransitionEfficiency requires outputs")

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        next_acts = next_layer_activations if next_layer_activations is not None else self._next_layer_activations

        if next_acts is None:
            logger.warning("No next layer activations, returning activation magnitude as proxy")
            return outputs.abs().mean(dim=0)

        if next_acts.ndim > 2:
            next_acts = next_acts.reshape(next_acts.shape[0], -1)

        min_batch = min(outputs.shape[0], next_acts.shape[0])

        # Compute downstream importance
        downstream_imp = compute_downstream_importance(
            outputs[:min_batch].float(),
            next_acts[:min_batch].float(),
            self.max_refs,
        )

        # Compute within-layer redundancy
        within_red = compute_within_layer_redundancy(outputs.float(), self.max_refs)

        # Efficiency = importance / (1 + redundancy)
        # Higher downstream importance = more efficient
        # Higher redundancy = less efficient (others carry same info)
        efficiency = downstream_imp / (1.0 + within_red)

        return efficiency
