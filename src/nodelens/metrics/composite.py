"""
Composite metrics for combining multiple importance scores.

This module provides metrics that combine multiple individual metrics
into a unified importance score for pruning decisions.

The composite score follows the formula from the alignment notes:
    Score(i) = α * log RQ(w_i) + β * I(Z; Y_i) + γ * S(i) - δ * R(i)

Where:
    - RQ: Rayleigh Quotient (alignment)
    - I(Z; Y_i): Mutual information about class
    - S(i): Synergy contribution
    - R(i): Redundancy (subtracted - we want low redundancy)
"""

import logging
from typing import Any, Dict, Optional

import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("composite_importance", aliases=["composite", "combined_score"])
class CompositeImportance(BaseMetric):
    """
    Composite importance metric combining multiple individual metrics.

    Combines alignment, class information, synergy, and redundancy
    into a single importance score for pruning.

    Default weights are based on the alignment theory:
    - High alignment (RQ) = important -> positive weight
    - High class MI = informative -> positive weight
    - High synergy = unique contribution -> positive weight
    - High redundancy = replaceable -> negative weight (subtract)

    Example:
        >>> metric = CompositeImportance(
        ...     weights={
        ...         "rayleigh_quotient": 1.0,
        ...         "mi_about_class": 0.5,
        ...         "average_redundancy": -0.3,  # Negative = penalize redundant
        ...     }
        ... )
        >>> scores = metric.compute(inputs=X, weights=W, targets=y)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalize_components: bool = True,
        log_transform_rq: bool = True,
        **config: Any,
    ):
        """
        Initialize composite importance metric.

        Args:
            weights: Dict mapping metric names to weights.
                     Negative weights subtract (e.g., for redundancy).
                     Default: {"rayleigh_quotient": 1.0, "mi_about_class": 0.5, "average_redundancy": -0.3}
            normalize_components: Whether to normalize each component to [0, 1] before combining
            log_transform_rq: Whether to apply log transform to RQ (as suggested in theory)
            **config: Additional configuration
        """
        super().__init__(**config)

        self.weights = weights or {
            "rayleigh_quotient": 1.0,
            "mi_about_class": 0.5,
            "average_redundancy": -0.3,
        }
        self.normalize_components = normalize_components
        self.log_transform_rq = log_transform_rq

        # Initialize component metrics lazily
        self._component_metrics = {}

    def _get_metric(self, name: str) -> BaseMetric:
        """Get or create a component metric by name."""
        if name not in self._component_metrics:
            from . import get_metric

            self._component_metrics[name] = get_metric(name)
        return self._component_metrics[name]

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
        Compute composite importance scores.

        Args:
            inputs: Input activations [batch_size, features]
            weights: Layer weights [num_neurons, features]
            outputs: Layer outputs (optional)
            targets: Class labels (optional, for conditional metrics)
            **kwargs: Additional arguments passed to component metrics

        Returns:
            Composite importance scores [num_neurons]
        """
        if inputs is None or weights is None:
            raise ValueError("Composite metric requires inputs and weights")

        # Determine number of neurons
        if weights.ndim > 2:
            num_neurons = weights.shape[0]
        else:
            num_neurons = weights.shape[0]

        device = weights.device
        composite_scores = torch.zeros(num_neurons, device=device, dtype=weights.dtype)

        # Compute each component metric
        for metric_name, weight in self.weights.items():
            try:
                metric = self._get_metric(metric_name)

                # Compute the metric
                scores = metric.compute(inputs=inputs, weights=weights, outputs=outputs, targets=targets, **kwargs)

                # Ensure correct shape
                if scores.numel() != num_neurons:
                    logger.warning(f"Metric {metric_name} returned {scores.numel()} scores, expected {num_neurons}")
                    continue

                scores = scores.flatten()[:num_neurons]

                # Apply log transform for RQ if requested
                if self.log_transform_rq and "rayleigh" in metric_name.lower():
                    scores = torch.log(scores.clamp(min=1e-10))

                # Normalize to [0, 1] if requested
                if self.normalize_components:
                    s_min, s_max = scores.min(), scores.max()
                    if s_max > s_min:
                        scores = (scores - s_min) / (s_max - s_min)

                # Add weighted component
                composite_scores += weight * scores.to(device)

                logger.debug(f"Added {metric_name} with weight {weight}, mean={scores.mean():.4f}")

            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                continue

        # Clean up numerical issues
        composite_scores = torch.nan_to_num(composite_scores, nan=0.0, posinf=0.0, neginf=0.0)

        return composite_scores


@register_metric("alignment_minus_redundancy", aliases=["rq_minus_r"])
class AlignmentMinusRedundancy(BaseMetric):
    """
    Simple composite: RQ - αR

    Neurons with high alignment but low redundancy get high scores.
    This is a simplified version of the full composite metric.

    Example:
        >>> metric = AlignmentMinusRedundancy(redundancy_weight=0.5)
        >>> scores = metric.compute(inputs=X, weights=W)
    """

    def __init__(
        self,
        redundancy_weight: float = 0.3,
        normalize: bool = True,
        **config: Any,
    ):
        super().__init__(**config)
        self.redundancy_weight = redundancy_weight
        self.normalize = normalize
        self._rq_metric = None
        self._redundancy_metric = None

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return False

    def _init_metrics(self):
        """Initialize component metrics lazily."""
        if self._rq_metric is None:
            from . import get_metric

            self._rq_metric = get_metric("rayleigh_quotient")
            self._redundancy_metric = get_metric("average_redundancy")

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute RQ - α * Redundancy."""
        self._init_metrics()

        # Compute RQ
        rq_scores = self._rq_metric.compute(inputs=inputs, weights=weights, **kwargs)

        # Compute redundancy
        redundancy_scores = self._redundancy_metric.compute(inputs=inputs, weights=weights, **kwargs)

        # Normalize if requested
        if self.normalize:
            # Normalize RQ
            rq_min, rq_max = rq_scores.min(), rq_scores.max()
            if rq_max > rq_min:
                rq_scores = (rq_scores - rq_min) / (rq_max - rq_min)

            # Normalize redundancy
            r_min, r_max = redundancy_scores.min(), redundancy_scores.max()
            if r_max > r_min:
                redundancy_scores = (redundancy_scores - r_min) / (r_max - r_min)

        # Combine: high RQ good, high redundancy bad
        composite = rq_scores - self.redundancy_weight * redundancy_scores

        return torch.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0)


@register_metric("class_informative_score", aliases=["class_info"])
class ClassInformativeScore(BaseMetric):
    """
    Combined score focusing on class-relevant neurons.

    Combines:
    - MI about class (high = informative about class)
    - Conditional RQ (high = aligned with within-class variance)
    - Delta RQ (high = sensitive to class structure)

    Neurons that score high on all these are class-informative.
    """

    def __init__(
        self,
        mi_weight: float = 1.0,
        cond_rq_weight: float = 0.5,
        delta_rq_weight: float = 0.3,
        **config: Any,
    ):
        super().__init__(**config)
        self.mi_weight = mi_weight
        self.cond_rq_weight = cond_rq_weight
        self.delta_rq_weight = delta_rq_weight
        self._metrics = {}

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return False

    def _get_metric(self, name: str):
        if name not in self._metrics:
            from . import get_metric

            self._metrics[name] = get_metric(name)
        return self._metrics[name]

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute class-informative score."""
        if targets is None:
            logger.warning("ClassInformativeScore requires targets, falling back to RQ")
            rq = self._get_metric("rayleigh_quotient")
            return rq.compute(inputs=inputs, weights=weights, **kwargs)

        num_neurons = weights.shape[0]
        device = weights.device
        composite = torch.zeros(num_neurons, device=device, dtype=weights.dtype)

        # MI about class
        try:
            mi_metric = self._get_metric("mi_about_class")
            mi_scores = mi_metric.compute(inputs=inputs, weights=weights, targets=targets, **kwargs)
            mi_scores = self._normalize(mi_scores)
            composite += self.mi_weight * mi_scores
        except Exception as e:
            logger.warning(f"MI about class failed: {e}")

        # Conditional RQ
        try:
            cond_rq = self._get_metric("conditional_rayleigh_quotient")
            cond_scores = cond_rq.compute(inputs=inputs, weights=weights, targets=targets, **kwargs)
            cond_scores = self._normalize(cond_scores)
            composite += self.cond_rq_weight * cond_scores
        except Exception as e:
            logger.warning(f"Conditional RQ failed: {e}")

        # Delta RQ (if targets available, measures class sensitivity)
        try:
            delta_rq = self._get_metric("delta_rq")
            delta_scores = delta_rq.compute(inputs=inputs, weights=weights, targets=targets, **kwargs)
            delta_scores = self._normalize(delta_scores)
            composite += self.delta_rq_weight * delta_scores
        except Exception as e:
            logger.debug(f"Delta RQ failed (expected without targets): {e}")

        return torch.nan_to_num(composite, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1]."""
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        return scores
