"""
Node Scoring Service for composite importance calculation.

This service combines multiple metrics (RQ, MI, redundancy, synergy)
into composite neuron importance scores for pruning and analysis.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class CompositeScores:
    """Container for multi-metric scores."""

    rq: Optional[torch.Tensor] = None
    mi: Optional[torch.Tensor] = None
    redundancy: Optional[torch.Tensor] = None
    synergy: Optional[torch.Tensor] = None
    composite: Optional[torch.Tensor] = None
    layer_name: Optional[str] = None


class NodeScoringService:
    """
    Service for computing composite neuron importance scores.

    Combines multiple metrics:
    - Rayleigh Quotient (RQ): Alignment with input covariance
    - Mutual Information (MI): Information about target
    - Redundancy: Overlap with other neurons
    - Synergy: Complementary information with others

    Composite score:
        Score = α·MI + β·Synergy - γ·Redundancy + δ·log(RQ)

    Example:
        >>> scorer = NodeScoringService(metrics={'rq': rq_metric, 'mi': mi_metric})
        >>> scores = scorer.compute_composite_scores(
        ...     inputs=activations.inputs['conv1'],
        ...     weights=activations.weights['conv1'],
        ...     targets=labels
        ... )
        >>> print(scores.composite)  # Final importance scores
    """

    def __init__(
        self, metrics: Dict[str, Any], alpha_mi: float = 0.3, beta_synergy: float = 0.2, gamma_redundancy: float = 0.3, delta_rq: float = 0.2
    ):
        """
        Initialize node scoring service.

        Args:
            metrics: Dictionary of initialized metrics
                Expected keys: 'rq', 'mi', 'redundancy', 'synergy'
            alpha_mi: Weight for MI term
            beta_synergy: Weight for synergy term (positive)
            gamma_redundancy: Weight for redundancy term (negative)
            delta_rq: Weight for RQ term
        """
        self.metrics = metrics
        self.alpha_mi = alpha_mi
        self.beta_synergy = beta_synergy
        self.gamma_redundancy = gamma_redundancy
        self.delta_rq = delta_rq

        # Normalize weights
        total = alpha_mi + beta_synergy + gamma_redundancy + delta_rq
        if total > 0:
            self.alpha_mi /= total
            self.beta_synergy /= total
            self.gamma_redundancy /= total
            self.delta_rq /= total

    def compute_composite_scores(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        include_redundancy: bool = True,
        include_synergy: bool = True,
        layer_name: Optional[str] = None,
        **metric_kwargs,
    ) -> CompositeScores:
        """
        Compute composite importance scores for all neurons in a layer.

        Args:
            inputs: Input activations [batch, features]
            weights: Layer weights [num_neurons, features]
            targets: Target labels [batch] for MI/synergy
            outputs: Layer outputs [batch, num_neurons]
            include_redundancy: Whether to compute redundancy
            include_synergy: Whether to compute synergy (requires targets)
            layer_name: Optional layer name for logging
            **metric_kwargs: Additional arguments for metrics

        Returns:
            CompositeScores with individual and composite scores
        """
        num_neurons = weights.shape[0]
        device = weights.device

        scores = CompositeScores(layer_name=layer_name)

        # 1. Compute RQ (always available)
        if "rq" in self.metrics:
            try:
                scores.rq = self.metrics["rq"].compute(inputs=inputs, weights=weights, targets=targets, **metric_kwargs)
                logger.debug(f"RQ computed: shape={scores.rq.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute RQ: {e}")
                scores.rq = torch.zeros(num_neurons, device=device)

        # 2. Compute MI (if targets available)
        if "mi" in self.metrics and targets is not None:
            try:
                scores.mi = self.metrics["mi"].compute(inputs=inputs, weights=weights, targets=targets, outputs=outputs, **metric_kwargs)
                logger.debug(f"MI computed: shape={scores.mi.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute MI: {e}")
                scores.mi = torch.zeros(num_neurons, device=device)

        # 3. Compute Redundancy (if requested and metric available)
        if include_redundancy and "redundancy" in self.metrics:
            try:
                scores.redundancy = self.metrics["redundancy"].compute(inputs=inputs, weights=weights, **metric_kwargs)
                logger.debug(f"Redundancy computed: shape={scores.redundancy.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute redundancy: {e}")
                scores.redundancy = torch.zeros(num_neurons, device=device)

        # 4. Compute Synergy (if requested, targets available, and metric available)
        if include_synergy and targets is not None and "synergy" in self.metrics:
            try:
                scores.synergy = self.metrics["synergy"].compute(inputs=inputs, weights=weights, targets=targets, outputs=outputs, **metric_kwargs)
                logger.debug(f"Synergy computed: shape={scores.synergy.shape}")
            except Exception as e:
                logger.warning(f"Failed to compute synergy: {e}")
                scores.synergy = torch.zeros(num_neurons, device=device)

        # 5. Compute Composite Score
        scores.composite = self._compute_composite(scores, num_neurons, device)

        return scores

    def _compute_composite(self, scores: CompositeScores, num_neurons: int, device: torch.device) -> torch.Tensor:
        """
        Compute weighted combination of individual scores.

        Args:
            scores: Individual metric scores
            num_neurons: Number of neurons
            device: Target device

        Returns:
            Composite scores [num_neurons]
        """
        composite = torch.zeros(num_neurons, device=device)

        # Add MI component
        if scores.mi is not None and self.alpha_mi > 0:
            # Normalize MI to [0, 1] range for combining
            mi_normalized = self._normalize_scores(scores.mi)
            composite += self.alpha_mi * mi_normalized

        # Add Synergy component (positive contribution)
        if scores.synergy is not None and self.beta_synergy > 0:
            synergy_normalized = self._normalize_scores(scores.synergy)
            composite += self.beta_synergy * synergy_normalized

        # Subtract Redundancy component (negative contribution)
        if scores.redundancy is not None and self.gamma_redundancy > 0:
            redundancy_normalized = self._normalize_scores(scores.redundancy)
            composite -= self.gamma_redundancy * redundancy_normalized

        # Add RQ component (log scale)
        if scores.rq is not None and self.delta_rq > 0:
            # Use log(RQ + eps) to handle scale
            rq_log = torch.log(scores.rq + 1e-8)
            rq_log_normalized = self._normalize_scores(rq_log)
            composite += self.delta_rq * rq_log_normalized

        return composite

    @staticmethod
    def _normalize_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Raw scores
            eps: Small value for numerical stability

        Returns:
            Normalized scores
        """
        scores_min = scores.min()
        scores_max = scores.max()

        if (scores_max - scores_min) < eps:
            # All scores are the same
            return torch.ones_like(scores) * 0.5

        normalized = (scores - scores_min) / (scores_max - scores_min + eps)
        return normalized

    def compute_layerwise_scores(
        self,
        activation_data,  # ActivationData from capture service
        targets: Optional[torch.Tensor] = None,
        layers: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, CompositeScores]:
        """
        Compute composite scores for multiple layers.

        Args:
            activation_data: ActivationData with inputs/weights
            targets: Target labels
            layers: Layers to score (None = all)
            **kwargs: Additional arguments for scoring

        Returns:
            Dictionary mapping layer names to CompositeScores
        """
        layers = layers or activation_data.layer_names

        layerwise_scores = {}

        for layer in layers:
            if layer not in activation_data.inputs or layer not in activation_data.weights:
                logger.warning(f"Skipping layer {layer}: missing inputs or weights")
                continue

            try:
                scores = self.compute_composite_scores(
                    inputs=activation_data.inputs[layer],
                    weights=activation_data.weights[layer],
                    targets=targets,
                    outputs=activation_data.outputs.get(layer),
                    layer_name=layer,
                    **kwargs,
                )
                layerwise_scores[layer] = scores

                logger.info(f"Layer {layer}: composite score range " f"[{scores.composite.min():.4f}, {scores.composite.max():.4f}]")

            except Exception as e:
                logger.error(f"Failed to score layer {layer}: {e}")

        return layerwise_scores

    def rank_neurons_globally(
        self, layerwise_scores: Dict[str, CompositeScores], return_indices: bool = False
    ) -> Tuple[List[Tuple[str, int, float]], Optional[Dict]]:
        """
        Rank all neurons across all layers by composite score.

        Args:
            layerwise_scores: Scores for each layer
            return_indices: Whether to return dictionary of per-layer indices

        Returns:
            Tuple of (ranked_list, optional_indices_dict)
            ranked_list: List of (layer_name, neuron_idx, score) tuples, sorted
            indices_dict: Per-layer sorted indices (if requested)
        """
        all_neurons = []

        for layer_name, scores in layerwise_scores.items():
            if scores.composite is None:
                continue

            for neuron_idx, score in enumerate(scores.composite):
                all_neurons.append((layer_name, neuron_idx, score.item()))

        # Sort by score (descending)
        all_neurons.sort(key=lambda x: x[2], reverse=True)

        # Create per-layer indices if requested
        indices_dict = None
        if return_indices:
            indices_dict = {}
            for layer_name, scores in layerwise_scores.items():
                if scores.composite is not None:
                    sorted_indices = torch.argsort(scores.composite, descending=True)
                    indices_dict[layer_name] = sorted_indices

        return all_neurons, indices_dict


def create_scoring_service(metrics: Dict[str, Any], **weights) -> NodeScoringService:
    """
    Factory function for creating NodeScoringService.

    Args:
        metrics: Dictionary of initialized metrics
        **weights: Weights for composite score (alpha_mi, beta_synergy, etc.)

    Returns:
        Configured NodeScoringService
    """
    return NodeScoringService(metrics, **weights)
