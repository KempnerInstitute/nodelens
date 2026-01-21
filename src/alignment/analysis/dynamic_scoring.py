"""
Dynamic scoring using training evolution and loss correlation.

Analyzes how metric scores evolve during training and correlates them
with loss changes to identify truly important neurons.
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class DynamicScoreAggregator:
    """
    Aggregate scores from training history with loss correlation.

    Combines:
    - Final scores (current state)
    - Score evolution (trend over training)
    - Loss correlation (impact on optimization)
    - Stability (consistency)

    Example:
        >>> aggregator = DynamicScoreAggregator()
        >>> dynamic_scores = aggregator.aggregate(
        ...     score_history=callback.get_history(),
        ...     loss_history=training_losses
        ... )
    """

    def __init__(self, weight_final: float = 0.4, weight_trend: float = 0.2, weight_loss_corr: float = 0.3, weight_stability: float = 0.1):
        """
        Initialize dynamic score aggregator.

        Args:
            weight_final: Weight for final score value
            weight_trend: Weight for trend (increase/decrease)
            weight_loss_corr: Weight for loss correlation
            weight_stability: Weight for stability (low variance)
        """
        self.weight_final = weight_final
        self.weight_trend = weight_trend
        self.weight_loss_corr = weight_loss_corr
        self.weight_stability = weight_stability

        # Normalize weights
        total = sum([weight_final, weight_trend, weight_loss_corr, weight_stability])
        self.weight_final /= total
        self.weight_trend /= total
        self.weight_loss_corr /= total
        self.weight_stability /= total

    def aggregate(
        self, score_history: Dict[str, Dict[str, List[float]]], loss_history: List[float], layer_name: str, metric_name: str = "rq"
    ) -> torch.Tensor:
        """
        Aggregate scores for a layer using training dynamics.

        Args:
            score_history: From AlignmentMetricsCallback.get_history()
            loss_history: Training loss at each step
            layer_name: Layer to process
            metric_name: Which metric to aggregate

        Returns:
            If per-neuron score history is provided, returns dynamic scores per neuron
            with shape \([num_neurons]\). If only scalar summaries are available in
            the callback history, falls back to returning the final scalar value as
            a 0-d tensor.
        """
        if layer_name not in score_history["history"]:
            raise ValueError(f"No history for layer {layer_name}")

        if metric_name not in score_history["history"][layer_name]:
            raise ValueError(f"No {metric_name} history for {layer_name}")

        # Prefer per-neuron tensor history when available (AlignmentMetricsCallback(track_per_neuron=True)).
        tensor_hist = score_history.get("tensor_history", {})
        if layer_name in tensor_hist and metric_name in tensor_hist[layer_name] and len(tensor_hist[layer_name][metric_name]) > 0:
            tensors = tensor_hist[layer_name][metric_name]
            score_evolution = torch.stack([t.detach().float().cpu().flatten() for t in tensors], dim=0)  # [T, N]
            loss_evolution = self._align_loss_history(score_history, loss_history, num_steps=score_evolution.shape[0])
            return self.aggregate_full(score_evolution, loss_evolution)

        # Fallback: scalar summaries only.
        scores_over_time = score_history["history"][layer_name][metric_name]
        logger.warning(
            "Dynamic scoring requires per-neuron score history (enable AlignmentMetricsCallback(track_per_neuron=True)); "
            "falling back to returning the final scalar metric summary."
        )
        return torch.tensor(scores_over_time[-1])  # final scalar value

    @staticmethod
    def _align_loss_history(score_history: Dict, loss_history: List[float], num_steps: int) -> List[float]:
        """Align a full loss curve to the callback's sampled metric steps."""
        if num_steps <= 0:
            return []

        # Ideal case: already aligned.
        if len(loss_history) == num_steps:
            return list(loss_history)

        steps = score_history.get("steps")
        if isinstance(steps, list) and len(steps) == num_steps and len(loss_history) > 0:
            max_step = max(steps) if steps else -1
            if max_step >= 0 and len(loss_history) > max_step:
                return [float(loss_history[s]) for s in steps]

        # Fallback: sample loss_history uniformly to match num_steps.
        if len(loss_history) == 0:
            return [0.0] * num_steps
        idxs = torch.linspace(0, len(loss_history) - 1, steps=num_steps).round().to(torch.long).tolist()
        return [float(loss_history[i]) for i in idxs]

    def compute_loss_correlation(
        self, score_evolution: torch.Tensor, loss_evolution: List[float]  # [num_steps, num_neurons]  # [num_steps]
    ) -> torch.Tensor:
        """
        Compute correlation between each neuron's score and training loss.

        High positive correlation: Neuron's importance grew as loss decreased
        → Neuron is important for learning

        Negative/low correlation: Neuron's importance didn't track loss
        → Neuron might be less critical

        Args:
            score_evolution: Score over time per neuron
            loss_evolution: Loss over time

        Returns:
            Correlation per neuron [num_neurons]
        """
        num_steps, num_neurons = score_evolution.shape

        # Convert loss to tensor
        loss_tensor = torch.tensor(loss_evolution, dtype=torch.float32)

        # Compute correlation for each neuron
        correlations = torch.zeros(num_neurons)

        for neuron_idx in range(num_neurons):
            neuron_scores = score_evolution[:, neuron_idx]

            # Pearson correlation
            corr = torch.corrcoef(torch.stack([neuron_scores, loss_tensor]))[0, 1]

            correlations[neuron_idx] = corr.abs()  # Use absolute value

        return correlations

    def compute_trend(self, score_evolution: torch.Tensor) -> torch.Tensor:  # [num_steps, num_neurons]
        """
        Compute trend (increasing/decreasing) for each neuron.

        Positive trend: Importance increased → likely important
        Negative trend: Importance decreased → less critical

        Returns:
            Trend per neuron [num_neurons]
        """
        # Linear regression slope over time for each neuron: y = a + b*t
        num_steps, num_neurons = score_evolution.shape
        time_steps = torch.arange(num_steps, dtype=torch.float32)

        trends_fitted = torch.zeros(num_neurons)

        for neuron_idx in range(num_neurons):
            y = score_evolution[:, neuron_idx]

            # Fit line: y = a + b*t
            # b = cov(t, y) / var(t)
            mean_t = time_steps.mean()
            mean_y = y.mean()

            cov_ty = ((time_steps - mean_t) * (y - mean_y)).sum()
            var_t = ((time_steps - mean_t) ** 2).sum()

            slope = cov_ty / (var_t + 1e-8)
            trends_fitted[neuron_idx] = slope

        return trends_fitted

    def compute_stability(self, score_evolution: torch.Tensor) -> torch.Tensor:  # [num_steps, num_neurons]
        """
        Compute stability (inverse variance) for each neuron.

        Low variance: Consistently important → reliable signal
        High variance: Fluctuating → less reliable

        Returns:
            Stability per neuron [num_neurons]
        """
        # Variance over time
        variance = score_evolution.var(dim=0)  # [num_neurons]

        # Stability = inverse variance
        stability = 1.0 / (variance + 1e-4)

        # Normalize to [0, 1]
        stability = (stability - stability.min()) / (stability.max() - stability.min() + 1e-8)

        return stability

    def aggregate_full(self, score_evolution: torch.Tensor, loss_evolution: List[float]) -> torch.Tensor:  # [num_steps, num_neurons]
        """
        Full aggregation using all components.

        Args:
            score_evolution: Score history per neuron
            loss_evolution: Loss history

        Returns:
            Aggregated dynamic scores [num_neurons]
        """
        # Compute components
        final_scores = score_evolution[-1]  # [num_neurons]
        trends = self.compute_trend(score_evolution)
        loss_corr = self.compute_loss_correlation(score_evolution, loss_evolution)
        stability = self.compute_stability(score_evolution)

        # Normalize each component to [0, 1]
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        final_norm = normalize(final_scores)
        trend_norm = normalize(trends + trends.abs().max())  # Make positive
        loss_corr_norm = normalize(loss_corr)
        stability_norm = stability  # Already normalized

        # Weighted combination
        dynamic_scores = (
            self.weight_final * final_norm
            + self.weight_trend * trend_norm
            + self.weight_loss_corr * loss_corr_norm
            + self.weight_stability * stability_norm
        )

        return dynamic_scores


def compute_dynamic_importance(
    score_history: Dict, loss_history: List[float], layer_name: str, metric_name: str = "rq", aggregation_weights: Optional[Dict] = None
) -> torch.Tensor:
    """
    Convenience function for dynamic importance computation.

    Args:
        score_history: Training history from callback
        loss_history: Loss values during training
        layer_name: Layer to process
        metric_name: Metric to use
        aggregation_weights: Optional custom weights

    Returns:
        Dynamic importance scores
    """
    aggregator = DynamicScoreAggregator(**(aggregation_weights or {}))

    return aggregator.aggregate(score_history, loss_history, layer_name, metric_name)
