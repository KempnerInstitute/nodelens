"""
Dynamic scoring using training evolution and loss correlation.

Analyzes how metric scores evolve during training and correlates them
with loss changes to identify truly important neurons.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

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
    
    def __init__(
        self,
        weight_final: float = 0.4,
        weight_trend: float = 0.2,
        weight_loss_corr: float = 0.3,
        weight_stability: float = 0.1
    ):
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
        self,
        score_history: Dict[str, Dict[str, List[float]]],
        loss_history: List[float],
        layer_name: str,
        metric_name: str = 'rq'
    ) -> torch.Tensor:
        """
        Aggregate scores for a layer using training dynamics.
        
        Args:
            score_history: From AlignmentMetricsCallback.get_history()
            loss_history: Training loss at each step
            layer_name: Layer to process
            metric_name: Which metric to aggregate
            
        Returns:
            Dynamic scores per neuron [num_neurons]
        """
        if layer_name not in score_history['history']:
            raise ValueError(f"No history for layer {layer_name}")
        
        if metric_name not in score_history['history'][layer_name]:
            raise ValueError(f"No {metric_name} history for {layer_name}")
        
        # Get score evolution
        scores_over_time = score_history['history'][layer_name][metric_name]
        # This is list of scalar means - need per-neuron history
        # For now, work with what we have
        
        # If we have per-neuron history (not yet implemented in callback):
        # scores_over_time = [step1_scores, step2_scores, ...] 
        # where each is [num_neurons]
        
        # For now, provide framework for when per-neuron tracking is added
        logger.warning(
            "Current callback tracks scalar means. "
            "For per-neuron dynamic scoring, need to track full tensors."
        )
        
        # Return placeholder
        return torch.tensor(scores_over_time[-1])  # Final value
    
    def compute_loss_correlation(
        self,
        score_evolution: torch.Tensor,  # [num_steps, num_neurons]
        loss_evolution: List[float]      # [num_steps]
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
    
    def compute_trend(
        self,
        score_evolution: torch.Tensor  # [num_steps, num_neurons]
    ) -> torch.Tensor:
        """
        Compute trend (increasing/decreasing) for each neuron.
        
        Positive trend: Importance increased → likely important
        Negative trend: Importance decreased → less critical
        
        Returns:
            Trend per neuron [num_neurons]
        """
        # Simple: final - initial
        trend = score_evolution[-1] - score_evolution[0]
        
        # More sophisticated: linear regression slope
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
    
    def compute_stability(
        self,
        score_evolution: torch.Tensor  # [num_steps, num_neurons]
    ) -> torch.Tensor:
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
    
    def aggregate_full(
        self,
        score_evolution: torch.Tensor,  # [num_steps, num_neurons]
        loss_evolution: List[float]
    ) -> torch.Tensor:
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
            self.weight_final * final_norm +
            self.weight_trend * trend_norm +
            self.weight_loss_corr * loss_corr_norm +
            self.weight_stability * stability_norm
        )
        
        return dynamic_scores


class TrainingAwareScoring:
    """
    Enhanced scoring using full training history.
    
    Requires per-neuron tracking during training (not just scalar means).
    """
    
    @staticmethod
    def enhance_callback_for_per_neuron_tracking():
        """
        Instructions for enhancing callback to track per-neuron evolution.
        
        Current callback tracks: scalar mean per step
        Enhanced version should track: full tensor per step (memory intensive!)
        
        Modification needed in AlignmentMetricsCallback:
        
        ```python
        # Instead of:
        score_value = scores.mean().item()
        self.history[layer][metric].append(score_value)
        
        # Do:
        if self.track_per_neuron:
            self.history[layer][metric].append(scores.cpu())  # Full tensor
        else:
            self.history[layer][metric].append(scores.mean().item())
        ```
        
        Then dynamic scoring becomes very powerful!
        """
        pass


def compute_dynamic_importance(
    score_history: Dict,
    loss_history: List[float],
    layer_name: str,
    metric_name: str = 'rq',
    aggregation_weights: Optional[Dict] = None
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

