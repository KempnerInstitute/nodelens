"""
Pruning distribution strategies - how to allocate sparsity across layers.

Given a target overall sparsity (e.g., 70%), these strategies determine
how much to prune from each individual layer.
"""

import logging
from enum import Enum
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Available distribution strategies."""

    UNIFORM = "uniform"  # Same % per layer
    GLOBAL_THRESHOLD = "global_threshold"  # Global score threshold
    ADAPTIVE_SENSITIVITY = "adaptive_sensitivity"  # Based on layer sensitivity
    SIZE_PROPORTIONAL = "size_proportional"  # Based on layer size
    IMPORTANCE_WEIGHTED = "importance_weighted"  # Based on avg scores
    CASCADING = "cascading"  # Sequential until target
    HYBRID = "hybrid"  # Combination of multiple


class PruningDistributionManager:
    """
    Manages distribution of pruning across layers.

    Determines per-layer pruning amounts to achieve target overall sparsity
    while considering layer importance, size, sensitivity, etc.

    Example:
        >>> manager = PruningDistributionManager(
        ...     strategy='adaptive_sensitivity',
        ...     target_sparsity=0.7
        ... )
        >>> amounts = manager.compute_distribution(
        ...     model, layer_names, eval_fn
        ... )
        >>> # amounts = {'conv1': 0.85, 'conv2': 0.75, 'fc1': 0.50, ...}
    """

    def __init__(self, strategy: str = "uniform", target_sparsity: float = 0.5, min_amount: float = 0.0, max_amount: float = 0.95, **kwargs):
        """
        Initialize distribution manager.

        Args:
            strategy: Distribution strategy name
            target_sparsity: Target overall sparsity (0-1)
            min_amount: Minimum pruning per layer
            max_amount: Maximum pruning per layer
            **kwargs: Strategy-specific parameters
        """
        self.strategy = DistributionStrategy(strategy)
        self.target_sparsity = target_sparsity
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.kwargs = kwargs

    def compute_distribution(
        self, model: nn.Module, layer_names: List[str], layer_scores: Optional[Dict[str, torch.Tensor]] = None, eval_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Compute per-layer pruning amounts.

        Args:
            model: Model to analyze
            layer_names: Layers to prune
            layer_scores: Pre-computed importance scores per layer
            eval_fn: Evaluation function for sensitivity measurement

        Returns:
            Dict mapping layer names to pruning amounts
        """
        if self.strategy == DistributionStrategy.UNIFORM:
            return self._uniform_distribution(layer_names)

        elif self.strategy == DistributionStrategy.GLOBAL_THRESHOLD:
            if layer_scores is None:
                raise ValueError("Global threshold requires layer_scores")
            return self._global_threshold_distribution(layer_scores, model)

        elif self.strategy == DistributionStrategy.ADAPTIVE_SENSITIVITY:
            if eval_fn is None:
                raise ValueError("Adaptive sensitivity requires eval_fn")
            return self._adaptive_sensitivity_distribution(model, layer_names, eval_fn)

        elif self.strategy == DistributionStrategy.SIZE_PROPORTIONAL:
            return self._size_proportional_distribution(model, layer_names)

        elif self.strategy == DistributionStrategy.IMPORTANCE_WEIGHTED:
            if layer_scores is None:
                raise ValueError("Importance weighted requires layer_scores")
            return self._importance_weighted_distribution(layer_scores, model)

        elif self.strategy == DistributionStrategy.CASCADING:
            return self._cascading_distribution(layer_scores, model, layer_names)

        elif self.strategy == DistributionStrategy.HYBRID:
            return self._hybrid_distribution(model, layer_names, layer_scores, eval_fn)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _uniform_distribution(self, layer_names: List[str]) -> Dict[str, float]:
        """Same amount for all layers."""
        return {name: self.target_sparsity for name in layer_names}

    def _global_threshold_distribution(self, layer_scores: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        """
        Compute amounts based on global score threshold.

        Finds threshold such that overall sparsity = target.
        """
        # Collect all scores
        all_scores = []
        layer_sizes = {}

        for layer_name, scores in layer_scores.items():
            all_scores.append(scores.flatten())
            layer_sizes[layer_name] = len(scores)

        all_scores_cat = torch.cat(all_scores)

        # Find threshold for target sparsity
        k = int(len(all_scores_cat) * (1 - self.target_sparsity))  # Keep k elements
        threshold = torch.topk(all_scores_cat, k).values[-1]

        # Compute implied amount per layer
        amounts = {}
        for layer_name, scores in layer_scores.items():
            # Fraction below threshold in this layer
            below_threshold = (scores < threshold).float().mean().item()
            amounts[layer_name] = max(self.min_amount, min(self.max_amount, below_threshold))

        return amounts

    def _adaptive_sensitivity_distribution(self, model: nn.Module, layer_names: List[str], eval_fn: Callable) -> Dict[str, float]:
        """
        Adaptive distribution based on layer sensitivity.

        Uses AdaptiveSensitivityPruning logic.
        """
        from .strategies.adaptive import AdaptiveSensitivityPruning

        pruner = AdaptiveSensitivityPruning(target_sparsity=self.target_sparsity)
        sensitivities = pruner.compute_all_sensitivities(model, layer_names, eval_fn)

        return {name: sens.recommended_amount for name, sens in sensitivities.items()}

    def _size_proportional_distribution(self, model: nn.Module, layer_names: List[str]) -> Dict[str, float]:
        """
        Distribute based on layer size.

        Larger layers pruned more (have more params to spare).
        """
        # Get layer sizes
        layer_sizes = {}
        total_size = 0

        for name in layer_names:
            layer = dict(model.named_modules())[name]
            size = layer.weight.numel()
            layer_sizes[name] = size
            total_size += size

        # Compute fractions
        layer_fractions = {name: size / total_size for name, size in layer_sizes.items()}

        # Adjust amounts based on size
        # Larger fraction → prune more
        amounts = {}
        for name, fraction in layer_fractions.items():
            # Scale around target
            adjustment = (fraction - 0.5) * 0.4  # ±0.2 around target
            amount = self.target_sparsity + adjustment
            amounts[name] = max(self.min_amount, min(self.max_amount, amount))

        # Normalize to hit target
        amounts = self._normalize_to_target(amounts, layer_sizes)

        return amounts

    def _importance_weighted_distribution(self, layer_scores: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
        """
        Distribute based on average layer importance.

        Low average importance → prune more.
        """
        # Compute average importance per layer
        layer_importance = {name: scores.mean().item() for name, scores in layer_scores.items()}

        # Normalize importances
        max_importance = max(layer_importance.values())
        min_importance = min(layer_importance.values())
        importance_range = max_importance - min_importance

        if importance_range < 1e-6:
            # All same importance → uniform
            return self._uniform_distribution(list(layer_scores.keys()))

        # Compute amounts (inverse to importance)
        amounts = {}
        for name, importance in layer_importance.items():
            # Normalize to [0, 1]
            norm_importance = (importance - min_importance) / importance_range

            # Inverse: high importance → low amount
            amount = self.target_sparsity + 0.3 * (1 - norm_importance) - 0.15
            amounts[name] = max(self.min_amount, min(self.max_amount, amount))

        # Normalize to hit target
        layer_sizes = {name: dict(model.named_modules())[name].weight.numel() for name in layer_scores.keys()}
        amounts = self._normalize_to_target(amounts, layer_sizes)

        return amounts

    def _cascading_distribution(self, layer_scores: Dict[str, torch.Tensor], model: nn.Module, layer_names: List[str]) -> Dict[str, float]:
        """
        Sequential pruning until target hit.

        Order determined by importance.
        """
        # Rank layers by average importance (lowest first)
        if layer_scores:
            layer_importance = {name: scores.mean().item() for name, scores in layer_scores.items()}
            sorted_layers = sorted(layer_importance.keys(), key=lambda x: layer_importance[x])
        else:
            # Default order
            sorted_layers = layer_names

        # Compute total params and target
        layer_sizes = {name: dict(model.named_modules())[name].weight.numel() for name in layer_names}
        total_params = sum(layer_sizes.values())
        target_to_remove = int(total_params * self.target_sparsity)

        # Distribute sequentially
        amounts = {}
        removed_so_far = 0

        for layer_name in sorted_layers:
            remaining_to_remove = target_to_remove - removed_so_far
            layer_size = layer_sizes[layer_name]

            if remaining_to_remove <= 0:
                amounts[layer_name] = 0.0
            else:
                # Remove up to max_amount from this layer
                amount = min(remaining_to_remove / layer_size, self.max_amount)
                amounts[layer_name] = amount
                removed_so_far += int(layer_size * amount)

        return amounts

    def _hybrid_distribution(
        self, model: nn.Module, layer_names: List[str], layer_scores: Optional[Dict], eval_fn: Optional[Callable]
    ) -> Dict[str, float]:
        """
        Hybrid: Combine multiple distribution strategies.

        Weighted average of uniform, global, and adaptive.
        """
        # Compute multiple distributions
        uniform = self._uniform_distribution(layer_names)

        # Try to compute others if data available
        if layer_scores:
            global_dist = self._global_threshold_distribution(layer_scores, model)
            importance = self._importance_weighted_distribution(layer_scores, model)
        else:
            global_dist = uniform
            importance = uniform

        if eval_fn:
            try:
                adaptive = self._adaptive_sensitivity_distribution(model, layer_names, eval_fn)
            except Exception:
                adaptive = uniform
        else:
            adaptive = uniform

        # Weighted combination
        weights = {"uniform": 0.2, "global": 0.3, "importance": 0.2, "adaptive": 0.3}

        amounts = {}
        for layer_name in layer_names:
            amount = (
                weights["uniform"] * uniform[layer_name]
                + weights["global"] * global_dist[layer_name]
                + weights["importance"] * importance[layer_name]
                + weights["adaptive"] * adaptive[layer_name]
            )
            amounts[layer_name] = max(self.min_amount, min(self.max_amount, amount))

        return amounts

    def _normalize_to_target(self, amounts: Dict[str, float], layer_sizes: Dict[str, int]) -> Dict[str, float]:
        """
        Normalize amounts to exactly hit target overall sparsity.

        Args:
            amounts: Initial per-layer amounts
            layer_sizes: Parameters per layer

        Returns:
            Normalized amounts
        """
        # Compute current total
        current_total = sum(amounts[name] * layer_sizes[name] for name in amounts)
        total_params = sum(layer_sizes.values())
        target_total = self.target_sparsity * total_params

        if current_total < 1e-6:
            return amounts

        # Scale factor
        scale = target_total / current_total

        # Apply scaling and clip
        normalized = {}
        for name in amounts:
            scaled_amount = amounts[name] * scale
            normalized[name] = max(self.min_amount, min(self.max_amount, scaled_amount))

        return normalized

    def print_distribution(self, amounts: Dict[str, float], model: nn.Module, layer_scores: Optional[Dict] = None):
        """Print human-readable distribution plan."""
        print("\n" + "=" * 80)
        print(f"Pruning Distribution ({self.strategy.value})")
        print(f"Target Overall Sparsity: {self.target_sparsity:.1%}")
        print("=" * 80)

        # Compute statistics
        layer_info = []
        total_params = 0
        total_pruned = 0

        for layer_name in sorted(amounts.keys()):
            layer = dict(model.named_modules())[layer_name]
            size = layer.weight.numel()
            amount = amounts[layer_name]
            pruned = int(size * amount)

            total_params += size
            total_pruned += pruned

            avg_score = layer_scores[layer_name].mean().item() if layer_scores and layer_name in layer_scores else None

            layer_info.append((layer_name, size, amount, pruned, avg_score))

        # Print table
        print(f"\n{'Layer':<25} {'Size':<10} {'Amount':<8} {'Pruned':<10} {'Avg Score':<10}")
        print("-" * 80)

        for name, size, amount, pruned, avg_score in layer_info:
            score_str = f"{avg_score:.4f}" if avg_score is not None else "N/A"
            print(f"{name:<25} {size:<10} {amount:>7.1%} {pruned:<10} {score_str:<10}")

        # Summary
        overall = total_pruned / total_params if total_params > 0 else 0

        print("-" * 80)
        print(f"{'TOTAL':<25} {total_params:<10} {overall:>7.1%} {total_pruned:<10}")
        print("=" * 80)

        if abs(overall - self.target_sparsity) > 0.01:
            print(f"Warning: Achieved {overall:.1%} vs target {self.target_sparsity:.1%}")
        else:
            print(f"Target sparsity achieved: {overall:.1%}")

        print()


def compute_pruning_distribution(
    model: nn.Module, layer_names: List[str], strategy: str = "adaptive_sensitivity", target_sparsity: float = 0.5, **kwargs
) -> Dict[str, float]:
    """
    Convenience function for computing pruning distribution.

    Args:
        model: Model to analyze
        layer_names: Layers to prune
        strategy: Distribution strategy
        target_sparsity: Target overall sparsity
        **kwargs: Strategy-specific parameters (e.g., eval_fn, layer_scores)

    Returns:
        Per-layer pruning amounts
    """
    manager = PruningDistributionManager(strategy=strategy, target_sparsity=target_sparsity)

    return manager.compute_distribution(model, layer_names, **kwargs)
