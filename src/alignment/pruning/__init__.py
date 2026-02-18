"""
Pruning module for the alignment framework.

This module provides comprehensive pruning capabilities:

Strategies:
- Magnitude-based: MagnitudePruning, IterativeMagnitudePruning, GlobalMagnitudePruning
- Gradient-based: GradientPruning, FisherPruning, MomentumPruning
- Random: RandomPruning, LayerwiseRandomPruning, BernoulliPruning
- Parallel: ParallelModePruning, TensorizedPruning, AsyncParallelPruning

Pruning Modes:
- 'low': Prune weights with lowest importance scores (default)
- 'high': Prune weights with highest importance scores
- 'random': Prune weights randomly

Example:
    Basic pruning::

        from alignment.pruning import get_pruning_strategy, PruningConfig

        # Prune low-magnitude weights
        strategy = get_pruning_strategy('magnitude')
        mask = strategy.prune(layer, amount=0.5)

        # Prune high-magnitude weights
        config = PruningConfig(amount=0.5, pruning_mode='high')
        strategy = get_pruning_strategy('magnitude', config=config)

    Parallel pruning::

        from alignment.pruning.strategies import ParallelModePruning

        # Apply multiple modes simultaneously
        strategy = ParallelModePruning(modes=['low', 'high', 'random'])
        result = strategy.prune_parallel(layer, amount=0.5)

        # Access individual masks
        low_mask = result.masks['low']
        high_mask = result.masks['high']
"""

import logging
from typing import Optional, Type, Union

from .base import BasePruningStrategy, IterativePruningStrategy, PruningConfig
from .pipeline import PruningPipelineOptions, run_pruning_pipeline
from .strategies import (
    AdaptiveMovementPruning,
    AdaptiveSensitivityPruning,
    AlignmentPruning,
    AsyncParallelPruning,
    BernoulliPruning,
    CascadingAlignmentPruning,
    EigenvectorPruning,
    FisherPruning,
    FLAPPruning,
    GlobalAlignmentPruning,
    GlobalMagnitudePruning,
    GradientPruning,
    HybridPruning,
    IterativeMagnitudePruning,
    LayerSensitivity,
    LayerwiseRandomPruning,
    LLMPrunerChannelMode,
    MagnitudePruning,
    MomentumPruning,
    MovementPruning,
    OWLPruning,
    ParallelModePruning,
    RandomPruning,
    RIAPruning,
    SlimLLMPruning,
    SparseGPTPruning,
    TensorizedPruning,
    WandaPruning,
)

logger = logging.getLogger(__name__)

# Registry of available pruning strategies
PRUNING_STRATEGIES = {
    # Magnitude-based strategies
    "magnitude": MagnitudePruning,
    "iterative_magnitude": IterativeMagnitudePruning,
    "global_magnitude": GlobalMagnitudePruning,
    # Gradient-based strategies
    "gradient": GradientPruning,
    "fisher": FisherPruning,
    "momentum": MomentumPruning,
    # Alignment-based strategies
    "alignment": AlignmentPruning,
    "hybrid": HybridPruning,
    "global_alignment": GlobalAlignmentPruning,
    "cascading_alignment": CascadingAlignmentPruning,
    # Eigenvector-based (PCA pruning)
    "eigenvector": EigenvectorPruning,
    # Movement-based (Sanh et al. NeurIPS 2020)
    "movement": MovementPruning,
    "adaptive_movement": AdaptiveMovementPruning,
    # Adaptive sensitivity-based
    "adaptive_sensitivity": AdaptiveSensitivityPruning,
    # Random strategies (kept for backward compatibility)
    # Note: Consider using selection_mode='random' instead
    "random": RandomPruning,
    "bernoulli": BernoulliPruning,
    # Parallel strategies
    "parallel_mode": ParallelModePruning,
    "tensorized": TensorizedPruning,
    "async_parallel": AsyncParallelPruning,
    # LLM Baselines (Sun et al. 2023, Frantar & Alistarh 2023)
    "wanda": WandaPruning,
    "sparsegpt": SparseGPTPruning,
    # Additional LLM Baselines (OWL, LLM-Pruner, FLAP, RIA, SlimLLM)
    "owl": OWLPruning,
    "llm_pruner": LLMPrunerChannelMode,
    "flap": FLAPPruning,
    "ria": RIAPruning,
    "slimllm": SlimLLMPruning,
}


def get_pruning_strategy(name: str, **kwargs) -> BasePruningStrategy:
    """
    Get a pruning strategy by name.

    Args:
        name: Name of the pruning strategy. Can be:
            - A registered strategy name (e.g., "magnitude", "alignment")
            - A metric name (e.g., "rayleigh_quotient", "activation_l2_norm")
              which will use AlignmentPruning with that metric
        **kwargs: Additional arguments for the strategy

    Returns:
        Initialized pruning strategy

    Raises:
        ValueError: If strategy name is not found
    """
    # Known alignment metrics that should use AlignmentPruning
    ALIGNMENT_METRICS = {
        "rayleigh_quotient",
        "activation_l2_norm",
        "mutual_information_gaussian",
        "pairwise_redundancy_gaussian",
        "synergy_gaussian_mmi",
        "activation_variance",
        "activation_mean",
    }

    if name in PRUNING_STRATEGIES:
        strategy_class = PRUNING_STRATEGIES[name]
        return strategy_class(**kwargs)
    elif name in ALIGNMENT_METRICS:
        # Use AlignmentPruning with the specified metric
        return AlignmentPruning(metric=name, **kwargs)
    else:
        available = list(PRUNING_STRATEGIES.keys()) + list(ALIGNMENT_METRICS)
        raise ValueError(f"Unknown pruning strategy: {name}. " f"Available strategies: {available}")


def list_pruning_strategies() -> list:
    """List all available pruning strategies."""
    return list(PRUNING_STRATEGIES.keys())


__all__ = [
    # Base classes
    "BasePruningStrategy",
    "IterativePruningStrategy",
    "PruningConfig",
    # Magnitude strategies
    "MagnitudePruning",
    "IterativeMagnitudePruning",
    "GlobalMagnitudePruning",
    # Gradient strategies
    "GradientPruning",
    "FisherPruning",
    "MomentumPruning",
    # Alignment strategies
    "AlignmentPruning",
    "HybridPruning",
    "GlobalAlignmentPruning",
    "CascadingAlignmentPruning",
    # Eigenvector (PCA) strategy
    "EigenvectorPruning",
    # Movement-based (Sanh et al. 2020)
    "MovementPruning",
    "AdaptiveMovementPruning",
    # Adaptive sensitivity-based
    "AdaptiveSensitivityPruning",
    "LayerSensitivity",
    # Random strategies
    "RandomPruning",
    "LayerwiseRandomPruning",
    "BernoulliPruning",
    # Parallel strategies
    "ParallelModePruning",
    "TensorizedPruning",
    "AsyncParallelPruning",
    # LLM Baselines
    "WandaPruning",
    "SparseGPTPruning",
    # Functions
    "get_pruning_strategy",
    "list_pruning_strategies",
    # Pipeline helpers
    "PruningPipelineOptions",
    "run_pruning_pipeline",
]
