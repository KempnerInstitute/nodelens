"""
Pruning module for the alignment framework.

This module provides:
- Various pruning strategies (magnitude, gradient, random, etc.)
- Pruning experiments for analysis
"""

from typing import Optional, Union, Type
import logging

from .base import BasePruningStrategy, IterativePruningStrategy, PruningConfig
from .strategies import (
    MagnitudePruning,
    IterativeMagnitudePruning,
    GlobalMagnitudePruning,
    GradientPruning,
    FisherPruning,
    MomentumPruning,
    RandomPruning,
    BernoulliPruning,
)
from .experiments import (
    ProgressiveDropoutExperiment,
    CascadingLayerPruningExperiment,
    LayerIsolatedPruningExperiment,
    EigenvectorDropoutExperiment,
)

logger = logging.getLogger(__name__)

# Registry of available pruning strategies
PRUNING_STRATEGIES = {
    # Magnitude-based strategies
    'magnitude': MagnitudePruning,
    'iterative_magnitude': IterativeMagnitudePruning,
    'global_magnitude': GlobalMagnitudePruning,
    
    # Gradient-based strategies
    'gradient': GradientPruning,
    'fisher': FisherPruning,
    'momentum': MomentumPruning,
    
    # Random strategies
    'random': RandomPruning,
    'bernoulli': BernoulliPruning,
}


def get_pruning_strategy(
    name: str,
    **kwargs
) -> BasePruningStrategy:
    """
    Get a pruning strategy by name.
    
    Args:
        name: Name of the pruning strategy
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Initialized pruning strategy
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in PRUNING_STRATEGIES:
        available = list(PRUNING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown pruning strategy: {name}. "
            f"Available strategies: {available}"
        )
    
    strategy_class = PRUNING_STRATEGIES[name]
    return strategy_class(**kwargs)


def list_pruning_strategies() -> list:
    """List all available pruning strategies."""
    return list(PRUNING_STRATEGIES.keys())


__all__ = [
    # Base classes
    'BasePruningStrategy',
    'IterativePruningStrategy',
    'PruningConfig',
    
    # Magnitude strategies
    'MagnitudePruning',
    'IterativeMagnitudePruning',
    'GlobalMagnitudePruning',
    
    # Gradient strategies
    'GradientPruning',
    'FisherPruning',
    'MomentumPruning',
    
    # Random strategies
    'RandomPruning',
    'BernoulliPruning',
    
    # Experiments
    'ProgressiveDropoutExperiment',
    'CascadingLayerPruningExperiment',
    'LayerIsolatedPruningExperiment',
    'EigenvectorDropoutExperiment',
    
    # Functions
    'get_pruning_strategy',
    'list_pruning_strategies',
] 