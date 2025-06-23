"""
Pruning module for neural network alignment analysis.

This module provides various pruning strategies and utilities for analyzing
the effect of pruning on network alignment.

Quick Start:
    >>> from alignment.pruning import MagnitudePruning, PruningConfig
    >>> 
    >>> # Simple magnitude pruning
    >>> strategy = MagnitudePruning()
    >>> mask = strategy.prune(layer, amount=0.5)
    >>> 
    >>> # Configured pruning
    >>> config = PruningConfig(amount=0.7, structured=True)
    >>> strategy = MagnitudePruning(config)
    >>> mask = strategy.prune(conv_layer)

Available Strategies:
    - MagnitudePruning: Prune based on weight magnitudes
    - GradientPruning: Prune based on gradient information
    - RandomPruning: Random pruning (baseline)
    - FisherPruning: Fisher information-based pruning
    - And more...

See the module README.md for detailed documentation.
"""

# Base classes
from .base import (
    BasePruningStrategy,
    IterativePruningStrategy,
    PruningConfig
)

# Magnitude-based strategies
from .strategies.magnitude import (
    MagnitudePruning,
    IterativeMagnitudePruning,
    GlobalMagnitudePruning
)

# Gradient-based strategies
from .strategies.gradient import (
    GradientPruning,
    FisherPruning,
    MomentumPruning
)

# Random strategies (baselines)
from .strategies.random import (
    RandomPruning,
    LayerwiseRandomPruning,
    BernoulliPruning
)

# Strategy registry for easy access
PRUNING_STRATEGIES = {
    'magnitude': MagnitudePruning,
    'iterative_magnitude': IterativeMagnitudePruning,
    'global_magnitude': GlobalMagnitudePruning,
    'gradient': GradientPruning,
    'fisher': FisherPruning,
    'momentum': MomentumPruning,
    'random': RandomPruning,
    'layerwise_random': LayerwiseRandomPruning,
    'bernoulli': BernoulliPruning,
}


def get_pruning_strategy(name: str, **kwargs) -> BasePruningStrategy:
    """
    Get a pruning strategy by name.
    
    Args:
        name: Name of the pruning strategy
        **kwargs: Arguments to pass to the strategy constructor
        
    Returns:
        Initialized pruning strategy
        
    Raises:
        ValueError: If strategy name is not recognized
        
    Examples:
        >>> strategy = get_pruning_strategy('magnitude', amount=0.5)
        >>> strategy = get_pruning_strategy('fisher')
    """
    if name not in PRUNING_STRATEGIES:
        available = ', '.join(PRUNING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown pruning strategy: {name}. "
            f"Available strategies: {available}"
        )
    
    strategy_class = PRUNING_STRATEGIES[name]
    return strategy_class(**kwargs)


def list_pruning_strategies() -> list:
    """
    List all available pruning strategies.
    
    Returns:
        List of strategy names
        
    Examples:
        >>> strategies = list_pruning_strategies()
        >>> print(strategies)
        ['magnitude', 'iterative_magnitude', 'global_magnitude', ...]
    """
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
    'LayerwiseRandomPruning',
    'BernoulliPruning',
    
    # Utility functions
    'get_pruning_strategy',
    'list_pruning_strategies',
    
    # Registry
    'PRUNING_STRATEGIES',
] 