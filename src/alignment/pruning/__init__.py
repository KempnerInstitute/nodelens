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
    LayerwiseRandomPruning,
    BernoulliPruning,
    ParallelModePruning,
    TensorizedPruning,
    AsyncParallelPruning,
    AlignmentPruning,
    HybridPruning,
    GlobalAlignmentPruning,
    CascadingAlignmentPruning,
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
    
    # Alignment-based strategies
    'alignment': AlignmentPruning,
    'hybrid': HybridPruning,
    'global_alignment': GlobalAlignmentPruning,
    'cascading_alignment': CascadingAlignmentPruning,
    
    # Random strategies (kept for backward compatibility)
    # Note: Consider using selection_mode='random' instead
    'random': RandomPruning,
    'bernoulli': BernoulliPruning,
    
    # Parallel strategies
    'parallel_mode': ParallelModePruning,
    'tensorized': TensorizedPruning,
    'async_parallel': AsyncParallelPruning,
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
    
    # Alignment strategies
    'AlignmentPruning',
    'HybridPruning',
    'GlobalAlignmentPruning',
    'CascadingAlignmentPruning',
    
    # Random strategies
    'RandomPruning',
    'LayerwiseRandomPruning',
    'BernoulliPruning',
    
    # Parallel strategies
    'ParallelModePruning',
    'TensorizedPruning',
    'AsyncParallelPruning',
    
    # Functions
    'get_pruning_strategy',
    'list_pruning_strategies',
] 