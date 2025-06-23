"""
Pruning module for the alignment framework.

This module provides:
- Various pruning strategies (magnitude, gradient, random, etc.)
- Structured pruning utilities
- Pruning experiments for analysis
"""

from typing import Optional, Union, Type
import logging

from .base import BasePruningStrategy
from .strategies import (
    MagnitudePruning,
    GradientPruning,
    RandomPruning,
    FisherPruning,
    L1Pruning,
    L2Pruning,
    TaylorPruning,
    HessianPruning,
    ActivationPruning,
    ConnectionSensitivityPruning
)
from .structured import (
    StructuredPruningStrategy,
    ChannelPruning,
    FilterPruning,
    BlockPruning,
    PatternPruning,
    NMPruning
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
    # Unstructured strategies
    'magnitude': MagnitudePruning,
    'gradient': GradientPruning,
    'random': RandomPruning,
    'fisher': FisherPruning,
    'l1': L1Pruning,
    'l2': L2Pruning,
    'taylor': TaylorPruning,
    'hessian': HessianPruning,
    'activation': ActivationPruning,
    'connection_sensitivity': ConnectionSensitivityPruning,
    
    # Structured strategies
    'channel': ChannelPruning,
    'filter': FilterPruning,
    'block': BlockPruning,
    'pattern': PatternPruning,
    'nm': NMPruning,
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
    # Base class
    'BasePruningStrategy',
    
    # Unstructured strategies
    'MagnitudePruning',
    'GradientPruning',
    'RandomPruning',
    'FisherPruning',
    'L1Pruning',
    'L2Pruning',
    'TaylorPruning',
    'HessianPruning',
    'ActivationPruning',
    'ConnectionSensitivityPruning',
    
    # Structured strategies
    'StructuredPruningStrategy',
    'ChannelPruning',
    'FilterPruning',
    'BlockPruning',
    'PatternPruning',
    'NMPruning',
    
    # Experiments
    'ProgressiveDropoutExperiment',
    'CascadingLayerPruningExperiment',
    'LayerIsolatedPruningExperiment',
    'EigenvectorDropoutExperiment',
    
    # Functions
    'get_pruning_strategy',
    'list_pruning_strategies',
] 