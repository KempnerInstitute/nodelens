"""Pruning strategies submodule."""

from .magnitude import (
    MagnitudePruning,
    IterativeMagnitudePruning,
    GlobalMagnitudePruning
)

from .gradient import (
    GradientPruning,
    FisherPruning,
    MomentumPruning
)

from .random import (
    RandomPruning,
    LayerwiseRandomPruning,
    BernoulliPruning
)

__all__ = [
    'MagnitudePruning',
    'IterativeMagnitudePruning',
    'GlobalMagnitudePruning',
    'GradientPruning',
    'FisherPruning',
    'MomentumPruning',
    'RandomPruning',
    'LayerwiseRandomPruning',
    'BernoulliPruning',
] 