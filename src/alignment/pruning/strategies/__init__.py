"""Pruning strategies."""

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

from .parallel import (
    ParallelModePruning,
    TensorizedPruning,
    AsyncParallelPruning,
    ParallelPruningResult
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
    'ParallelModePruning',
    'TensorizedPruning',
    'AsyncParallelPruning',
    'ParallelPruningResult',
] 