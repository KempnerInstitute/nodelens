"""
Pruning strategies for the alignment framework.
"""

from .magnitude import (
    MagnitudePruning,
    IterativeMagnitudePruning,
    GlobalMagnitudePruning,
)
from .gradient import (
    GradientPruning,
    FisherPruning,
    MomentumPruning,
)
from .random import (
    RandomPruning,
    LayerwiseRandomPruning,
    BernoulliPruning,
)
from .parallel import (
    ParallelModePruning,
    TensorizedPruning,
    AsyncParallelPruning,
)
from .alignment_based import (
    AlignmentPruning,
    HybridPruning,
    GlobalAlignmentPruning,
)

__all__ = [
    # Magnitude
    'MagnitudePruning',
    'IterativeMagnitudePruning',
    'GlobalMagnitudePruning',
    
    # Gradient
    'GradientPruning',
    'FisherPruning',
    'MomentumPruning',
    
    # Random
    'RandomPruning',
    'LayerwiseRandomPruning',
    'BernoulliPruning',
    
    # Parallel
    'ParallelModePruning',
    'TensorizedPruning',
    'AsyncParallelPruning',
    
    # Alignment-based
    'AlignmentPruning',
    'HybridPruning',
    'GlobalAlignmentPruning',
] 