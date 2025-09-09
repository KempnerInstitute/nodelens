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
from .parallel_batch import ParallelBatchPruning

from .alignment_based import (
    AlignmentPruning,
    HybridPruning,
    GlobalAlignmentPruning,
)
from .cascading import CascadingAlignmentPruning

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
    'ParallelBatchPruning',

    
    # Alignment-based
    'AlignmentPruning',
    'HybridPruning',
    'GlobalAlignmentPruning',
    'CascadingAlignmentPruning',
] 