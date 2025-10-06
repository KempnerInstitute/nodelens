"""
Pruning strategies for the alignment framework.
"""

from .alignment_based import AlignmentPruning, GlobalAlignmentPruning, HybridPruning
from .cascading import CascadingAlignmentPruning
from .gradient import FisherPruning, GradientPruning, MomentumPruning
from .magnitude import (
    GlobalMagnitudePruning,
    IterativeMagnitudePruning,
    MagnitudePruning,
)
from .parallel import AsyncParallelPruning, ParallelModePruning, TensorizedPruning
from .parallel_batch import ParallelBatchPruning
from .random import BernoulliPruning, LayerwiseRandomPruning, RandomPruning

__all__ = [
    # Magnitude
    "MagnitudePruning",
    "IterativeMagnitudePruning",
    "GlobalMagnitudePruning",
    # Gradient
    "GradientPruning",
    "FisherPruning",
    "MomentumPruning",
    # Random
    "RandomPruning",
    "LayerwiseRandomPruning",
    "BernoulliPruning",
    # Parallel
    "ParallelModePruning",
    "TensorizedPruning",
    "AsyncParallelPruning",
    "ParallelBatchPruning",
    # Alignment-based
    "AlignmentPruning",
    "HybridPruning",
    "GlobalAlignmentPruning",
    "CascadingAlignmentPruning",
]
