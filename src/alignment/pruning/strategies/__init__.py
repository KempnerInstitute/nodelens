"""
Pruning strategies for the alignment framework.
"""

from .alignment_based import AlignmentPruning, GlobalAlignmentPruning, HybridPruning
from .cascading import CascadingAlignmentPruning
from .cluster_aware import ClusterAwarePruning, ClusterAwarePruningConfig, CompositePruning
from .gradient import FisherPruning, GradientPruning, MomentumPruning
from .llm_baselines import WandaPruning, SparseGPTPruning
from .magnitude import GlobalMagnitudePruning, IterativeMagnitudePruning, MagnitudePruning
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
    # Cluster-aware (vision paper)
    "ClusterAwarePruning",
    "ClusterAwarePruningConfig",
    "CompositePruning",
    # LLM Baselines (Wanda, SparseGPT)
    "WandaPruning",
    "SparseGPTPruning",
]
