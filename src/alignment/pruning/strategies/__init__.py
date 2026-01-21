"""
Pruning strategies for the alignment framework.
"""

from .adaptive import AdaptiveSensitivityPruning, LayerSensitivity
from .alignment_based import AlignmentPruning, GlobalAlignmentPruning, HybridPruning
from .cascading import CascadingAlignmentPruning
from .cluster_aware import ClusterAwarePruning, ClusterAwarePruningConfig, CompositePruning
from .eigenvector import EigenvectorPruning
from .gradient import FisherPruning, GradientPruning, MomentumPruning
from .movement import AdaptiveMovementPruning, MovementPruning
from .llm_baselines import WandaPruning, SparseGPTPruning, OWLPruning, LLMPrunerChannelMode
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
    # Eigenvector-based (PCA pruning)
    "EigenvectorPruning",
    # Movement-based (Sanh et al. 2020)
    "MovementPruning",
    "AdaptiveMovementPruning",
    # Adaptive sensitivity-based
    "AdaptiveSensitivityPruning",
    "LayerSensitivity",
    # Cluster-aware (vision paper) - includes depth/sparsity adaptive options via config
    "ClusterAwarePruning",
    "ClusterAwarePruningConfig",
    "CompositePruning",
    # LLM Baselines (Wanda, SparseGPT, OWL, LLM-Pruner)
    "WandaPruning",
    "SparseGPTPruning",
    "OWLPruning",
    "LLMPrunerChannelMode",
]
