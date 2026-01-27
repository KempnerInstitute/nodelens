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
from .llm_baselines import (
    WandaPruning,
    SparseGPTPruning,
    OWLPruning,
    LLMPrunerChannelMode,
    FLAPPruning,
    RIAPruning,
    SlimLLMPruning,
)
from .magnitude import GlobalMagnitudePruning, IterativeMagnitudePruning, MagnitudePruning
from .parallel import AsyncParallelPruning, ParallelModePruning, TensorizedPruning
from .parallel_batch import ParallelBatchPruning
from .random import BernoulliPruning, LayerwiseRandomPruning, RandomPruning

# Optional strategy: CHIP (may not be vendored in all repos).
try:
    from .chip import CHIPPruning, compute_chip_scores, chip_score_channels  # type: ignore
except Exception:  # pragma: no cover
    # Provide import-time stability for users who don't have CHIP code vendored.
    class CHIPPruning:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CHIPPruning is unavailable: missing `alignment.pruning.strategies.chip`."
            )

    def compute_chip_scores(*args, **kwargs):  # type: ignore
        raise ImportError(
            "compute_chip_scores is unavailable: missing `alignment.pruning.strategies.chip`."
        )

    def chip_score_channels(*args, **kwargs):  # type: ignore
        raise ImportError(
            "chip_score_channels is unavailable: missing `alignment.pruning.strategies.chip`."
        )

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
    # Cluster-aware (vision models) - includes depth/sparsity adaptive options via config
    "ClusterAwarePruning",
    "ClusterAwarePruningConfig",
    "CompositePruning",
    # LLM Baselines (Wanda, SparseGPT, OWL, LLM-Pruner, FLAP, RIA, SlimLLM)
    "WandaPruning",
    "SparseGPTPruning",
    "OWLPruning",
    "LLMPrunerChannelMode",
    "FLAPPruning",
    "RIAPruning",
    "SlimLLMPruning",
    # CHIP (Sui et al. NeurIPS 2021) - inter-channel correlation pruning
    "CHIPPruning",
    "compute_chip_scores",
    "chip_score_channels",
]
