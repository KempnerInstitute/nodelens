"""
Experiments module for alignment analysis.

This module provides various experiments for analyzing neural network alignment,
including general alignment analysis, LLM alignment, and cluster-based analysis.
"""

from .base import BaseExperiment, ExperimentConfig
from .general_alignment import GeneralAlignmentConfig, GeneralAlignmentExperiment
# NOTE: Keep the package importable even if optional / heavy experiment modules
# have missing dependencies or transient syntax issues. This is important because
# vision-only workflows (cluster/pruning) should not break due to LLM-only code.
try:
    from .llm_experiments import LLMAlignmentExperiment
except Exception as _exc:  # pragma: no cover
    LLMAlignmentExperiment = None  # type: ignore[assignment]
from .cluster_experiments import (
    ClusterAnalysisExperiment,
    ClusterAnalysisConfig,
    VisionExperiment,  # backward compat
    VisionExperimentConfig,  # backward compat
)

__all__ = [
    # Base classes
    "BaseExperiment",
    "ExperimentConfig",
    # Main experiments
    "GeneralAlignmentExperiment",
    "GeneralAlignmentConfig",
    # Optional (may be None if import failed)
    "LLMAlignmentExperiment",
    "ClusterAnalysisExperiment",
    "ClusterAnalysisConfig",
    "VisionExperiment",  # backward compat alias
    "VisionExperimentConfig",  # backward compat alias
]
