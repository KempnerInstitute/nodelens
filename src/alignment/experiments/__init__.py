"""
Experiments module for alignment analysis.

This module provides various experiments for analyzing neural network alignment,
including general alignment analysis, LLM alignment, and cluster-based analysis.
"""

from .base import BaseExperiment, ExperimentConfig
from .general_alignment import GeneralAlignmentConfig, GeneralAlignmentExperiment
from .llm_experiments import LLMAlignmentExperiment
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
    "LLMAlignmentExperiment",
    "ClusterAnalysisExperiment",
    "ClusterAnalysisConfig",
    "VisionExperiment",  # backward compat alias
    "VisionExperimentConfig",  # backward compat alias
]
