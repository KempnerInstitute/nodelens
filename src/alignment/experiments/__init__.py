"""
Experiments module for alignment analysis.

This module provides various experiments for analyzing neural network alignment,
including general alignment analysis, multi-network experiments, and utilities.
"""

from .base import BaseExperiment, ExperimentConfig
# Configuration components
from .config_components import (CNNConfig, EvaluationConfig,
                                MultiNetworkConfig, PruningConfig,
                                TrainingConfig,
                                create_backward_compatible_config,
                                create_config_from_dict)
from .general_alignment import (GeneralAlignmentConfig,
                                GeneralAlignmentExperiment)
from .llm_experiments import LLMAlignmentExperiment
# Training utilities
from .training_utils import (convert_training_history,
                             create_experiment_trainer, evaluate_with_metrics,
                             train_with_metrics)

__all__ = [
    # Base classes
    "BaseExperiment",
    "ExperimentConfig",
    # Main experiments
    "GeneralAlignmentExperiment",
    "GeneralAlignmentConfig",
    "LLMAlignmentExperiment",
    # Configuration components
    "TrainingConfig",
    "PruningConfig",
    "EvaluationConfig",
    "CNNConfig",
    "MultiNetworkConfig",
    "create_config_from_dict",
    "create_backward_compatible_config",
    # Training utilities
    "create_experiment_trainer",
    "train_with_metrics",
    "evaluate_with_metrics",
    "convert_training_history",
]
