"""
Experiments module for alignment analysis.

This module provides various experiments for analyzing neural network alignment,
including general alignment analysis, multi-network experiments, and utilities.
"""

from .base import BaseExperiment, ExperimentConfig
from .general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig
from .llm_experiments import LLMAlignmentExperiment

# Configuration components
from .config_components import (
    TrainingConfig,
    PruningConfig,
    EvaluationConfig,
    CNNConfig,
    MultiNetworkConfig,
    create_config_from_dict,
    create_backward_compatible_config
)

# Training utilities
from .training_utils import (
    create_experiment_trainer,
    train_with_metrics,
    evaluate_with_metrics,
    convert_training_history
)

__all__ = [
    # Base classes
    'BaseExperiment',
    'ExperimentConfig',
    # Main experiments
    'GeneralAlignmentExperiment',
    'GeneralAlignmentConfig',
    'LLMAlignmentExperiment',
    # Configuration components
    'TrainingConfig',
    'PruningConfig',
    'EvaluationConfig',
    'CNNConfig',
    'MultiNetworkConfig',
    'create_config_from_dict',
    'create_backward_compatible_config',
    # Training utilities
    'create_experiment_trainer',
    'train_with_metrics',
    'evaluate_with_metrics',
    'convert_training_history',
] 