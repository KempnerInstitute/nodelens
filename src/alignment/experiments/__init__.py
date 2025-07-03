"""
Alignment experiments module.

This module provides the framework for running various alignment experiments
including training, evaluation, pruning, and analysis.
"""

from .base import BaseExperiment, ExperimentConfig
from .runner import ExperimentRunner
from .general_alignment import GeneralAlignmentExperiment, GeneralAlignmentConfig
from .config_components import (
    TrainingConfig,
    PruningConfig,
    EvaluationConfig,
    CNNConfig,
    MultiNetworkConfig,
    create_standard_training_config,
    create_standard_pruning_config,
    create_quick_test_config,
    flatten_config_dict,
    unflatten_config_dict
)

__all__ = [
    'BaseExperiment',
    'ExperimentConfig',
    'ExperimentRunner',
    'GeneralAlignmentExperiment',
    'GeneralAlignmentConfig',
    # Config components
    'TrainingConfig',
    'PruningConfig',
    'EvaluationConfig',
    'CNNConfig',
    'MultiNetworkConfig',
    # Factory functions
    'create_standard_training_config',
    'create_standard_pruning_config',
    'create_quick_test_config',
    # Compatibility helpers
    'flatten_config_dict',
    'unflatten_config_dict'
] 