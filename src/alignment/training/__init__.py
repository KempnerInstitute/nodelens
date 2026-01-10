"""
Training utilities for neural networks.
"""

from .base import BaseTrainer, TrainingConfig
from .experiment_trainer import ExperimentTrainer, ExperimentTrainingConfig
from .multi_network import TensorizedNetworkWrapper, train_networks_fully_tensorized
from .evaluation import (
    evaluate_classification,
    evaluate_perplexity,
    evaluate_regression,
    evaluate_model,
    EvaluationManager,
)

__all__ = [
    "BaseTrainer",
    "TrainingConfig",
    "train_networks_fully_tensorized",
    "TensorizedNetworkWrapper",
    "ExperimentTrainer",
    "ExperimentTrainingConfig",
    # Evaluation
    "evaluate_classification",
    "evaluate_perplexity",
    "evaluate_regression",
    "evaluate_model",
    "EvaluationManager",
]
