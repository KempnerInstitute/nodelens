"""
Core module for the alignment metrics framework.

This module provides the foundational abstractions, protocols, and registries
used throughout the framework.
"""

from .protocols import (
    AlignmentMetric,
    ModelWrapper as ModelWrapperProtocol,
    DatasetWrapper,
    Experiment,
    MetricAggregator,
    ResultReporter,
)
from .registry import (
    Registry,
    register_metric,
    register_model,
    register_dataset,
    register_experiment,
    get_metric,
    get_model,
    get_dataset,
    get_experiment,
)
from .base import (
    BaseMetric,
    BaseModel,
    BaseDataset,
    BaseExperiment,
)
from .wrappers import (
    ModelWrapper,
)

__all__ = [
    # Protocols
    "AlignmentMetric",
    "ModelWrapper",
    "DatasetWrapper",
    "Experiment",
    "MetricAggregator",
    "ResultReporter",
    # Registry
    "Registry",
    "register_metric",
    "register_model",
    "register_dataset",
    "register_experiment",
    "get_metric",
    "get_model",
    "get_dataset",
    "get_experiment",
    # Base classes
    "BaseMetric",
    "BaseModel",
    "BaseDataset",
    "BaseExperiment",
] 