"""
Core module for the alignment metrics framework.

This module provides the foundational abstractions, protocols, and registries
used throughout the framework.
"""

from alignment.core.protocols import (
    AlignmentMetric,
    ModelWrapper,
    DatasetWrapper,
    Experiment,
    MetricAggregator,
    ResultReporter,
)
from alignment.core.registry import (
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
from alignment.core.base import (
    BaseMetric,
    BaseModel,
    BaseDataset,
    BaseExperiment,
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