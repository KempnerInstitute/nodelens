"""
Core module for the alignment metrics framework.

This module provides the foundational abstractions, protocols, and registries
used throughout the framework.
"""

from .base import BaseDataset, BaseExperiment, BaseMetric, BaseModel
from .protocols import AlignmentMetric, DatasetWrapper, Experiment, MetricAggregator
from .protocols import ModelWrapper as ModelWrapperProtocol
from .protocols import ResultReporter
from .registry import (
    Registry,
    get_dataset,
    get_experiment,
    get_metric,
    get_model,
    register_dataset,
    register_experiment,
    register_metric,
    register_model,
)

__all__ = [
    # Protocols
    "AlignmentMetric",
    "ModelWrapperProtocol",
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
