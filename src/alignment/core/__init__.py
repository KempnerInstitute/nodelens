"""
Core module for the alignment metrics framework.

This module provides the foundational abstractions, protocols, and registries
used throughout the framework.

Key Components:
- **Protocols**: Interface definitions that components must implement
- **Registry**: Central registration system for discoverable components
- **Base Classes**: Optional abstract base classes for convenience

Example - Registering a custom metric:

    from alignment.core import register_metric, BaseMetric
    
    @register_metric("my_metric", category="custom", tags=["experimental"])
    class MyMetric(BaseMetric):
        name = "my_metric"
        
        def compute(self, outputs, **kwargs):
            return per_neuron_scores

Example - Using registered components:

    from alignment.core import get_metric, METRIC_REGISTRY
    
    # By name
    metric = get_metric("rayleigh_quotient")
    
    # List available
    print(METRIC_REGISTRY.list())
    
    # Search by tag
    print(METRIC_REGISTRY.search(tags=["alignment"]))
"""

from .base import BaseDataset, BaseExperiment, BaseMetric, BaseModel

# Protocols (interface definitions)
from .protocols import (
    AlignmentMetric,
    DatasetWrapper,
    Experiment,
    MetricAggregator,
    ResultReporter,
    # New protocols for enhanced modularity
    Analyzer,
    Pruner,
    Visualizer,
    Evaluator,
    Preprocessor,
    # Base classes from protocols
    BaseMetric as BaseMetricProtocol,
    BaseAnalyzer,
    BasePruner,
    # Config dataclasses
    MetricConfig,
    AnalyzerConfig,
    PrunerConfig,
)
from .protocols import ModelWrapper as ModelWrapperProtocol

# Registry system
from .registry import (
    # Core registry class
    Registry,
    ComponentInfo,
    # Global registries
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    EXPERIMENT_REGISTRY,
    AGGREGATOR_REGISTRY,
    REPORTER_REGISTRY,
    ANALYZER_REGISTRY,
    VISUALIZER_REGISTRY,
    PRUNER_REGISTRY,
    EVALUATOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    ALL_REGISTRIES,
    # Registration decorators
    register_metric,
    register_model,
    register_dataset,
    register_experiment,
    register_aggregator,
    register_reporter,
    register_analyzer,
    register_visualizer,
    register_pruner,
    register_evaluator,
    register_preprocessor,
    # Getter functions
    get_metric,
    get_model,
    get_dataset,
    get_experiment,
    get_aggregator,
    get_reporter,
    get_analyzer,
    get_visualizer,
    get_pruner,
    get_evaluator,
    get_preprocessor,
    # Unified factory functions
    create_component,
    create_from_config,
    list_all_components,
    print_registry_summary,
    # Discovery
    discover_and_register,
    discover_plugins,
    initialize_registries,
)

__all__ = [
    # Protocols (interfaces)
    "AlignmentMetric",
    "ModelWrapperProtocol",
    "DatasetWrapper",
    "Experiment",
    "MetricAggregator",
    "ResultReporter",
    "Analyzer",
    "Pruner",
    "Visualizer",
    "Evaluator",
    "Preprocessor",
    # Base classes
    "BaseMetric",
    "BaseMetricProtocol",
    "BaseAnalyzer",
    "BasePruner",
    "BaseModel",
    "BaseDataset",
    "BaseExperiment",
    # Config dataclasses
    "MetricConfig",
    "AnalyzerConfig",
    "PrunerConfig",
    # Registry
    "Registry",
    "ComponentInfo",
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "EXPERIMENT_REGISTRY",
    "AGGREGATOR_REGISTRY",
    "REPORTER_REGISTRY",
    "ANALYZER_REGISTRY",
    "VISUALIZER_REGISTRY",
    "PRUNER_REGISTRY",
    "EVALUATOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "ALL_REGISTRIES",
    # Registration decorators
    "register_metric",
    "register_model",
    "register_dataset",
    "register_experiment",
    "register_aggregator",
    "register_reporter",
    "register_analyzer",
    "register_visualizer",
    "register_pruner",
    "register_evaluator",
    "register_preprocessor",
    # Getter functions
    "get_metric",
    "get_model",
    "get_dataset",
    "get_experiment",
    "get_aggregator",
    "get_reporter",
    "get_analyzer",
    "get_visualizer",
    "get_pruner",
    "get_evaluator",
    "get_preprocessor",
    # Factory functions
    "create_component",
    "create_from_config",
    "list_all_components",
    "print_registry_summary",
    # Discovery
    "discover_and_register",
    "discover_plugins",
    "initialize_registries",
]
