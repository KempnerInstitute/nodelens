"""
Configuration management for the Neural Network Alignment framework.

This module provides:
- Legacy config loading (load_config)
- Unified config schema (UnifiedConfig)
- Config validation

Usage:
    # Legacy (still works)
    from alignment.configs import load_config
    config = load_config("path/to/config.yaml")
    
    # New unified config (recommended)
    from alignment.configs import load_unified_config, UnifiedConfig
    config = load_unified_config("path/to/config.yaml")
    
    # Programmatic config
    from alignment.configs import UnifiedConfig, ExperimentConfig
    config = UnifiedConfig(
        experiment=ExperimentConfig(name="my_exp", type="cluster_analysis"),
        ...
    )
"""

from .config_loader import load_config, save_config
from .config_validator import validate_config

# Unified config system
from .unified_config import (
    # Main config class
    UnifiedConfig,
    # Sub-config classes
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    CalibrationConfig,
    MetricsConfig,
    MetricItemConfig,
    ClusteringConfig,
    SupernodeConfig,
    HaloConfig,
    CascadeConfig,
    PruningConfig,
    PruningMethodConfig,
    EvaluationConfig,
    VisualizationConfig,
    OutputConfig,
    # Loading functions
    load_unified_config,
    create_config_template,
)

__all__ = [
    # Legacy
    "load_config",
    "save_config",
    "validate_config",
    # Unified config
    "UnifiedConfig",
    "ExperimentConfig",
    "ModelConfig",
    "DatasetConfig",
    "CalibrationConfig",
    "MetricsConfig",
    "MetricItemConfig",
    "ClusteringConfig",
    "SupernodeConfig",
    "HaloConfig",
    "CascadeConfig",
    "PruningConfig",
    "PruningMethodConfig",
    "EvaluationConfig",
    "VisualizationConfig",
    "OutputConfig",
    "load_unified_config",
    "create_config_template",
]
