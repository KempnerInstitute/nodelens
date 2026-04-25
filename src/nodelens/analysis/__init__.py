"""
Analysis utilities for alignment experiments.

This module provides tools for:
- Aggregating experiment results
- Generating reports in multiple formats
- Creating visualizations
- Running unified analysis from configuration

The module has been simplified to provide unified interfaces for common tasks.
"""

# Aggregation
from .aggregation import LayerAggregator, MetricAggregator, ResultAggregator

# Unified Analysis Runner
from .analysis_runner import AnalysisConfig, AnalysisRunner, run_analysis_from_config

# Cascade Analysis
from .cascade_analysis import CascadeAnalysis, CascadeResult, DamagePrediction, DamageResult

# Clustering Analysis
from .clustering import ClusterResult, CrossLayerHaloAnalysis, HaloResult, MetricSpaceClustering

# Mechanism validation (general-purpose; reused across experiments)
from .mechanism_validation import (
    HaloReceiverDisruptionResult,
    SynergyPairLesionResult,
    validate_halo_receiver_disruption,
    validate_synergy_pair_lesions,
)

# Semantic hooks (non-pruning interpretability-facing analyses)
from .semantic_hooks import ClassSelectivityResult, compute_class_selectivity

# Unified Reporting
from .unified_reporter import UnifiedReporter, generate_quick_report

# Unified Visualization
from .visualization.unified_visualizer import UnifiedVisualizer, plot_quick_summary

__all__ = [
    # Aggregation
    "ResultAggregator",
    "MetricAggregator",
    "LayerAggregator",
    # Reporting
    "UnifiedReporter",
    "generate_quick_report",
    # Visualization
    "UnifiedVisualizer",
    "plot_quick_summary",
    # Analysis Runner
    "AnalysisRunner",
    "AnalysisConfig",
    "run_analysis_from_config",
    # Clustering
    "MetricSpaceClustering",
    "ClusterResult",
    "CrossLayerHaloAnalysis",
    "HaloResult",
    # Cascade Analysis
    "CascadeAnalysis",
    "DamagePrediction",
    "CascadeResult",
    "DamageResult",
    # Mechanism validation
    "HaloReceiverDisruptionResult",
    "SynergyPairLesionResult",
    "validate_halo_receiver_disruption",
    "validate_synergy_pair_lesions",
    # Semantic hooks
    "ClassSelectivityResult",
    "compute_class_selectivity",
]
