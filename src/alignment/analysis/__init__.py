"""
Analysis utilities for alignment experiments.

This module provides tools for:
- Aggregating experiment results
- Generating reports in multiple formats
- Creating visualizations

The module has been simplified to provide unified interfaces for common tasks.
"""

# Aggregation
from .aggregation import (
    ResultAggregator,
    MetricAggregator,
    LayerAggregator,
)

# Unified Reporting
from .unified_reporter import (
    UnifiedReporter,
    generate_quick_report,
)

# Unified Visualization
from .visualization.unified_visualizer import (
    UnifiedVisualizer,
    plot_quick_summary,
)

__all__ = [
    # Aggregation
    'ResultAggregator',
    'MetricAggregator',
    'LayerAggregator',
    # Reporting
    'UnifiedReporter',
    'generate_quick_report',
    # Visualization
    'UnifiedVisualizer',
    'plot_quick_summary',
] 