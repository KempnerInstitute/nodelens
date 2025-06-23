"""
Analysis utilities for alignment experiments.

This module provides tools for:
- Aggregating experiment results
- Generating reports
- Visualizing metrics
"""

# Aggregation
from .aggregation import (
    ResultAggregator,
    MetricAggregator,
    LayerAggregator,
)

# Reporting
from .reporting import (
    HTMLReporter,
    MarkdownReporter,
    JSONReporter,
)

# Visualization
from .visualization import (
    plot_metric_evolution,
    plot_layer_comparison,
    plot_correlation_matrix,
    create_interactive_dashboard,
)

__all__ = [
    # Aggregation
    'ResultAggregator',
    'MetricAggregator',
    'LayerAggregator',
    # Reporting
    'HTMLReporter',
    'MarkdownReporter',
    'JSONReporter',
    # Visualization
    'plot_metric_evolution',
    'plot_layer_comparison',
    'plot_correlation_matrix',
    'create_interactive_dashboard',
] 