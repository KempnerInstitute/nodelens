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
    MetricVisualizer,
    LayerVisualizer,
    ComparisonVisualizer,
    AlignmentVisualizer,
    PruningVisualizer,
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
    'MetricVisualizer',
    'LayerVisualizer',
    'ComparisonVisualizer',
    'AlignmentVisualizer',
    'PruningVisualizer',
] 