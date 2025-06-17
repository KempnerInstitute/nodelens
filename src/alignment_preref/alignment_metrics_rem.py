"""
Alignment metrics for neural network analysis.

DEPRECATED: This module is deprecated and will be removed in a future version.
Please import directly from `alignment.metrics` instead.

This module provides various metrics for measuring alignment between weight vectors
and activation vectors, including RQ (representation quality), MI (mutual information),
and other metrics for analyzing neural network representations.
"""

import warnings
from typing import Dict, List, Tuple, Union, Optional, Callable

# Show deprecation warning when the module is imported
warnings.warn(
    "The alignment.alignment_metrics module is deprecated and will be removed in a future version. "
    "Please import directly from alignment.metrics instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from metrics_utils instead of metrics.py to avoid circular dependencies
from alignment_refac1.utils.metrics_utils import (
    AlignmentMetricBase,
    RQMetric,
    MIMetric,
    WeightSimilarityMetric,
    NodeRedundancyMetric,
    AlignmentMetricsFactory as AlignmentMetrics,
    alignment
)

# For backward compatibility, re-export everything
__all__ = [
    'AlignmentMetricBase',
    'RQMetric',
    'MIMetric',
    'WeightSimilarityMetric',
    'NodeRedundancyMetric',
    'AlignmentMetrics',
    'alignment'
]