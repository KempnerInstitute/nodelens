"""
Clustering analysis module for neural network channels.

Provides clustering in (RQ, Redundancy, Synergy) space to identify
functional types: Critical, Redundant, Synergistic, Background.
"""

from .metric_clustering import MetricSpaceClustering, ClusterResult
from .cross_layer_halo import CrossLayerHaloAnalysis, HaloResult

__all__ = [
    "MetricSpaceClustering",
    "ClusterResult",
    "CrossLayerHaloAnalysis",
    "HaloResult",
]
