"""
Clustering analysis module for neural network channels.

Provides clustering in (RQ, Redundancy, Synergy) space to identify
functional types: Critical, Redundant, Synergistic, Background.

Includes:
- MetricSpaceClustering: K-means clustering with metric ablation support
- CrossLayerHaloAnalysis: Downstream dependency analysis with permutation baselines
- AblationResult: Results from metric ablation studies
"""

from .metric_clustering import (
    MetricSpaceClustering, 
    ClusterResult, 
    AblationResult,
    METRIC_ABLATIONS,
)
from .cross_layer_halo import CrossLayerHaloAnalysis, HaloResult

__all__ = [
    "MetricSpaceClustering",
    "ClusterResult",
    "AblationResult",
    "METRIC_ABLATIONS",
    "CrossLayerHaloAnalysis",
    "HaloResult",
]
