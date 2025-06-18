"""
Alignment metrics package.
"""

from ..core.base import BaseMetric as AlignmentMetric
from .rayleigh import RayleighQuotient, RayleighQuotientAlternative
from .information import (
    MutualInformationGaussian, 
    MutualInformationBinning,
    AverageRedundancyGaussian,
    PartialInformationDecompositionSI,
    PartialInformationDecompositionUIY,
    PartialInformationDecompositionUIZ,
    PartialInformationDecompositionCI,
    ConditionalMutualInformation,
    MIProjectionVsMeanInput,
)
from .similarity import (
    ActivationCosineSimilarity,
    NodeRedundancy,
    WeightCosineSimilarity,
    WeightDotSimilarity,
    WeightEuclideanDistance,
    NodeCorrelation,
)

# Metric registry for easy lookup
METRIC_REGISTRY = {
    # Rayleigh quotient metrics
    'rayleigh_quotient': RayleighQuotient,
    'rayleigh_quotient_alternative': RayleighQuotientAlternative,
    
    # Information-theoretic metrics
    'mutual_information_gaussian': MutualInformationGaussian,
    'mutual_information_binning': MutualInformationBinning,
    'average_redundancy_gaussian': AverageRedundancyGaussian,
    'pid_si': PartialInformationDecompositionSI,
    'pid_uiy': PartialInformationDecompositionUIY,
    'pid_uiz': PartialInformationDecompositionUIZ,
    'pid_ci': PartialInformationDecompositionCI,
    'conditional_mutual_information': ConditionalMutualInformation,
    'mi_projection_vs_mean_input': MIProjectionVsMeanInput,
    
    # Similarity metrics
    'activation_cosine_similarity': ActivationCosineSimilarity,
    'node_redundancy': NodeRedundancy,
    'weight_cosine_similarity': WeightCosineSimilarity,
    'weight_dot_similarity': WeightDotSimilarity,
    'weight_euclidean_distance': WeightEuclideanDistance,
    'node_correlation': NodeCorrelation,
}

__all__ = [
    'AlignmentMetric',
    'METRIC_REGISTRY',
] + list(METRIC_REGISTRY.keys()) 