"""Similarity-based metrics for neural network alignment analysis."""

from .cosine_similarity import WeightCosineSimilarity, ActivationCosineSimilarity
# from .node_correlation import NodeCorrelation  # TODO: Implement
# from .weight_similarity import WeightDotSimilarity, WeightEuclideanDistance  # TODO: Implement

__all__ = [
    'WeightCosineSimilarity',
    'ActivationCosineSimilarity',
    # 'NodeCorrelation',  # TODO: Implement
    # 'WeightDotSimilarity',  # TODO: Implement
    # 'WeightEuclideanDistance',  # TODO: Implement
] 