"""Information-theoretic metrics for neural network alignment analysis."""

from .mutual_information import MutualInformationGaussian, MutualInformationBinning
# from .conditional import ConditionalMutualInformation  # TODO: Implement conditional MI
from .pid import (
    PartialInformationDecomposition, 
    SharedInformation, 
    UniqueInformationX,
    UniqueInformationY,
    SynergisticInformation
)
from .redundancy import AverageRedundancy, NodeRedundancy, LayerRedundancy

__all__ = [
    'MutualInformationGaussian',
    'MutualInformationBinning',
    # 'ConditionalMutualInformation',  # TODO: Implement conditional MI
    'PartialInformationDecomposition',
    'SharedInformation',
    'UniqueInformationX',
    'UniqueInformationY',
    'SynergisticInformation',
    'AverageRedundancy',
    'NodeRedundancy',
    'LayerRedundancy',
] 