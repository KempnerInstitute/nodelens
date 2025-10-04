"""
Information-theoretic metrics for neural network alignment.
"""

from .mutual_information import (
    MutualInformationGaussian,
    MutualInformationBinning,
)
from .redundancy import AverageRedundancy
from .pid import (
    SharedInformation,
    UniqueInformationX,
    UniqueInformationY,
    SynergisticInformation as PIDSynergisticInformation,
)
from .conditional_mutual_information import ConditionalMutualInformation
from .mi_projection import MIProjectionVsMeanInput
from .gaussian_mi import GaussianMIAnalytic
from .pairwise_gaussian import PairwiseRedundancyGaussian
from .synergy_mmi import SynergyGaussianMMI

# Import higher-order metrics if available
try:
    from .higher_order import (
        TotalCorrelation,
        InteractionInformation,
        ConnectedInformation,
        SynergisticInformation,
    )
    _has_higher_order = True
except ImportError:
    _has_higher_order = False

__all__ = [
    # Mutual Information
    'MutualInformationGaussian',
    'MutualInformationBinning',
    'GaussianMIAnalytic',
    # Redundancy
    'AverageRedundancy',
    'PairwiseRedundancyGaussian',
    # Synergy
    'SynergyGaussianMMI',
    # PID
    'SharedInformation',
    'UniqueInformationX',
    'UniqueInformationY',
    'PIDSynergisticInformation',
    # Conditional MI
    'ConditionalMutualInformation',
    # MI Projection
    'MIProjectionVsMeanInput',
]

if _has_higher_order:
    __all__.extend([
        'TotalCorrelation',
        'InteractionInformation',
        'ConnectedInformation',
        'SynergisticInformation',
    ]) 