"""
Information-theoretic metrics for neural network alignment.
"""

from .conditional_mutual_information import ConditionalMutualInformation
from .gaussian_mi import GaussianMIAnalytic
from .mi_projection import MIProjectionVsMeanInput
from .mutual_information import MutualInformationBinning, MutualInformationGaussian
from .pairwise_gaussian import PairwiseRedundancyGaussian
from .pid import SharedInformation
from .pid import SynergisticInformation as PIDSynergisticInformation
from .pid import UniqueInformationX, UniqueInformationY
from .redundancy import AverageRedundancy
from .synergy_mmi import SynergyGaussianMMI
from .synergy_continuous import SynergyContinuousTarget

# Import higher-order metrics if available
try:
    from .higher_order import ConnectedInformation, InteractionInformation, SynergisticInformation, TotalCorrelation

    _has_higher_order = True
except ImportError:
    _has_higher_order = False

__all__ = [
    # Mutual Information
    "MutualInformationGaussian",
    "MutualInformationBinning",
    "GaussianMIAnalytic",
    # Redundancy
    "AverageRedundancy",
    "PairwiseRedundancyGaussian",
    # Synergy
    "SynergyGaussianMMI",
    # PID
    "SharedInformation",
    "UniqueInformationX",
    "UniqueInformationY",
    "PIDSynergisticInformation",
    # Conditional MI
    "ConditionalMutualInformation",
    # MI Projection
    "MIProjectionVsMeanInput",
]

if _has_higher_order:
    __all__.extend(
        [
            "TotalCorrelation",
            "InteractionInformation",
            "ConnectedInformation",
            "SynergisticInformation",
        ]
    )
