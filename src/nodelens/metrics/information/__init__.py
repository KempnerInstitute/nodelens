"""
Information-theoretic metrics for neural network analysis.
"""

from .conditional_mutual_information import ConditionalMutualInformation
from .gaussian_mi import FastGaussianMI, GaussianMIAnalytic
from .mi_projection import MIProjectionVsMeanInput
from .mutual_information import MutualInformationBinning, MutualInformationGaussian
from .pairwise_gaussian import PairwiseRedundancyGaussian
from .pid import SharedInformation
from .pid import SynergisticInformation as PIDSynergisticInformation
from .pid import UniqueInformationX, UniqueInformationY
from .redundancy import AverageRedundancy, LayerRedundancy
from .synergy_continuous import SynergyContinuousTarget
from .synergy_mmi import SynergyGaussianMMI

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
    "FastGaussianMI",  # Fast MI variant using GAP for CNNs
    # Redundancy
    "AverageRedundancy",
    "LayerRedundancy",
    "PairwiseRedundancyGaussian",
    # Synergy
    "SynergyGaussianMMI",
    "SynergyContinuousTarget",
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
