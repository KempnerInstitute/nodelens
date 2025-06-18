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
    # Redundancy
    'AverageRedundancy',
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