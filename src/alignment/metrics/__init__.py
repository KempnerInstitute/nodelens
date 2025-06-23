"""
Metrics for measuring neural network alignment.
"""

from ..core.registry import METRIC_REGISTRY

# Import all metric modules to register them
from . import rayleigh
from . import information
from . import similarity

# Import spectral metrics
from .spectral.spectral_alignment import (
    SpectralGapMetric,
    EigenvalueAlignmentMetric,
    SpectralClusteringAlignment,
    PowerIterationAlignment
)

# Import higher-order information metrics
from .information.higher_order import (
    TotalCorrelation,
    OInformation,
    SInformation,
    ConnectedInformation
)

# Import task-specific metrics
from .task_specific import (
    ClassificationAlignment,
    LanguageModelAlignment,
    VisionTaskAlignment,
    ReinforcementLearningAlignment
)

# Register new metrics
METRIC_REGISTRY['spectral_gap'] = SpectralGapMetric
METRIC_REGISTRY['eigenvalue_alignment'] = EigenvalueAlignmentMetric
METRIC_REGISTRY['spectral_clustering'] = SpectralClusteringAlignment
METRIC_REGISTRY['power_iteration'] = PowerIterationAlignment

METRIC_REGISTRY['total_correlation'] = TotalCorrelation
METRIC_REGISTRY['o_information'] = OInformation
METRIC_REGISTRY['s_information'] = SInformation
METRIC_REGISTRY['connected_information'] = ConnectedInformation

METRIC_REGISTRY['classification_alignment'] = ClassificationAlignment
METRIC_REGISTRY['language_model_alignment'] = LanguageModelAlignment
METRIC_REGISTRY['vision_task_alignment'] = VisionTaskAlignment
METRIC_REGISTRY['reinforcement_learning_alignment'] = ReinforcementLearningAlignment

# For convenience, expose the registry
__all__ = ['METRIC_REGISTRY'] 