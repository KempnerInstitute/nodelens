"""Task-specific alignment metrics for different domains and objectives."""

# General task alignment
# Activation-based importance
from .activation_magnitude import ActivationL2Norm, ActivationMean, ActivationNorm, ActivationVariance

# Domain-specific metrics
from .classification import ClassificationAlignment
from .general import ClassSelectivity, FeatureImportance, RepresentationQuality, TaskAlignment
from .language_model import LanguageModelAlignment
from .reinforcement_learning import ReinforcementLearningAlignment
from .vision import VisionTaskAlignment

__all__ = [
    # General
    "TaskAlignment",
    "FeatureImportance",
    "RepresentationQuality",
    "ClassSelectivity",
    # Activation-based
    "ActivationL2Norm",
    "ActivationMean",
    "ActivationNorm",
    "ActivationVariance",
    # Domain-specific
    "ClassificationAlignment",
    "LanguageModelAlignment",
    "VisionTaskAlignment",
    "ReinforcementLearningAlignment",
]
