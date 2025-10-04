"""
Service Layer for Alignment Framework.

This module provides high-level services that compose core functionality
for common operations like activation capture, scoring, and mask generation.
"""

from .activation_capture import (
    ActivationCaptureService,
    ActivationData,
    create_capture_service
)
from .scoring import (
    NodeScoringService,
    CompositeScores,
    create_scoring_service
)
from .mask_ops import MaskOperations

__all__ = [
    # Activation capture
    'ActivationCaptureService',
    'ActivationData',
    'create_capture_service',
    
    # Scoring
    'NodeScoringService',
    'CompositeScores',
    'create_scoring_service',
    
    # Mask operations
    'MaskOperations',
]

