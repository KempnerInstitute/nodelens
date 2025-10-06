"""
Service Layer for Alignment Framework.

This module provides high-level services that compose core functionality
for common operations like activation capture, scoring, and mask generation.
"""

from .activation_capture import (
    ActivationCaptureService,
    ActivationData,
    create_capture_service,
)
from .mask_ops import MaskOperations
from .scoring import CompositeScores, NodeScoringService, create_scoring_service

__all__ = [
    # Activation capture
    "ActivationCaptureService",
    "ActivationData",
    "create_capture_service",
    # Scoring
    "NodeScoringService",
    "CompositeScores",
    "create_scoring_service",
    # Mask operations
    "MaskOperations",
]
