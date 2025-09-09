"""
Model wrappers and implementations for the alignment metrics framework.

This module provides wrappers that add activation tracking and other
functionality needed for alignment analysis to standard PyTorch models.
"""

from .base import BaseModelWrapper
from .wrappers import (
    ModelWrapper,
    AlignmentNetwork,
    ActivationTracker,
)
from .architectures.standard_models import (
    MLP,
    CNN2P2,
    SimpleConvNet,
    create_model,
)
from . import hub  # registers torchvision/timm/huggingface model loaders

# Register standard models
from ..core.registry import register_model

# Register the models
register_model("mlp")(MLP)
register_model("cnn2p2")(CNN2P2)
register_model("simple_conv")(SimpleConvNet)
register_model("simpleconvnet")(SimpleConvNet)

__all__ = [
    'BaseModelWrapper',
    'ModelWrapper',
    'AlignmentNetwork',
    'ActivationTracker',
    'MLP',
    'CNN2P2',
    'SimpleConvNet',
    'create_model',
    'hub',
] 