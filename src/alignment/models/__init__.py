"""
Model wrappers and implementations for the alignment metrics framework.

This module provides wrappers that add activation tracking and other
functionality needed for alignment analysis to standard PyTorch models.
"""

# Register standard models
from ..core.registry import register_model
from . import hub  # registers torchvision/timm/huggingface model loaders
from .architectures.standard_models import CNN2P2, MLP, SimpleConvNet, create_model
from .hub import adapt_model_for_dataset
from .base import BaseModelWrapper
from .transformers import LLaMAWrapper, TransformerWrapperEnhanced
from .wrappers import ActivationTracker, AlignmentNetwork, ModelWrapper

# Register the models
register_model("mlp")(MLP)
register_model("cnn2p2")(CNN2P2)
register_model("simple_conv")(SimpleConvNet)
register_model("simpleconvnet")(SimpleConvNet)

__all__ = [
    "BaseModelWrapper",
    "ModelWrapper",
    "AlignmentNetwork",
    "ActivationTracker",
    "TransformerWrapperEnhanced",
    "LLaMAWrapper",
    "MLP",
    "CNN2P2",
    "SimpleConvNet",
    "create_model",
    "adapt_model_for_dataset",
    "hub",
]
