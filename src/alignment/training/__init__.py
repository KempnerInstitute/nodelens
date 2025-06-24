"""
Training utilities for neural networks.
"""

from .base import BaseTrainer, TrainingConfig
from .multi_network import train_networks_fully_tensorized, TensorizedNetworkWrapper

__all__ = [
    'BaseTrainer',
    'TrainingConfig',
    'train_networks_fully_tensorized',
    'TensorizedNetworkWrapper',
] 