"""
Training utilities for the alignment framework.

This module provides various training methods including
standard and fully tensorized training.
"""

from .tensorized import train_networks_fully_tensorized

__all__ = [
    'train_networks_fully_tensorized',
] 