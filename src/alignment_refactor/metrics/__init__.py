"""
Metrics module for the alignment metrics framework.

This module provides various metrics for measuring alignment between
neural network layers and their inputs/outputs.
"""

# Import all metric classes for easy access
from alignment_refactor.metrics.base import BaseMetric, MetricComputer
from alignment_refactor.metrics.rayleigh import *
from alignment_refactor.metrics.information import *
from alignment_refactor.metrics.similarity import *

# Auto-discover and register all metrics
from alignment_refactor.core.registry import discover_and_register

# Discover metrics in submodules
discover_and_register('alignment_refactor.metrics.rayleigh', 'metrics')
discover_and_register('alignment_refactor.metrics.information', 'metrics')
discover_and_register('alignment_refactor.metrics.similarity', 'metrics')

__all__ = [
    'BaseMetric',
    'MetricComputer',
] 