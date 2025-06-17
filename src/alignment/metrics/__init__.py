"""
Metrics module for the alignment metrics framework.

This module provides various metrics for measuring alignment between
neural network layers and their inputs/outputs.
"""

# Import all metric classes for easy access
from alignment.metrics.base import BaseMetric, MetricComputer
from alignment.metrics.rayleigh import *
from alignment.metrics.information import *
from alignment.metrics.similarity import *

# Auto-discover and register all metrics
from alignment.core.registry import discover_and_register

# Discover metrics in submodules
discover_and_register('alignment.metrics.rayleigh', 'metrics')
discover_and_register('alignment.metrics.information', 'metrics')
discover_and_register('alignment.metrics.similarity', 'metrics')

__all__ = [
    'BaseMetric',
    'MetricComputer',
] 