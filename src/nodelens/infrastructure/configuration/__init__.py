"""
Configuration infrastructure for the alignment framework.

NOTE: This module provides basic configuration utilities.
For the main experiment configuration system, use nodelens.configs instead:
    from nodelens.configs import ExperimentConfig, load_config

This module contains simpler utilities that can be used standalone.
"""

from .config import Config, ExperimentConfig, load_config, merge_configs, save_config

__all__ = [
    "Config",
    "ExperimentConfig",
    "load_config",
    "save_config",
    "merge_configs",
]
