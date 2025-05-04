"""
Alignment toolkit for neural network analysis.

This package provides tools for analyzing and understanding the alignment of
neural network weights with their inputs, enabling insights into how networks
represent and process information.
"""

# Basic library imports
import os
import sys
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Define version
__version__ = "0.3.0"

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import core modules
from alignment.config import (
    BaseConfig, 
    ModelConfig, 
    DatasetConfig, 
    TrainingConfig, 
    AlignmentConfig,
    ExperimentConfig
)

# Import metrics
from alignment.metrics import (
    AlignmentMetric,
    RankAlignmentMetric,
    NullSpaceAlignmentMetric,
    get_metric
)

# Import models
from alignment.models import (
    register_model,
    create_model,
    get_available_models,
    AlignmentNetwork
)

# Import training and evaluation
from alignment.training import (
    train_model,
    evaluate_model,
    load_checkpoint
)

# Import dropout functionality
from alignment.dropout import (
    progressive_dropout,
    eigenvector_dropout
)

# Import experiment base class
from alignment.experiments.experiment import Experiment

# Import experiment implementations
from alignment.experiments.alignment_experiments import AlignmentExperiment

# Import utility functions
from alignment.utils import (
    setup_logging,
    timer,
    debug,
    to_numpy,
    to_tensor,
    check_iterable,
    ensure_device
)

# Legacy imports for backward compatibility
# Rename train_model and evaluate_model for backward compatibility
train = train_model
evaluate = evaluate_model
from alignment.utils import timed

# Export key components
__all__ = [
    # Core classes
    "Experiment",
    "AlignmentExperiment",
    
    # Configuration
    "BaseConfig",
    "ModelConfig", 
    "DatasetConfig", 
    "TrainingConfig", 
    "AlignmentConfig",
    "ExperimentConfig",
    
    # Models
    "register_model",
    "create_model",
    "get_available_models",
    "AlignmentNetwork",
    
    # Metrics
    "AlignmentMetric",
    "RankAlignmentMetric",
    "NullSpaceAlignmentMetric",
    "get_metric",
    
    # Training and evaluation
    "train_model",
    "evaluate_model",
    "load_checkpoint",
    
    # Dropout functionality
    "progressive_dropout",
    "eigenvector_dropout",
    
    # Utilities
    "setup_logging",
    "timer",
    "debug",
    "to_numpy",
    "to_tensor",
    "check_iterable",
    "ensure_device",
    
    # Legacy components (backward compatibility)
    "train",
    "evaluate",
    "timed",
] 