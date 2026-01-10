"""
Infrastructure module for the alignment framework.

This module provides utilities for:
- Distributed computing (multi-GPU training)
- Storage (checkpointing, logging, job directories)
- GPU optimization (accelerated metric computations)
- JIT compilation (optimized metric functions)

USAGE STATUS:
- Storage (checkpoint, logging, job_directory): ACTIVELY USED
- Configuration: See alignment.configs for the main config system
- Computing (distributed, GPU, JIT): AVAILABLE but not currently integrated
  These are optimized implementations ready for future performance improvements.

Example:
    >>> from alignment.infrastructure import save_checkpoint, load_checkpoint
    >>> from alignment.infrastructure import setup_logging, get_logger
    >>> from alignment.infrastructure.storage import create_job_directory
"""

# Computing infrastructure
from .computing import (  # Distributed; GPU optimization; JIT compilation
    GPUAcceleratedMetrics,
    JITMutualInformation,
    JITNodeCorrelation,
    JITRayleighQuotient,
    cleanup_distributed,
    create_jit_metric,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)

# Storage infrastructure
from .storage import get_logger  # Checkpointing; Logging
from .storage import MetricLogger, load_checkpoint, log_metrics, save_checkpoint, save_model_for_inference, setup_logging

__all__ = [
    # Distributed computing
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "GPUAcceleratedMetrics",
    "JITRayleighQuotient",
    "JITMutualInformation",
    "JITNodeCorrelation",
    "create_jit_metric",
    # Storage
    "save_checkpoint",
    "load_checkpoint",
    "save_model_for_inference",
    "setup_logging",
    "get_logger",
    "log_metrics",
    "MetricLogger",
]
