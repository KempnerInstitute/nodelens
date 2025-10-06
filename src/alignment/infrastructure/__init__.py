"""
Infrastructure module for the alignment framework.

This module provides utilities for distributed computing, storage,
configuration management, and optimization.
"""

# Computing infrastructure
from .computing import (  # Distributed; GPU optimization; JIT compilation
    DistributedConfig,
    DistributedTrainer,
    GPUOptimizer,
    JITCompiler,
    cleanup_distributed,
    compile_model,
    get_gpu_memory_stats,
    get_rank,
    get_world_size,
    is_main_process,
    optimize_gpu_memory,
    optimize_trace,
    setup_distributed,
)

# Configuration infrastructure
from .configuration import (
    Config,
    DataConfig,
    ExperimentConfig,
    MetricConfig,
    ModelConfig,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)

# Storage infrastructure
from .storage import (  # Checkpointing; Logging
    CheckpointManager,
    cleanup_old_checkpoints,
    get_logger,
    load_checkpoint,
    log_experiment_config,
    log_metrics,
    save_checkpoint,
    setup_logging,
)

__all__ = [
    # Computing
    'DistributedConfig',
    'DistributedTrainer',
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'get_world_size',
    'get_rank',
    'GPUOptimizer',
    'optimize_gpu_memory',
    'get_gpu_memory_stats',
    'JITCompiler',
    'compile_model',
    'optimize_trace',
    # Storage
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    'cleanup_old_checkpoints',
    'setup_logging',
    'get_logger',
    'log_metrics',
    'log_experiment_config',
    # Configuration
    'Config',
    'ExperimentConfig',
    'MetricConfig',
    'ModelConfig',
    'DataConfig',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config',
]
