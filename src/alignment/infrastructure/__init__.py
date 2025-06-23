"""
Infrastructure module for the alignment framework.

This module provides utilities for distributed computing, storage,
configuration management, and optimization.
"""

# Computing infrastructure
from .computing import (
    # Distributed
    DistributedConfig,
    DistributedTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    # GPU optimization
    GPUOptimizer,
    optimize_gpu_memory,
    get_gpu_memory_stats,
    # JIT compilation
    JITCompiler,
    compile_model,
    optimize_trace,
)

# Storage infrastructure
from .storage import (
    # Checkpointing
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
    # Logging
    setup_logging,
    get_logger,
    log_metrics,
    log_experiment_config,
)

# Configuration infrastructure
from .configuration import (
    Config,
    ExperimentConfig,
    MetricConfig,
    ModelConfig,
    DataConfig,
    load_config,
    save_config,
    merge_configs,
    validate_config,
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