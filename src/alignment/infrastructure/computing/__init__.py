"""Computing infrastructure for the alignment framework."""

from .distributed import (
    DistributedConfig,
    DistributedTrainer,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)

# Import from optimized submodule
from .optimized.gpu import GPUOptimizer, get_gpu_memory_stats, optimize_gpu_memory
from .optimized.jit import JITCompiler, compile_model, optimize_trace

__all__ = [
    # Distributed computing
    'DistributedConfig',
    'DistributedTrainer',
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'get_world_size',
    'get_rank',
    # GPU optimization
    'GPUOptimizer',
    'optimize_gpu_memory',
    'get_gpu_memory_stats',
    # JIT compilation
    'JITCompiler',
    'compile_model',
    'optimize_trace',
]
