"""Computing infrastructure for the alignment framework."""

from .distributed import (
    DistributedConfig,
    DistributedTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
)

# Import from optimized submodule
from .optimized.gpu import (
    GPUOptimizer,
    optimize_gpu_memory,
    get_gpu_memory_stats,
)
from .optimized.jit import (
    JITCompiler,
    compile_model,
    optimize_trace,
)

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