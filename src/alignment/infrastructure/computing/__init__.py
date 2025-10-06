"""Computing infrastructure for the alignment framework."""

from .distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)

# Import from optimized submodule
from .optimized.gpu import GPUAcceleratedMetrics
from .optimized.jit import (
    JITMutualInformation,
    JITNodeCorrelation,
    JITRayleighQuotient,
    create_jit_metric,
)

__all__ = [
    # Distributed computing
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "get_world_size",
    "get_rank",
    # GPU optimization
    "GPUAcceleratedMetrics",
    # JIT compilation
    "JITRayleighQuotient",
    "JITMutualInformation",
    "JITNodeCorrelation",
    "create_jit_metric",
]
