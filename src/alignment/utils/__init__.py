"""
Utility functions and helpers for the alignment framework.
"""

from alignment.utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    barrier,
    reduce_tensor,
    gather_tensor,
    DistributedMetricComputer,
)
from alignment.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_model_for_inference,
)
from alignment.utils.logging import (
    setup_logging,
    get_logger,
    log_metrics,
)
from alignment.utils.config import (
    load_config,
    save_config,
    merge_configs,
    Config,
)

# Import batch processing utilities
try:
    from alignment.utils.batch_processing import (
        BatchMetricProcessor,
        StreamingMetricComputer,
        compute_metrics_parallel,
        batch_mutual_information,
    )
    _batch_processing_available = True
except ImportError:
    _batch_processing_available = False

# Import experiment tracking utilities
try:
    from alignment.utils.experiment_tracking import (
        ExperimentTracker,
        WandBTracker,
        TensorBoardTracker,
        MultiTracker,
        DummyTracker,
        create_tracker,
    )
    _tracking_available = True
except ImportError:
    _tracking_available = False

# Import optimized implementations
try:
    from alignment.utils.optimized import (
        # GPU functions
        gpu_histogram1d,
        gpu_histogram2d,
        gpu_mutual_information,
        gpu_entropy,
        gpu_conditional_entropy,
        GPUAcceleratedMetrics,
        # JIT functions
        JITRayleighQuotient,
        JITMutualInformation,
        JITNodeCorrelation,
        create_jit_metric,
    )
    _optimized_available = True
except ImportError:
    _optimized_available = False

__all__ = [
    # Distributed utilities
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'barrier',
    'reduce_tensor',
    'gather_tensor',
    'DistributedMetricComputer',
    # Checkpoint utilities
    'save_checkpoint',
    'load_checkpoint',
    'save_model_for_inference',
    # Logging utilities
    'setup_logging',
    'get_logger',
    'log_metrics',
    # Config utilities
    'load_config',
    'save_config',
    'merge_configs',
    'Config',
]

# Add new utilities if available
if _batch_processing_available:
    __all__.extend([
        'BatchMetricProcessor',
        'StreamingMetricComputer',
        'compute_metrics_parallel',
        'batch_mutual_information',
    ])

if _tracking_available:
    __all__.extend([
        'ExperimentTracker',
        'WandBTracker',
        'TensorBoardTracker',
        'MultiTracker',
        'DummyTracker',
        'create_tracker',
    ])

if _optimized_available:
    __all__.extend([
        # GPU functions
        'gpu_histogram1d',
        'gpu_histogram2d',
        'gpu_mutual_information',
        'gpu_entropy',
        'gpu_conditional_entropy',
        'GPUAcceleratedMetrics',
        # JIT functions
        'JITRayleighQuotient',
        'JITMutualInformation',
        'JITNodeCorrelation',
        'create_jit_metric',
    ]) 