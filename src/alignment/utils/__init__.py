"""Alignment utility functions and tools."""

# Import distributed utilities
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    all_reduce,
    DistributedMetricComputer
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    is_checkpoint_complete
)
from .logging import (
    setup_logging,
    get_logger,
    set_log_level
)
from .config import (
    load_config,
    save_config,
    validate_config,
    merge_configs,
    resolve_config_references,
    ConfigManager
)
from .batch_processing import (
    BatchMetricProcessor,
    compute_metrics_parallel,
    StreamingMetricComputer,
    create_metric_batches,
    aggregate_metric_results,
    batch_mutual_information,
    batch_rayleigh_quotient,
    batch_weight_similarity,
    batch_cka
)
from .experiment_tracking import (
    create_tracker,
    ExperimentTracker,
    WandBTracker,
    TensorBoardTracker,
    MultiTracker,
    DummyTracker,
    MetricLogger,
    ResultCache,
    ExperimentSummary
)

# From optimized submodule
from .optimized import (
    compute_batch_mi_cuda,
    batch_cov_cuda,
    batch_kl_divergence_cuda,
    batch_entropy_cuda,
    batch_cosine_similarity_cuda,
    batch_matrix_sqrt,
    batch_svd,
    batch_pca,
    batch_correlation,
    batch_histogram,
    jit_histogram_binning,
    jit_kernel_matrix,
    jit_entropy,
    jit_correlation
)

# Import pruning utilities
from .pruning import (
    PruningUtilities,
    PruningConfig,
    create_pruning_schedule,
    get_pruning_mask,
    apply_pruning,
    compute_pruning_statistics,
    AdaptivePruning,
    StructuredPruning,
    MagnitudePruner,
    GradientPruner,
    HessianPruner
)

__all__ = [
    # Distributed
    'setup_distributed',
    'cleanup_distributed',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'all_reduce',
    'DistributedMetricComputer',
    # Checkpoint
    'save_checkpoint',
    'load_checkpoint',
    'is_checkpoint_complete',
    # Logging
    'setup_logging',
    'get_logger',
    'set_log_level',
    # Config
    'load_config',
    'save_config',
    'validate_config',
    'merge_configs',
    'resolve_config_references',
    'ConfigManager',
    # Batch processing
    'BatchMetricProcessor',
    'compute_metrics_parallel',
    'StreamingMetricComputer',
    'create_metric_batches',
    'aggregate_metric_results',
    'batch_mutual_information',
    'batch_rayleigh_quotient',
    'batch_weight_similarity',
    'batch_cka',
    # Experiment tracking
    'create_tracker',
    'ExperimentTracker',
    'WandBTracker',
    'TensorBoardTracker',
    'MultiTracker',
    'DummyTracker',
    'MetricLogger',
    'ResultCache',
    'ExperimentSummary',
    # Optimized operations
    'compute_batch_mi_cuda',
    'batch_cov_cuda',
    'batch_kl_divergence_cuda',
    'batch_entropy_cuda',
    'batch_cosine_similarity_cuda',
    'batch_matrix_sqrt',
    'batch_svd',
    'batch_pca',
    'batch_correlation',
    'batch_histogram',
    'jit_histogram_binning',
    'jit_kernel_matrix',
    'jit_entropy',
    'jit_correlation',
    # Pruning
    'PruningUtilities',
    'PruningConfig',
    'create_pruning_schedule',
    'get_pruning_mask',
    'apply_pruning',
    'compute_pruning_statistics',
    'AdaptivePruning',
    'StructuredPruning',
    'MagnitudePruner',
    'GradientPruner',
    'HessianPruner'
] 