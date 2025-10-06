"""
Optimized implementations of alignment metric computations.

This module provides GPU-accelerated and JIT-compiled versions of common operations.
"""

# GPU-accelerated functions
from .gpu import (
    GPUAcceleratedMetrics,
    gpu_conditional_entropy,
    gpu_entropy,
    gpu_histogram1d,
    gpu_histogram2d,
    gpu_mutual_information,
)

# JIT-compiled functions
from .jit import (
    JITMutualInformation,
    JITNodeCorrelation,
    JITRayleighQuotient,
    benchmark_jit_vs_regular,
    compute_batch_histogram_jit,
    compute_cosine_similarity_matrix_jit,
    compute_eigenvalue_entropy_jit,
    compute_mutual_information_gaussian_jit,
    compute_node_correlation_jit,
    compute_rayleigh_quotient_jit,
    compute_spectral_norm_jit,
    create_jit_metric,
)

__all__ = [
    # GPU functions
    "gpu_histogram1d",
    "gpu_histogram2d",
    "gpu_mutual_information",
    "gpu_entropy",
    "gpu_conditional_entropy",
    "GPUAcceleratedMetrics",
    # JIT functions
    "compute_rayleigh_quotient_jit",
    "compute_cosine_similarity_matrix_jit",
    "compute_mutual_information_gaussian_jit",
    "compute_eigenvalue_entropy_jit",
    "compute_node_correlation_jit",
    "compute_spectral_norm_jit",
    "compute_batch_histogram_jit",
    "JITRayleighQuotient",
    "JITMutualInformation",
    "JITNodeCorrelation",
    "create_jit_metric",
    "benchmark_jit_vs_regular",
]
