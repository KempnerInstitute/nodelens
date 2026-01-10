"""
Metrics for measuring neural network alignment, redundancy, and synergy.

=============================================================================
METRIC TAXONOMY (from drafts/alignment_notes/alignment_red.tex)
=============================================================================

1. ALIGNMENT METRICS (Rayleigh Quotient based)
   - rayleigh_quotient (RQ): Measures alignment with input covariance
     RQ(w) = w^T Σ_X w / w^T w
     
   - conditional_rayleigh_quotient: Class-conditioned RQ
     RQ(w; Σ_{X|y}) for each class y, then averaged
     
   - delta_rq: Difference between unconditional and conditional RQ
     Δ_RQ(w) = RQ(w; Σ_X) - E_y[RQ(w; Σ_{X|y})]

2. MUTUAL INFORMATION METRICS (Gaussian approximation)
   - gaussian_mi_analytic: MI between inputs and outputs (RQ-related)
     I(X; y) = 0.5 * log(1 + w^T Σ_X w / σ_n^2)
     For linear-Gaussian: log RQ is a proxy for MI
     
   - mi_about_class: I(Z; Y) - MI between activations and class labels
     Uses Gaussian conditional variance formula

3. REDUNDANCY METRICS (Pairwise Gaussian)
   - pairwise_redundancy_gaussian: Target-free redundancy between neurons
     I(Y_i; Y_j) = -0.5 * log(1 - ρ²)
     where ρ = (w_i^T Σ_X w_j) / sqrt((w_i^T Σ_X w_i)(w_j^T Σ_X w_j))
     
   - average_redundancy: Per-neuron redundancy averaged over partners
     R(i) = (1/K) Σ_j I(Y_i; Y_j)
     
   - halo_redundancy: Redundancy analysis by neuron groups (halo vs non-halo)
   
   - cross_layer_redundancy: Redundancy with previous layer neurons
     R(Y_i^l || Y^{l-1}) = mean_j I(Y_i^l; Y_j^{l-1})

4. SYNERGY METRICS (PID with MMI redundancy)
   - gaussian_pid_synergy_mmi: Target-conditional synergy
     S_MMI(Z; Y_i, Y_j) = I(Z; [Y_i,Y_j]) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j))
     
   - synergy_gaussian_mmi: Alternative implementation (same formula)

5. COMPOSITE SCORE (for pruning)
   Score(i) = α * log RQ(w_i) + β * I(Z; Y_i) + γ * S(i) - δ * R(i)
   
   High RQ + High target MI + High synergy + Low redundancy = Important neuron
   
   Extended with cross-layer term:
   Score(Y_i^l) = α·RQ + β·MI - γ·R_within - δ·R_cross

6. ACTIVATION STATISTICS
   - activation_l2_norm: ||Y_i||_2 averaged over batch
   - activation_variance: Var(Y_i) over batch

7. MULTI-SUPERNODE METRICS (Extension)
   - multi_supernode: Cluster supernodes into k groups
   - multi_supernode_importance: Importance based on cluster membership

8. CROSS-LAYER METRICS (SCAR-aligned Extension)
   - cross_layer_redundancy: Downstream importance (how much next layer depends on each neuron)
   - cross_layer_importance: RQ + Downstream_Importance - Redundancy (SCAR logic)
   - layer_transition_efficiency: Downstream_Importance / (1 + Redundancy)
   
=============================================================================
METRIC CONFIGURATION OPTIONS
=============================================================================

All metrics share these common options (via BaseMetric):
  - force_cpu_for_large_ops: bool (default: True) - Move to CPU for large tensors
  - cpu_threshold: int (default: 1e8) - Element count threshold for CPU fallback

CNN Input Handling Modes (for convolutional layers):
  Metrics handle 4D inputs [B, C, H, W] differently. The recommended approach
  depends on the metric and use case:
  
  1. "unfold" - Unfold spatial dims into patches [B*P, C*K*K]
     - Most accurate for RQ/covariance metrics
     - Memory: O(B * P * C * K^2) where P = H*W patches
     - Use for: RQ, MI (when weights have kernel shape)
     
  2. "spatial" - Reshape [B, C, H, W] -> [B*H*W, C]  
     - Treats each spatial location as a sample
     - Preserves spatial variance information
     - Memory: O(B * H * W * C)
     - Use for: Fast covariance estimation
     
  3. "gap" (Global Average Pooling) - [B, C, H, W] -> [B, C]
     - Fastest, lowest memory
     - Loses spatial information
     - Memory: O(B * C)
     - Use for: Quick experiments, early conv layers
     
  4. "channel" - Per-channel statistics only
     - Aggregates over spatial dims
     - Memory: O(C)
     - Use for: Activation magnitude metrics

INDIVIDUAL METRIC OPTIONS:
--------------------------

rayleigh_quotient:
  relative: bool = True          # Normalize by trace(Σ) for relative alignment
  min_samples: int = 2           # Minimum samples for covariance
  scale_by_norm: bool = False    # Scale covariance by Frobenius norm
  regularization: float = 1e-6   # Diagonal regularization for stability

rq_fast (FastRayleighQuotient):
  relative: bool = True          # Same as RQ
  regularization: float = 1e-6   # Same as RQ
  # Uses GAP for CNN inputs - 10-100x faster than patchwise

rq_spatial (SpatialRayleighQuotient):
  # Same options as RQ
  # Uses spatial reshaping for CNN inputs

gaussian_mi_analytic:
  expansion_order: int = 2       # Edgeworth expansion order (0=pure Gaussian)
  noise_std: float = 0.1         # Assumed noise standard deviation
  regularization: float = 1e-6   # Covariance regularization
  per_neuron: bool = True        # Per-neuron MI (True) or joint MI (False)

pairwise_redundancy_gaussian:
  num_pairs: int = 10            # Partners to sample per neuron
  sampling_strategy: str = "random"  # "random", "nearest", "all"
  mode: str = "output_based"     # "output_based" (fast) or "covariance_based"
  regularization: float = 1e-6   # For covariance_based mode

average_redundancy:
  min_samples: int = 2           # Minimum samples
  use_correlation: bool = True   # Use correlation (True) or covariance (False)

mi_about_class:
  method: str = "gaussian"       # "gaussian" or "binning"
  bins: int = 10                 # For binning method
  min_samples_per_class: int = 5 # Minimum samples per class

conditional_rayleigh_quotient:
  relative: bool = True          # Same as RQ
  min_samples: int = 2           # Minimum samples per class
  regularization: float = 1e-6   # Same as RQ
  return_delta: bool = False     # Return ΔRQ = RQ_uncond - RQ_cond

activation_l2_norm:
  aggregate_method: str = "l2"   # "l2", "mean", "max"
  use_absolute: bool = True      # Take absolute value before aggregation

activation_outlier_index:
  quantile: float = 0.999        # High percentile for outlier detection
  eps: float = 1e-6              # Numerical stability

composite_importance:
  weights: dict                  # {metric_name: weight} - negative for penalties
  normalize_components: bool = True  # Normalize to [0, 1] before combining
  log_transform_rq: bool = True  # Apply log to RQ

cross_layer_importance:
  rq_weight: float = 0.25        # Weight for RQ (α)
  downstream_weight: float = 0.35    # Weight for downstream importance (β)
  within_redundancy_weight: float = 0.25  # Penalty for redundancy (γ)
  activation_weight: float = 0.15     # Weight for activation magnitude
  normalize: bool = True         # Normalize components to [0, 1]
  max_refs: int = 512            # Max reference neurons for efficiency

halo_redundancy:
  supernode_fraction: float = 0.01   # Top fraction as supernodes
  halo_fraction: float = 0.10        # Fraction of non-supernodes as halo
  max_samples: int = 1000            # Max activation samples
  max_pairs_per_group: int = 500     # Max pairs for efficiency

=============================================================================
"""

from ..core.registry import METRIC_REGISTRY

# Import all metric modules to register them
from . import information, rayleigh, similarity, spectral, task_specific
from .information import gaussian_pid  # Register gaussian PID synergy
from .information import pairwise_gaussian  # Ensure registration side-effects

# Import conditional metrics (class-conditioned versions)
from . import conditional_metrics  # Register conditional RQ, MI about class, etc.

# Import composite metrics (combinations for pruning)
from . import composite  # Register composite_importance, alignment_minus_redundancy, etc.

# Import halo and multi-supernode metrics (extensions)
from . import halo_redundancy  # Register halo_redundancy
from . import multi_supernode  # Register multi_supernode, multi_supernode_importance
from . import cross_layer  # Register cross_layer_redundancy, cross_layer_importance, layer_transition_efficiency


def get_metric(name: str, **kwargs):
    """
    Get a metric instance by name.

    Args:
        name: Name of the metric (see module docstring for taxonomy)
        **kwargs: Parameters to pass to metric constructor

    Returns:
        Instantiated metric object
        
    Recommended metrics for pruning (from paper):
        - rayleigh_quotient: Alignment with input covariance
        - gaussian_mi_analytic: MI directly related to RQ
        - pairwise_redundancy_gaussian: Target-free redundancy
        - gaussian_pid_synergy_mmi: Target-conditional synergy
        - mi_about_class: MI between activations and class labels
    """
    return METRIC_REGISTRY.create(name, **kwargs)


def list_metrics():
    """
    List all available metrics.

    Returns:
        List of metric names
    """
    return METRIC_REGISTRY.list()


def get_recommended_metrics():
    """
    Get the recommended core metrics for alignment analysis and pruning.
    
    Based on the analytical framework in the alignment notes:
    1. rayleigh_quotient - Alignment proxy
    2. gaussian_mi_analytic - MI (RQ-related)
    3. pairwise_redundancy_gaussian - Redundancy
    4. gaussian_pid_synergy_mmi - Synergy
    5. mi_about_class - Class information
    
    Returns:
        List of recommended metric names
    """
    return [
        "rayleigh_quotient",
        "gaussian_mi_analytic", 
        "pairwise_redundancy_gaussian",
        "gaussian_pid_synergy_mmi",
        "mi_about_class",
    ]


def get_extended_metrics():
    """
    Get extended metrics for advanced analysis.
    
    These include:
    1. halo_redundancy - Redundancy by neuron groups
    2. multi_supernode - Cluster-based supernode analysis
    3. cross_layer_redundancy - Inter-layer redundancy
    4. cross_layer_importance - Combined score with cross-layer terms
    5. layer_transition_efficiency - New information per layer
    
    Returns:
        List of extended metric names
    """
    return [
        "halo_redundancy",
        "multi_supernode",
        "multi_supernode_importance",
        "cross_layer_redundancy",
        "cross_layer_importance",
        "layer_transition_efficiency",
    ]


def get_metric_category(name: str) -> str:
    """
    Get the category of a metric.
    
    Args:
        name: Metric name
        
    Returns:
        Category string: 'alignment', 'mi', 'redundancy', 'synergy', 'activation', 'other'
    """
    name_lower = name.lower()
    
    if 'rayleigh' in name_lower or 'rq' in name_lower:
        return 'alignment'
    elif 'synergy' in name_lower or 'pid' in name_lower:
        return 'synergy'
    elif 'redundancy' in name_lower:
        return 'redundancy'
    elif 'mi' in name_lower or 'mutual' in name_lower or 'information' in name_lower:
        return 'mi'
    elif 'activation' in name_lower or 'norm' in name_lower or 'variance' in name_lower:
        return 'activation'
    else:
        return 'other'


def get_optimization_status() -> dict:
    """
    Check availability of metric optimizations (JIT, GPU).
    
    Returns:
        Dictionary with optimization availability status:
        {
            'jit_available': bool,
            'gpu_available': bool,
            'cuda_available': bool,
            'available_jit_functions': list,
            'available_gpu_functions': list,
        }
    """
    import torch
    
    status = {
        'jit_available': False,
        'gpu_available': False,
        'cuda_available': torch.cuda.is_available(),
        'available_jit_functions': [],
        'available_gpu_functions': [],
    }
    
    # Check JIT functions
    try:
        from ..infrastructure.computing.optimized.jit import (
            compute_rayleigh_quotient_jit,
            compute_mutual_information_gaussian_jit,
            compute_node_correlation_jit,
        )
        status['jit_available'] = True
        status['available_jit_functions'] = [
            'rayleigh_quotient',
            'mutual_information',
            'node_correlation',
            'cosine_similarity',
            'eigenvalue_entropy',
            'spectral_norm',
        ]
    except ImportError:
        pass
    
    # Check GPU functions
    try:
        from ..infrastructure.computing.optimized.gpu import (
            gpu_histogram1d,
            gpu_mutual_information,
            GPUAcceleratedMetrics,
        )
        status['gpu_available'] = True
        status['available_gpu_functions'] = [
            'histogram1d',
            'histogram2d', 
            'mutual_information',
            'entropy',
            'conditional_entropy',
            'fast_covariance',
            'fast_correlation',
        ]
    except ImportError:
        pass
    
    return status


def get_metric_with_optimizations(
    name: str,
    use_jit: bool = False,
    use_gpu_acceleration: bool = False,
    **kwargs
):
    """
    Get a metric instance with optimization options.
    
    Args:
        name: Name of the metric
        use_jit: Enable JIT-compiled computations (20-50% faster)
        use_gpu_acceleration: Enable GPU-accelerated functions
        **kwargs: Additional metric parameters
        
    Returns:
        Instantiated metric object with optimizations enabled
        
    Example:
        >>> metric = get_metric_with_optimizations(
        ...     "rayleigh_quotient",
        ...     use_jit=True,
        ...     relative=True
        ... )
    """
    return METRIC_REGISTRY.create(
        name,
        use_jit=use_jit,
        use_gpu_acceleration=use_gpu_acceleration,
        **kwargs
    )


# For convenience, expose the registry and functions
__all__ = [
    "METRIC_REGISTRY", 
    "get_metric", 
    "list_metrics",
    "get_recommended_metrics",
    "get_extended_metrics",
    "get_metric_category",
    "get_optimization_status",
    "get_metric_with_optimizations",
]
