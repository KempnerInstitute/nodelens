"""
Metrics for measuring neural network alignment, redundancy, and synergy.

=============================================================================
METRIC TAXONOMY (from alignment_notes/main.tex and vision_synergy_icml.tex)
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

4. SYNERGY METRICS (PID with MMI redundancy)
   - gaussian_pid_synergy_mmi: Target-conditional synergy
     S_MMI(Z; Y_i, Y_j) = I(Z; [Y_i,Y_j]) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j))
     
   - synergy_gaussian_mmi: Alternative implementation (same formula)

5. COMPOSITE SCORE (for pruning)
   Score(i) = α * log RQ(w_i) + β * I(Z; Y_i) + γ * S(i) - δ * R(i)
   
   High RQ + High target MI + High synergy + Low redundancy = Important neuron

6. ACTIVATION STATISTICS
   - activation_l2_norm: ||Y_i||_2 averaged over batch
   - activation_variance: Var(Y_i) over batch
   
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


# For convenience, expose the registry and functions
__all__ = [
    "METRIC_REGISTRY", 
    "get_metric", 
    "list_metrics",
    "get_recommended_metrics",
    "get_metric_category",
]
