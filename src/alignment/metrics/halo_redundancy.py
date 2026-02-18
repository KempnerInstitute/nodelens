"""
Halo Redundancy Analysis Module.

Computes and visualizes redundancy patterns within and between halo/non-halo groups.

Theory (Gaussian mutual information):
- Pairwise redundancy: I(Y_i; Y_j) = -0.5 * log(1 - ρ²)
- Per-neuron redundancy: R(Y_i) = mean over neighbors of I(Y_i; Y_j)
- Halo = neurons with high connectivity to supernodes
- This analysis validates whether halo membership correlates with redundancy

The three redundancy groups are:
1. Within-Halo: R(halo_i, halo_j) - redundancy among halo neurons
2. Within-Non-Halo: R(non_halo_i, non_halo_j) - redundancy among non-halo neurons
3. Cross-Group: R(halo_i, non_halo_j) - redundancy between groups
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@dataclass
class HaloRedundancyResult:
    """Results from halo redundancy analysis for a single layer."""

    layer_idx: int
    layer_name: str

    # Group sizes
    num_supernodes: int
    num_halo: int
    num_non_halo: int

    # Within-Halo redundancy
    halo_halo_mean: float
    halo_halo_std: float
    halo_halo_median: float

    # Within-Non-Halo redundancy
    non_halo_mean: float
    non_halo_std: float
    non_halo_median: float

    # Cross-group redundancy
    cross_group_mean: float
    cross_group_std: float
    cross_group_median: float

    # Raw values for plotting (sampled)
    halo_halo_values: Optional[np.ndarray] = None
    non_halo_values: Optional[np.ndarray] = None
    cross_group_values: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "num_supernodes": self.num_supernodes,
            "num_halo": self.num_halo,
            "num_non_halo": self.num_non_halo,
            "halo_halo": {
                "mean": self.halo_halo_mean,
                "std": self.halo_halo_std,
                "median": self.halo_halo_median,
            },
            "non_halo_non_halo": {
                "mean": self.non_halo_mean,
                "std": self.non_halo_std,
                "median": self.non_halo_median,
            },
            "cross_group": {
                "mean": self.cross_group_mean,
                "std": self.cross_group_std,
                "median": self.cross_group_median,
            },
        }


@dataclass
class LayerDepthAnalysis:
    """Analysis results across all layers (by depth)."""

    layers: List[int]
    layer_names: List[str]

    # Per-layer means
    halo_halo_by_layer: List[float]
    non_halo_by_layer: List[float]
    cross_group_by_layer: List[float]

    # Per-layer stds
    halo_halo_std_by_layer: List[float]
    non_halo_std_by_layer: List[float]
    cross_group_std_by_layer: List[float]

    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "layers": self.layers,
            "layer_names": self.layer_names,
            "halo_halo_by_layer": self.halo_halo_by_layer,
            "non_halo_by_layer": self.non_halo_by_layer,
            "cross_group_by_layer": self.cross_group_by_layer,
            "halo_halo_std_by_layer": self.halo_halo_std_by_layer,
            "non_halo_std_by_layer": self.non_halo_std_by_layer,
            "cross_group_std_by_layer": self.cross_group_std_by_layer,
            "summary": self.summary,
        }


def compute_correlation_matrix(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise |correlation| matrix.

    Args:
        activations: [batch_size, num_neurons]

    Returns:
        Absolute correlation matrix [num_neurons, num_neurons]
    """
    # Center
    centered = activations - activations.mean(dim=0, keepdim=True)
    # Normalize
    std = centered.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    normalized = centered / std
    # Correlation
    corr = (normalized.T @ normalized) / max(1, activations.shape[0] - 1)
    corr = torch.clamp(corr, -1, 1)
    return torch.abs(corr)


def correlation_to_redundancy(corr: torch.Tensor) -> torch.Tensor:
    """
    Convert correlation to redundancy using Gaussian MI formula.

    Theory (Gaussian mutual information):
        I(Y_i; Y_j) = -0.5 * log(1 - ρ²)

    This is the mutual information between jointly Gaussian variables.

    Args:
        corr: Correlation values (can be matrix or 1D)

    Returns:
        Redundancy (mutual information) values
    """
    rho_sq = corr**2
    rho_sq = torch.clamp(rho_sq, 0, 0.999999)  # Avoid log(0)
    redundancy = -0.5 * torch.log(1 - rho_sq)
    return redundancy


def identify_halo_groups(
    weights: torch.Tensor,
    supernode_fraction: float = 0.01,
    halo_fraction: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Identify supernode, halo, and non-halo neurons based on weight magnitude.

    Args:
        weights: Layer weights [hidden_dim, intermediate_dim] (down_proj for LLMs)
        supernode_fraction: Top fraction to consider as supernodes
        halo_fraction: Fraction of non-supernodes to consider as halo

    Returns:
        Tuple of (supernode_indices, halo_indices, non_halo_indices)
    """
    # Compute neuron magnitude (L2 norm of weights)
    if weights.ndim == 2:
        neuron_magnitude = weights.abs().sum(dim=0)  # [intermediate_dim]
    else:
        neuron_magnitude = weights.abs().flatten()

    num_neurons = len(neuron_magnitude)

    # Step 1: Identify supernodes (top by magnitude)
    num_supernodes = max(1, int(supernode_fraction * num_neurons))
    _, supernode_indices = torch.topk(neuron_magnitude, num_supernodes)
    supernode_mask = torch.zeros(num_neurons, dtype=torch.bool, device=weights.device)
    supernode_mask[supernode_indices] = True

    # Step 2: Define halo from non-supernodes (top by magnitude among non-supernodes)
    non_supernode_indices = (~supernode_mask).nonzero(as_tuple=True)[0]
    non_supernode_magnitude = neuron_magnitude[non_supernode_indices]

    num_halo = max(1, int(halo_fraction * len(non_supernode_indices)))
    _, halo_relative_indices = torch.topk(non_supernode_magnitude, num_halo)
    halo_indices = non_supernode_indices[halo_relative_indices]

    halo_mask = torch.zeros(num_neurons, dtype=torch.bool, device=weights.device)
    halo_mask[halo_indices] = True

    # Step 3: Non-halo = not supernode and not halo
    non_halo_mask = ~supernode_mask & ~halo_mask
    non_halo_indices = non_halo_mask.nonzero(as_tuple=True)[0]

    return supernode_indices, halo_indices, non_halo_indices


@register_metric("halo_redundancy")
class HaloRedundancy(BaseMetric):
    """
    Compute redundancy within halo, within non-halo, and between groups.

    This metric analyzes the information structure of halo vs non-halo neurons,
    helping to validate whether halo membership correlates with redundancy.

    Theory (Gaussian mutual information):
        Redundancy: I(Y_i; Y_j) = -0.5 * log(1 - ρ²)

    If halo neurons are indeed "echo chambers" of supernodes, we expect:
        - High within-halo redundancy
        - Lower cross-group redundancy
        - Variable within-non-halo redundancy
    """

    def __init__(
        self,
        supernode_fraction: float = 0.01,
        halo_fraction: float = 0.10,
        max_samples: int = 1000,
        max_pairs_per_group: int = 500,
        store_values: bool = True,
        **config: Any,
    ):
        """
        Initialize halo redundancy metric.

        Args:
            supernode_fraction: Top fraction to consider as supernodes
            halo_fraction: Fraction of non-supernodes to consider as halo
            max_samples: Maximum activation samples to use
            max_pairs_per_group: Maximum pairs to sample per group for efficiency
            store_values: Whether to store raw values for plotting
        """
        super().__init__(**config)
        self.supernode_fraction = supernode_fraction
        self.halo_fraction = halo_fraction
        self.max_samples = max_samples
        self.max_pairs_per_group = max_pairs_per_group
        self.store_values = store_values

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return True  # Needs activations (outputs of intermediate layer)

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        layer_name: str = "",
        **kwargs: Any,
    ) -> HaloRedundancyResult:
        """
        Compute halo redundancy analysis for a layer.

        Args:
            inputs: Not used
            weights: Layer weights for halo identification
            outputs: Activations [batch_size, num_neurons]
            layer_idx: Layer index
            layer_name: Layer name

        Returns:
            HaloRedundancyResult with all statistics
        """
        if weights is None or outputs is None:
            raise ValueError("HaloRedundancy requires weights and outputs (activations)")

        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        # Sample for efficiency
        if outputs.shape[0] > self.max_samples:
            indices = torch.randperm(outputs.shape[0], device=outputs.device)[: self.max_samples]
            outputs = outputs[indices]

        # Identify groups
        supernode_idx, halo_idx, non_halo_idx = identify_halo_groups(weights, self.supernode_fraction, self.halo_fraction)

        # Compute correlation matrix
        corr_matrix = compute_correlation_matrix(outputs.float())
        corr_matrix.fill_diagonal_(0)  # Exclude self-correlations

        # Convert to redundancy
        redundancy_matrix = correlation_to_redundancy(corr_matrix)

        # Sample indices for efficiency
        halo_idx_np = halo_idx.cpu().numpy()
        non_halo_idx_np = non_halo_idx.cpu().numpy()

        if len(halo_idx_np) > self.max_pairs_per_group:
            halo_idx_np = np.random.choice(halo_idx_np, self.max_pairs_per_group, replace=False)
        if len(non_halo_idx_np) > self.max_pairs_per_group:
            non_halo_idx_np = np.random.choice(non_halo_idx_np, self.max_pairs_per_group, replace=False)

        # Extract redundancy values for each group
        # Halo-Halo
        halo_halo_red = redundancy_matrix[np.ix_(halo_idx_np, halo_idx_np)]
        upper_tri_mask = torch.triu(torch.ones_like(halo_halo_red), diagonal=1).bool()
        halo_halo_values = halo_halo_red[upper_tri_mask].cpu().numpy()

        # Non-halo - Non-halo
        non_halo_red = redundancy_matrix[np.ix_(non_halo_idx_np, non_halo_idx_np)]
        upper_tri_mask_nh = torch.triu(torch.ones_like(non_halo_red), diagonal=1).bool()
        non_halo_values = non_halo_red[upper_tri_mask_nh].cpu().numpy()

        # Cross-group
        cross_red = redundancy_matrix[np.ix_(halo_idx_np, non_halo_idx_np)]
        cross_values = cross_red.flatten().cpu().numpy()

        # Compute statistics
        result = HaloRedundancyResult(
            layer_idx=layer_idx,
            layer_name=layer_name,
            num_supernodes=len(supernode_idx),
            num_halo=len(halo_idx),
            num_non_halo=len(non_halo_idx),
            halo_halo_mean=float(np.mean(halo_halo_values)) if len(halo_halo_values) > 0 else 0.0,
            halo_halo_std=float(np.std(halo_halo_values)) if len(halo_halo_values) > 0 else 0.0,
            halo_halo_median=float(np.median(halo_halo_values)) if len(halo_halo_values) > 0 else 0.0,
            non_halo_mean=float(np.mean(non_halo_values)) if len(non_halo_values) > 0 else 0.0,
            non_halo_std=float(np.std(non_halo_values)) if len(non_halo_values) > 0 else 0.0,
            non_halo_median=float(np.median(non_halo_values)) if len(non_halo_values) > 0 else 0.0,
            cross_group_mean=float(np.mean(cross_values)) if len(cross_values) > 0 else 0.0,
            cross_group_std=float(np.std(cross_values)) if len(cross_values) > 0 else 0.0,
            cross_group_median=float(np.median(cross_values)) if len(cross_values) > 0 else 0.0,
            halo_halo_values=halo_halo_values[:5000] if self.store_values else None,
            non_halo_values=non_halo_values[:5000] if self.store_values else None,
            cross_group_values=cross_values[:5000] if self.store_values else None,
        )

        return result


def analyze_layers_depth(
    results: List[HaloRedundancyResult],
) -> LayerDepthAnalysis:
    """
    Aggregate halo redundancy results across layers for depth analysis.

    Args:
        results: List of HaloRedundancyResult for each layer

    Returns:
        LayerDepthAnalysis with per-layer statistics
    """
    layers = [r.layer_idx for r in results]
    layer_names = [r.layer_name for r in results]

    halo_halo_by_layer = [r.halo_halo_mean for r in results]
    non_halo_by_layer = [r.non_halo_mean for r in results]
    cross_group_by_layer = [r.cross_group_mean for r in results]

    halo_halo_std_by_layer = [r.halo_halo_std for r in results]
    non_halo_std_by_layer = [r.non_halo_std for r in results]
    cross_group_std_by_layer = [r.cross_group_std for r in results]

    # Compute summary
    summary = {
        "avg_halo_halo": float(np.mean(halo_halo_by_layer)),
        "avg_non_halo": float(np.mean(non_halo_by_layer)),
        "avg_cross_group": float(np.mean(cross_group_by_layer)),
        "halo_vs_non_halo_ratio": float(np.mean(halo_halo_by_layer)) / max(float(np.mean(non_halo_by_layer)), 1e-8),
        "interpretation": _interpret_results(halo_halo_by_layer, non_halo_by_layer, cross_group_by_layer),
    }

    return LayerDepthAnalysis(
        layers=layers,
        layer_names=layer_names,
        halo_halo_by_layer=halo_halo_by_layer,
        non_halo_by_layer=non_halo_by_layer,
        cross_group_by_layer=cross_group_by_layer,
        halo_halo_std_by_layer=halo_halo_std_by_layer,
        non_halo_std_by_layer=non_halo_std_by_layer,
        cross_group_std_by_layer=cross_group_std_by_layer,
        summary=summary,
    )


def _interpret_results(
    halo_halo: List[float],
    non_halo: List[float],
    cross_group: List[float],
) -> Dict[str, str]:
    """Generate interpretation of results."""
    avg_halo = np.mean(halo_halo)
    avg_non_halo = np.mean(non_halo)
    avg_cross = np.mean(cross_group)

    interpretation = {}

    # Halo vs Non-halo comparison
    if avg_halo > avg_non_halo * 1.2:
        interpretation["halo_vs_non_halo"] = "VALID: Halo neurons are MORE redundant than non-halo"
    elif avg_halo < avg_non_halo * 0.8:
        interpretation["halo_vs_non_halo"] = "REVISE: Non-halo neurons are MORE redundant - consider revising halo definition"
    else:
        interpretation["halo_vs_non_halo"] = "SIMILAR: Similar redundancy in both groups - may need different selection criteria"

    # Cross-group analysis
    if avg_cross < avg_halo * 0.8:
        interpretation["cross_group"] = "SEPARATED: Cross-group correlation LOW - groups carry different information"
    else:
        interpretation["cross_group"] = "MIXED: Cross-group correlation similar - information not well separated by halo"

    # Echo chamber detection
    if avg_halo > avg_non_halo * 1.1 and avg_cross < avg_halo * 0.9:
        interpretation["echo_chamber"] = "DETECTED: Halo is an 'echo chamber' - safe to prune redundant halo members"
    else:
        interpretation["echo_chamber"] = "NOT_DETECTED: Halo structure not clearly separated"

    return interpretation
