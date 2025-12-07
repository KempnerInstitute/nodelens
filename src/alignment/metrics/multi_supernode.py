"""
Multi-Supernode Clustering and Extended Metrics.

Extends the supernode concept from a single group to k distinct clusters,
computing separate halo and metrics for each supernode cluster.

This allows for:
1. Identifying multiple supernode communities
2. Computing per-cluster halo membership
3. Measuring redundancy within and between supernode clusters
4. More nuanced pruning based on cluster structure

Theory (extending alignment_notes):
- Instead of treating top k% as a single supernode group, we cluster them
- Each cluster represents a different "functional group" of important neurons
- Halo is defined relative to each cluster
- Redundancy patterns may differ between clusters
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..core.base import BaseMetric
from ..core.registry import register_metric

logger = logging.getLogger(__name__)


@dataclass
class SupernodeCluster:
    """Represents a single supernode cluster."""
    
    cluster_id: int
    indices: np.ndarray  # Indices of neurons in this cluster
    centroid: np.ndarray  # Cluster centroid in feature space
    
    # Halo for this cluster
    halo_indices: Optional[np.ndarray] = None
    
    # Metrics for this cluster
    mean_magnitude: float = 0.0
    mean_rq: float = 0.0
    internal_redundancy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "num_neurons": len(self.indices),
            "num_halo": len(self.halo_indices) if self.halo_indices is not None else 0,
            "mean_magnitude": self.mean_magnitude,
            "mean_rq": self.mean_rq,
            "internal_redundancy": self.internal_redundancy,
        }


@dataclass
class MultiSupernodeResult:
    """Results from multi-supernode clustering analysis."""
    
    layer_idx: int
    layer_name: str
    num_clusters: int
    total_supernodes: int
    
    # Per-cluster results
    clusters: List[SupernodeCluster]
    
    # Cross-cluster metrics
    between_cluster_redundancy: np.ndarray  # [k, k] matrix
    cluster_separation_score: float  # How well separated are clusters
    
    # Global metrics
    average_within_cluster_redundancy: float
    average_between_cluster_redundancy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "layer_name": self.layer_name,
            "num_clusters": self.num_clusters,
            "total_supernodes": self.total_supernodes,
            "clusters": [c.to_dict() for c in self.clusters],
            "cluster_separation_score": self.cluster_separation_score,
            "average_within_cluster_redundancy": self.average_within_cluster_redundancy,
            "average_between_cluster_redundancy": self.average_between_cluster_redundancy,
        }


def kmeans_clustering(
    features: torch.Tensor,
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-means clustering implementation in PyTorch.
    
    Args:
        features: Feature matrix [n_samples, n_features]
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Tuple of (cluster_assignments [n_samples], centroids [n_clusters, n_features])
    """
    n_samples, n_features = features.shape
    device = features.device
    
    # Initialize centroids using k-means++
    centroids = torch.zeros(n_clusters, n_features, device=device)
    
    # Pick first centroid randomly
    idx = torch.randint(0, n_samples, (1,), device=device)
    centroids[0] = features[idx]
    
    # Pick remaining centroids with probability proportional to distance
    for i in range(1, n_clusters):
        distances = torch.cdist(features, centroids[:i])
        min_distances = distances.min(dim=1).values
        probs = min_distances ** 2
        probs = probs / probs.sum()
        
        # Sample from distribution
        idx = torch.multinomial(probs, 1)
        centroids[i] = features[idx]
    
    # Iterative refinement
    for iteration in range(max_iter):
        # Assign points to nearest centroid
        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = features[mask].mean(dim=0)
            else:
                # Reinitialize empty cluster
                new_centroids[k] = features[torch.randint(0, n_samples, (1,), device=device)]
        
        # Check convergence
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        
        if shift < tol:
            break
    
    return assignments, centroids


def identify_supernode_clusters(
    weights: torch.Tensor,
    activations: Optional[torch.Tensor] = None,
    supernode_fraction: float = 0.05,
    n_clusters: int = 4,
    clustering_features: str = "weights",  # "weights", "activations", or "combined"
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """
    Identify supernodes and cluster them into k groups.
    
    Args:
        weights: Layer weights [hidden_dim, intermediate_dim] or [out, in]
        activations: Optional activations for clustering [batch, neurons]
        supernode_fraction: Fraction of neurons to consider as supernodes
        n_clusters: Number of supernode clusters
        clustering_features: What features to use for clustering
        
    Returns:
        Tuple of:
            - supernode_indices: All supernode indices
            - cluster_indices: List of indices per cluster
            - cluster_assignments: Cluster assignment for each supernode
    """
    # Compute neuron magnitude
    if weights.ndim == 2:
        neuron_magnitude = weights.abs().sum(dim=0)  # [num_neurons]
    else:
        neuron_magnitude = weights.abs().flatten()
    
    num_neurons = len(neuron_magnitude)
    
    # Identify supernodes (top by magnitude)
    num_supernodes = max(n_clusters, int(supernode_fraction * num_neurons))
    _, supernode_indices = torch.topk(neuron_magnitude, num_supernodes)
    
    # Get features for clustering
    if clustering_features == "weights" or activations is None:
        # Use weight patterns for clustering
        if weights.ndim == 2:
            cluster_features = weights[:, supernode_indices].T.float()  # [num_supernodes, hidden_dim]
        else:
            cluster_features = weights[supernode_indices].unsqueeze(1).float()
    elif clustering_features == "activations":
        # Use activation patterns
        cluster_features = activations[:, supernode_indices].T.float()  # [num_supernodes, batch]
    else:  # "combined"
        weight_features = weights[:, supernode_indices].T.float() if weights.ndim == 2 else weights[supernode_indices].unsqueeze(1).float()
        if activations is not None:
            act_features = activations[:, supernode_indices].T.float()
            # Normalize and concatenate
            weight_features = weight_features / (weight_features.norm(dim=1, keepdim=True) + 1e-8)
            act_features = act_features / (act_features.norm(dim=1, keepdim=True) + 1e-8)
            cluster_features = torch.cat([weight_features, act_features], dim=1)
        else:
            cluster_features = weight_features
    
    # Normalize features for clustering
    cluster_features = cluster_features / (cluster_features.norm(dim=1, keepdim=True) + 1e-8)
    
    # Cluster supernodes
    assignments, centroids = kmeans_clustering(cluster_features, n_clusters)
    
    # Group indices by cluster
    cluster_indices_list = []
    for k in range(n_clusters):
        mask = assignments == k
        cluster_idx = supernode_indices[mask]
        cluster_indices_list.append(cluster_idx)
    
    return supernode_indices, cluster_indices_list, assignments


def compute_cluster_halo(
    weights: torch.Tensor,
    cluster_indices: torch.Tensor,
    all_supernode_indices: torch.Tensor,
    halo_fraction: float = 0.10,
) -> torch.Tensor:
    """
    Compute halo neurons for a specific supernode cluster.
    
    Halo = neurons with high connection to this cluster's supernodes
    but not themselves supernodes.
    
    Args:
        weights: Layer weights
        cluster_indices: Indices of supernodes in this cluster
        all_supernode_indices: All supernode indices (to exclude)
        halo_fraction: Fraction of non-supernodes to consider as halo
        
    Returns:
        Indices of halo neurons for this cluster
    """
    num_neurons = weights.shape[1] if weights.ndim == 2 else weights.shape[0]
    device = weights.device
    
    # Create supernode mask
    supernode_mask = torch.zeros(num_neurons, dtype=torch.bool, device=device)
    supernode_mask[all_supernode_indices] = True
    
    # Compute connection strength to this cluster's supernodes
    if weights.ndim == 2:
        cluster_weights = weights[:, cluster_indices]  # [hidden_dim, cluster_size]
        # Connection = similarity to cluster pattern
        connection_strength = (weights.T @ cluster_weights.mean(dim=1)).abs()
    else:
        cluster_weights = weights[cluster_indices]
        connection_strength = weights.abs()
    
    # Consider only non-supernodes
    non_supernode_indices = (~supernode_mask).nonzero(as_tuple=True)[0]
    non_supernode_connection = connection_strength[non_supernode_indices]
    
    # Select top fraction as halo
    num_halo = max(1, int(halo_fraction * len(non_supernode_indices)))
    _, halo_relative_indices = torch.topk(non_supernode_connection, num_halo)
    halo_indices = non_supernode_indices[halo_relative_indices]
    
    return halo_indices


def compute_pairwise_redundancy(
    activations: torch.Tensor,
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    max_pairs: int = 500,
) -> float:
    """
    Compute average pairwise redundancy between two sets of neurons.
    
    Args:
        activations: [batch, neurons]
        indices1: First set of neuron indices
        indices2: Second set of neuron indices
        max_pairs: Maximum pairs to sample
        
    Returns:
        Average redundancy (MI in nats)
    """
    if len(indices1) == 0 or len(indices2) == 0:
        return 0.0
    
    # Sample if needed
    idx1 = indices1.cpu().numpy()
    idx2 = indices2.cpu().numpy()
    
    if len(idx1) > max_pairs:
        idx1 = np.random.choice(idx1, max_pairs, replace=False)
    if len(idx2) > max_pairs:
        idx2 = np.random.choice(idx2, max_pairs, replace=False)
    
    # Get activations
    act1 = activations[:, idx1].float()  # [batch, n1]
    act2 = activations[:, idx2].float()  # [batch, n2]
    
    # Normalize
    act1_norm = (act1 - act1.mean(dim=0, keepdim=True)) / (act1.std(dim=0, keepdim=True) + 1e-8)
    act2_norm = (act2 - act2.mean(dim=0, keepdim=True)) / (act2.std(dim=0, keepdim=True) + 1e-8)
    
    # Compute correlation matrix
    corr = (act1_norm.T @ act2_norm) / max(1, activations.shape[0] - 1)
    corr = torch.clamp(corr, -0.999, 0.999)
    
    # Convert to redundancy (MI)
    rho_sq = corr ** 2
    redundancy = -0.5 * torch.log(1 - rho_sq + 1e-8)
    
    # Handle same-set case (exclude diagonal)
    if torch.equal(torch.tensor(idx1), torch.tensor(idx2)):
        redundancy.fill_diagonal_(0)
        n = len(idx1)
        return float(redundancy.sum() / max(1, n * (n - 1)))
    
    return float(redundancy.mean())


@register_metric("multi_supernode")
class MultiSupernodeAnalysis(BaseMetric):
    """
    Multi-supernode clustering analysis with extended metrics.
    
    This extends the single-supernode concept to k clusters, enabling:
    - Identification of multiple functional groups
    - Per-cluster halo computation
    - Within and between cluster redundancy analysis
    
    Theory extension:
    Instead of treating all supernodes as one group, we identify k clusters
    that may represent different functional specializations. This provides
    more nuanced insights into network structure.
    """
    
    def __init__(
        self,
        supernode_fraction: float = 0.05,
        n_clusters: int = 4,
        halo_fraction: float = 0.10,
        clustering_features: str = "weights",
        max_pairs: int = 500,
        **config: Any,
    ):
        """
        Initialize multi-supernode analysis.
        
        Args:
            supernode_fraction: Fraction of neurons to consider as supernodes
            n_clusters: Number of supernode clusters
            halo_fraction: Fraction of non-supernodes to consider as halo
            clustering_features: "weights", "activations", or "combined"
            max_pairs: Maximum pairs for redundancy computation
        """
        super().__init__(**config)
        self.supernode_fraction = supernode_fraction
        self.n_clusters = n_clusters
        self.halo_fraction = halo_fraction
        self.clustering_features = clustering_features
        self.max_pairs = max_pairs
    
    @property
    def requires_inputs(self) -> bool:
        return False
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return True  # Needs activations
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        layer_name: str = "",
        **kwargs: Any,
    ) -> MultiSupernodeResult:
        """
        Perform multi-supernode clustering analysis.
        
        Args:
            inputs: Not used
            weights: Layer weights
            outputs: Activations [batch, neurons]
            layer_idx: Layer index
            layer_name: Layer name
            
        Returns:
            MultiSupernodeResult with all cluster information
        """
        if weights is None or outputs is None:
            raise ValueError("MultiSupernodeAnalysis requires weights and outputs")
        
        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        # Identify and cluster supernodes
        supernode_indices, cluster_indices_list, assignments = identify_supernode_clusters(
            weights=weights,
            activations=outputs,
            supernode_fraction=self.supernode_fraction,
            n_clusters=self.n_clusters,
            clustering_features=self.clustering_features,
        )
        
        # Compute per-cluster metrics
        clusters = []
        for k, cluster_idx in enumerate(cluster_indices_list):
            if len(cluster_idx) == 0:
                continue
            
            # Compute halo for this cluster
            halo_idx = compute_cluster_halo(
                weights, cluster_idx, supernode_indices, self.halo_fraction
            )
            
            # Compute internal redundancy
            internal_red = compute_pairwise_redundancy(
                outputs, cluster_idx, cluster_idx, self.max_pairs
            )
            
            # Compute mean magnitude
            neuron_mag = weights.abs().sum(dim=0) if weights.ndim == 2 else weights.abs()
            mean_mag = float(neuron_mag[cluster_idx].mean())
            
            cluster = SupernodeCluster(
                cluster_id=k,
                indices=cluster_idx.cpu().numpy(),
                centroid=weights[:, cluster_idx].mean(dim=1).cpu().numpy() if weights.ndim == 2 else weights[cluster_idx].mean(dim=0).cpu().numpy(),
                halo_indices=halo_idx.cpu().numpy(),
                mean_magnitude=mean_mag,
                internal_redundancy=internal_red,
            )
            clusters.append(cluster)
        
        # Compute between-cluster redundancy matrix
        n_actual_clusters = len(clusters)
        between_redundancy = np.zeros((n_actual_clusters, n_actual_clusters))
        
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i <= j:  # Upper triangle + diagonal
                    red = compute_pairwise_redundancy(
                        outputs,
                        torch.tensor(c1.indices, device=outputs.device),
                        torch.tensor(c2.indices, device=outputs.device),
                        self.max_pairs,
                    )
                    between_redundancy[i, j] = red
                    between_redundancy[j, i] = red
        
        # Compute summary statistics
        within_cluster_reds = [c.internal_redundancy for c in clusters]
        avg_within = float(np.mean(within_cluster_reds)) if within_cluster_reds else 0.0
        
        # Between-cluster: off-diagonal elements
        off_diag_mask = ~np.eye(n_actual_clusters, dtype=bool)
        if off_diag_mask.sum() > 0:
            avg_between = float(between_redundancy[off_diag_mask].mean())
        else:
            avg_between = 0.0
        
        # Cluster separation score: within / between ratio
        separation = avg_within / max(avg_between, 1e-8) if avg_between > 0 else float('inf')
        
        return MultiSupernodeResult(
            layer_idx=layer_idx,
            layer_name=layer_name,
            num_clusters=n_actual_clusters,
            total_supernodes=len(supernode_indices),
            clusters=clusters,
            between_cluster_redundancy=between_redundancy,
            cluster_separation_score=separation,
            average_within_cluster_redundancy=avg_within,
            average_between_cluster_redundancy=avg_between,
        )


@register_metric("multi_supernode_importance")
class MultiSupernodeImportance(BaseMetric):
    """
    Compute neuron importance scores based on multi-supernode structure.
    
    For each neuron, importance is computed as:
    - If supernode: high importance based on cluster membership
    - If halo: importance based on connection to clusters
    - If neither: lower importance, penalized by redundancy
    
    Score = base_importance * cluster_influence * (1 - redundancy_penalty)
    """
    
    def __init__(
        self,
        supernode_fraction: float = 0.05,
        n_clusters: int = 4,
        halo_fraction: float = 0.10,
        supernode_weight: float = 1.0,
        halo_weight: float = 0.7,
        regular_weight: float = 0.3,
        redundancy_penalty: float = 0.5,
        **config: Any,
    ):
        super().__init__(**config)
        self.supernode_fraction = supernode_fraction
        self.n_clusters = n_clusters
        self.halo_fraction = halo_fraction
        self.supernode_weight = supernode_weight
        self.halo_weight = halo_weight
        self.regular_weight = regular_weight
        self.redundancy_penalty = redundancy_penalty
    
    @property
    def requires_inputs(self) -> bool:
        return False
    
    @property
    def requires_weights(self) -> bool:
        return True
    
    @property
    def requires_outputs(self) -> bool:
        return True
    
    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute importance scores based on multi-supernode structure.
        
        Returns:
            Importance scores [num_neurons]
        """
        if weights is None or outputs is None:
            raise ValueError("MultiSupernodeImportance requires weights and outputs")
        
        # Flatten if needed
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        
        num_neurons = outputs.shape[1]
        device = outputs.device
        
        # Identify supernodes and clusters
        supernode_indices, cluster_indices_list, _ = identify_supernode_clusters(
            weights=weights,
            activations=outputs,
            supernode_fraction=self.supernode_fraction,
            n_clusters=self.n_clusters,
        )
        
        # Initialize scores with base magnitude
        neuron_mag = weights.abs().sum(dim=0) if weights.ndim == 2 else weights.abs()
        neuron_mag = neuron_mag / (neuron_mag.max() + 1e-8)  # Normalize
        
        scores = neuron_mag.clone().float()
        
        # Create masks
        supernode_mask = torch.zeros(num_neurons, dtype=torch.bool, device=device)
        supernode_mask[supernode_indices] = True
        
        halo_mask = torch.zeros(num_neurons, dtype=torch.bool, device=device)
        for cluster_idx in cluster_indices_list:
            if len(cluster_idx) > 0:
                halo_idx = compute_cluster_halo(
                    weights, cluster_idx, supernode_indices, self.halo_fraction
                )
                halo_mask[halo_idx] = True
        
        # Apply weights based on role
        scores[supernode_mask] *= self.supernode_weight
        scores[halo_mask & ~supernode_mask] *= self.halo_weight
        scores[~supernode_mask & ~halo_mask] *= self.regular_weight
        
        # Apply redundancy penalty for non-supernodes
        if self.redundancy_penalty > 0:
            # Compute per-neuron redundancy
            act_norm = outputs.float()
            act_norm = (act_norm - act_norm.mean(dim=0, keepdim=True)) / (act_norm.std(dim=0, keepdim=True) + 1e-8)
            
            # Sample reference neurons for efficiency
            num_refs = min(256, num_neurons)
            ref_indices = torch.randperm(num_neurons, device=device)[:num_refs]
            ref_acts = act_norm[:, ref_indices]
            
            # Correlation with references
            corr = (act_norm.T @ ref_acts) / max(1, outputs.shape[0] - 1)
            corr = torch.clamp(corr, -0.999, 0.999)
            
            # Redundancy
            rho_sq = corr ** 2
            mi = -0.5 * torch.log(1 - rho_sq + 1e-8)
            redundancy = mi.mean(dim=1)
            redundancy = redundancy / (redundancy.max() + 1e-8)  # Normalize
            
            # Apply penalty (only to non-supernodes)
            penalty = 1 - self.redundancy_penalty * redundancy
            scores[~supernode_mask] *= penalty[~supernode_mask]
        
        return scores
