"""Cross-layer halo analysis."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class HaloResult:
    """Result of halo analysis for a cluster."""
    layer_name: str
    source_cluster: str
    halo_indices: np.ndarray
    halo_size: int
    halo_redundancy_mean: float
    halo_synergy_mean: float
    influence_scores: Optional[np.ndarray] = None


class CrossLayerHaloAnalysis:
    """
    Analyze downstream halos of clusters.
    
    A halo is the set of channels in the next layer that receive
    disproportionate input from a given cluster.
    """
    
    def __init__(self, percentile: float = 90.0, use_activation_weight: bool = True):
        """
        Args:
            percentile: Threshold percentile for halo membership
            use_activation_weight: Whether to weight by activation std
        """
        self.percentile = percentile
        self.use_activation_weight = use_activation_weight
    
    def compute_influence(self, weights: np.ndarray, activations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute influence scores.
        
        Args:
            weights: Weight matrix [out_channels, in_channels]
            activations: Optional activations [batch, in_channels]
            
        Returns:
            Influence matrix [out_channels, in_channels]
        """
        w = np.abs(weights)
        if activations is not None and self.use_activation_weight:
            std = np.std(activations, axis=0)
            w = w * std[None, :]
        return w
    
    def find_halo(
        self,
        influence: np.ndarray,
        cluster_indices: np.ndarray,
    ) -> tuple:
        """
        Find receivers that get high relative influence from cluster.
        
        Args:
            influence: Influence matrix [out, in]
            cluster_indices: Indices of channels in source cluster
            
        Returns:
            (halo_indices, relative_influence_scores)
        """
        # Sum influence from cluster members
        infl_from_cluster = influence[:, cluster_indices].sum(axis=1)
        # Normalize by total incoming
        total_infl = influence.sum(axis=1) + 1e-10
        rel_infl = infl_from_cluster / total_infl
        # Threshold
        thresh = np.percentile(rel_infl, self.percentile)
        halo_mask = rel_infl >= thresh
        return np.where(halo_mask)[0], rel_infl
    
    def analyze_halo(
        self,
        halo_indices: np.ndarray,
        redundancy: np.ndarray,
        synergy: np.ndarray,
        layer_name: str = "",
        cluster_name: str = "",
    ) -> HaloResult:
        """
        Compute properties of a halo.
        
        Args:
            halo_indices: Indices of halo channels
            redundancy: Per-channel redundancy in next layer
            synergy: Per-channel synergy in next layer
            layer_name: Layer identifier
            cluster_name: Source cluster type
            
        Returns:
            HaloResult with summary statistics
        """
        if len(halo_indices) == 0:
            return HaloResult(
                layer_name=layer_name,
                source_cluster=cluster_name,
                halo_indices=halo_indices,
                halo_size=0,
                halo_redundancy_mean=0.0,
                halo_synergy_mean=0.0,
            )
        
        red_mean = float(np.mean(redundancy[halo_indices]))
        syn_mean = float(np.mean(synergy[halo_indices]))
        
        return HaloResult(
            layer_name=layer_name,
            source_cluster=cluster_name,
            halo_indices=halo_indices,
            halo_size=len(halo_indices),
            halo_redundancy_mean=red_mean,
            halo_synergy_mean=syn_mean,
        )
    
    def compute_cluster_to_cluster_flow(
        self,
        influence: np.ndarray,
        source_labels: np.ndarray,
        target_labels: np.ndarray,
        source_types: Dict[int, str],
        target_types: Dict[int, str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute cluster-to-cluster influence matrix.
        
        Args:
            influence: [out, in] influence matrix
            source_labels: Cluster labels for source layer
            target_labels: Cluster labels for target layer
            source_types: Mapping from cluster ID to type name
            target_types: Mapping from cluster ID to type name
            
        Returns:
            Nested dict: flow[source_type][target_type] = mean influence
        """
        flow = {}
        for src_id, src_type in source_types.items():
            flow[src_type] = {}
            src_mask = source_labels == src_id
            src_infl = influence[:, src_mask].sum(axis=1)  # [out]
            
            for tgt_id, tgt_type in target_types.items():
                tgt_mask = target_labels == tgt_id
                if tgt_mask.sum() > 0:
                    mean_infl = float(np.mean(src_infl[tgt_mask]))
                    flow[src_type][tgt_type] = mean_infl
                else:
                    flow[src_type][tgt_type] = 0.0
        
        return flow
