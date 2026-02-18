"""Cross-layer halo analysis."""
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
            Nested dict: flow[source_type][target_type] = normalized influence mass
        """
        flow = {}
        for src_id, src_type in source_types.items():
            flow[src_type] = {}
            src_mask = source_labels == src_id
            src_infl = influence[:, src_mask].sum(axis=1)  # [out]
            denom = float(src_infl.sum()) + 1e-10
            
            for tgt_id, tgt_type in target_types.items():
                tgt_mask = target_labels == tgt_id
                if tgt_mask.sum() > 0:
                    # Fraction of total outgoing influence mass from src cluster
                                flow[src_type][tgt_type] = float(src_infl[tgt_mask].sum()) / denom
                else:
                    flow[src_type][tgt_type] = 0.0
        
        return flow

    def permutation_baseline(
        self,
        influence: np.ndarray,
        labels: np.ndarray,
        type_mapping: Dict[int, str],
        redundancy: np.ndarray,
        synergy: np.ndarray,
        n_permutations: int = 100,
        seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute null distribution of halo effects by shuffling cluster labels.
        
        This establishes a baseline to determine if observed halo effects
        are statistically significant beyond random assignment.
        
        Args:
            influence: Influence matrix [out, in]
            labels: Original cluster labels for source layer
            type_mapping: Cluster ID to type name mapping
            redundancy: Per-channel redundancy in target layer
            synergy: Per-channel synergy in target layer
            n_permutations: Number of random permutations
            seed: Random seed for reproducibility
        
        Returns:
            Dict mapping cluster type to:
                - 'observed_red': Actual halo redundancy
                - 'observed_syn': Actual halo synergy
                - 'null_red_mean': Mean halo redundancy under null
                - 'null_red_std': Std of null distribution
                - 'null_syn_mean': Mean halo synergy under null
                - 'null_syn_std': Std of null distribution
                - 'z_red': Z-score of observed vs null for redundancy
                - 'z_syn': Z-score of observed vs null for synergy
                - 'p_red': Proportion of null >= observed (one-tailed)
                - 'p_syn': Proportion of null >= observed
        """
        rng = np.random.default_rng(seed)
        n_in = labels.size
        
        # Compute observed halo effects
        observed = {}
        for cid, ctype in type_mapping.items():
            cluster_idx = np.where(labels == cid)[0]
            if len(cluster_idx) == 0 or cluster_idx.max() >= influence.shape[1]:
                continue
            
            halo_idx, _ = self.find_halo(influence, cluster_idx)
            if len(halo_idx) == 0:
                continue
            
            observed[ctype] = {
                'halo_red': float(np.mean(redundancy[halo_idx])),
                'halo_syn': float(np.mean(synergy[halo_idx])),
                'halo_size': len(halo_idx),
            }
        
        # Run permutations
        null_results = {ctype: {'red': [], 'syn': []} for ctype in observed.keys()}
        
        for _ in range(n_permutations):
            # Shuffle labels while preserving cluster sizes
            perm_labels = rng.permutation(labels)
            
            for cid, ctype in type_mapping.items():
                if ctype not in null_results:
                    continue
                
                cluster_idx = np.where(perm_labels == cid)[0]
                if len(cluster_idx) == 0 or cluster_idx.max() >= influence.shape[1]:
                    continue
                
                halo_idx, _ = self.find_halo(influence, cluster_idx)
                if len(halo_idx) > 0:
                    null_results[ctype]['red'].append(float(np.mean(redundancy[halo_idx])))
                    null_results[ctype]['syn'].append(float(np.mean(synergy[halo_idx])))
        
        # Compute statistics
        results = {}
        for ctype, obs in observed.items():
            null_red = np.array(null_results[ctype]['red'])
            null_syn = np.array(null_results[ctype]['syn'])
            
            null_red_mean = float(np.mean(null_red)) if len(null_red) > 0 else 0.0
            null_red_std = float(np.std(null_red)) if len(null_red) > 0 else 1.0
            null_syn_mean = float(np.mean(null_syn)) if len(null_syn) > 0 else 0.0
            null_syn_std = float(np.std(null_syn)) if len(null_syn) > 0 else 1.0
            
            # Avoid division by zero
            null_red_std = max(null_red_std, 1e-10)
            null_syn_std = max(null_syn_std, 1e-10)
            
            z_red = (obs['halo_red'] - null_red_mean) / null_red_std
            z_syn = (obs['halo_syn'] - null_syn_mean) / null_syn_std
            
            # One-tailed p-value: proportion of null >= observed
            p_red = float(np.mean(null_red >= obs['halo_red'])) if len(null_red) > 0 else 1.0
            p_syn = float(np.mean(null_syn >= obs['halo_syn'])) if len(null_syn) > 0 else 1.0
            
            results[ctype] = {
                'observed_red': obs['halo_red'],
                'observed_syn': obs['halo_syn'],
                'halo_size': obs['halo_size'],
                'null_red_mean': null_red_mean,
                'null_red_std': null_red_std,
                'null_syn_mean': null_syn_mean,
                'null_syn_std': null_syn_std,
                'z_red': float(z_red),
                'z_syn': float(z_syn),
                'p_red': p_red,
                'p_syn': p_syn,
                'n_permutations': n_permutations,
            }
        
        return results
