"""
Cluster-aware pruning strategy with halo scoring and cluster constraints.

This implements a cluster-and-halo pruning approach:

Score_i = α·log(RQ_i) + β·Syn_i - γ·Red_i + λ·HaloSyn_i

With constraints:
1. Protect critical: prune at most p_C fraction from critical cluster per layer
2. Target redundant/background: prioritize pruning from these clusters  
3. Synergy-pair constraint: don't prune both members of top synergistic pairs
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base import BasePruningStrategy, PruningConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterAwarePruningConfig(PruningConfig):
    """Configuration for cluster-aware pruning."""
    
    # Score weights for the composite pruning score
    alpha: float = 1.0      # Weight for log(RQ)
    beta: float = 0.5       # Weight for Synergy
    gamma: float = 0.3      # Weight for Redundancy (subtracted)
    lambda_halo: float = 0.5  # Weight for HaloSyn term
    
    # Cluster constraints
    protect_critical_frac: float = 0.3   # Max fraction of critical to prune per layer
    target_redundant: bool = True        # Prioritize pruning redundant/background
    
    # Synergy-pair constraint
    synergy_pair_constraint: bool = True
    top_synergy_pairs: int = 10          # Number of top synergy pairs to protect
    
    # Halo parameters
    halo_percentile: float = 90.0
    use_activation_weight: bool = True
    
    # Clustering
    n_clusters: int = 4
    
    # Structured pruning (default True for channels)
    structured: bool = True
    
    # =========================================================================
    # IMPROVED FEATURES (depth/sparsity adaptive)
    # =========================================================================
    
    # Depth-adaptive weighting: vary weights based on layer depth
    depth_adaptive: bool = False          # Enable depth-adaptive weights
    early_layer_fraction: float = 0.3     # First 30% of layers are "early"
    
    # Early layer weights (less aggressive on RQ/synergy)
    early_alpha: float = 0.5
    early_beta: float = 0.3
    early_gamma: float = 0.2
    early_lambda_halo: float = 0.2
    
    # Late layer weights (full cluster-aware)
    late_alpha: float = 1.2
    late_beta: float = 0.8
    late_gamma: float = 0.5
    late_lambda_halo: float = 0.7
    
    # MI-aware scoring: use log(1+RQ) instead of log(RQ) for MI proxy
    use_mi_proxy: bool = False
    
    # Sparsity-adaptive protection: stronger critical protection at high sparsity
    sparsity_adaptive_protection: bool = False
    high_sparsity_threshold: float = 0.7  # When to strengthen protection
    high_sparsity_critical_frac: float = 0.1  # Max critical pruning at high sparsity


class ClusterAwarePruning(BasePruningStrategy):
    """
    Cluster-aware structured pruning with halo scoring.
    
    This strategy:
    1. Computes per-channel metrics (RQ, Redundancy, Synergy)
    2. Clusters channels into functional types (Critical, Redundant, Synergistic, Background)
    3. Computes downstream halo synergy for each channel
    4. Scores channels using composite formula with halo term
    5. Applies cluster constraints during selection
    
    Example:
        >>> config = ClusterAwarePruningConfig(amount=0.5)
        >>> strategy = ClusterAwarePruning(config)
        >>> scores = strategy.compute_importance_scores(
        ...     module, inputs=activations, logits=logits, labels=labels
        ... )
    """
    
    def __init__(
        self,
        config: Optional[ClusterAwarePruningConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
        precomputed_clusters: Optional[Dict[str, Any]] = None,
        precomputed_halos: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize cluster-aware pruning.
        
        Args:
            config: Pruning configuration
            precomputed_metrics: Optional dict with 'rq', 'redundancy', 'synergy' arrays
            precomputed_clusters: Optional dict with 'labels', 'type_mapping'
            precomputed_halos: Optional dict with 'halo_syn' per-channel array
        """
        super().__init__(config or ClusterAwarePruningConfig())
        self.config: ClusterAwarePruningConfig
        
        self.precomputed_metrics = precomputed_metrics
        self.precomputed_clusters = precomputed_clusters
        self.precomputed_halos = precomputed_halos
        
        # Cache for computed values
        self._metrics_cache = {}
        self._cluster_cache = {}
        self._halo_cache = {}
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_layer_weights: Optional[torch.Tensor] = None,
        next_layer_metrics: Optional[Dict[str, np.ndarray]] = None,
        layer_name: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute cluster-aware importance scores.
        
        Args:
            module: Module to compute scores for
            inputs: Input activations [batch, channels, ...]
            outputs: Output activations [batch, channels, ...]
            logits: Model logits [batch, n_classes]
            labels: True labels [batch]
            next_layer_weights: Weights of next layer for halo computation
            next_layer_metrics: Metrics of next layer channels for halo synergy
            layer_name: Layer identifier for caching
            
        Returns:
            Per-channel importance scores [n_channels]
        """
        # Get number of channels
        if hasattr(module, 'weight'):
            n_channels = module.weight.shape[0]
        elif outputs is not None:
            n_channels = outputs.shape[1]
        else:
            raise ValueError("Cannot determine number of channels")
        
        device = module.weight.device if hasattr(module, 'weight') else 'cpu'
        
        # 1. Get or compute per-channel metrics
        metrics = self._get_metrics(
            module, inputs, outputs, logits, labels, n_channels, layer_name
        )
        
        # 2. Get or compute clusters
        clusters = self._get_clusters(metrics, n_channels, layer_name)
        
        # 3. Get or compute halo synergy
        halo_syn = self._get_halo_syn(
            module, outputs, next_layer_weights, next_layer_metrics,
            clusters, n_channels, layer_name
        )
        
        # 4. Compute composite scores
        log_rq = np.log(np.clip(metrics['rq'], 1e-10, None))
        
        # Normalize each component to [0, 1] for stable weighting
        log_rq_norm = self._normalize(log_rq)
        syn_norm = self._normalize(metrics['synergy'])
        red_norm = self._normalize(metrics['redundancy'])
        halo_syn_norm = self._normalize(halo_syn)
        
        scores = (
            self.config.alpha * log_rq_norm +
            self.config.beta * syn_norm -
            self.config.gamma * red_norm +
            self.config.lambda_halo * halo_syn_norm
        )
        
        return torch.from_numpy(scores).float().to(device)
    
    def select_channels_to_prune(
        self,
        scores: torch.Tensor,
        n_prune: int,
        layer_name: str = "",
        protected_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Select channels to prune with cluster constraints.
        
        Args:
            scores: Per-channel importance scores [n_channels]
            n_prune: Number of channels to prune
            layer_name: Layer identifier for cached clusters
            
        Returns:
            List of channel indices to prune
        """
        n_channels = len(scores)
        scores_np = scores.cpu().numpy()
        
        # Get cluster info
        clusters = self._cluster_cache.get(layer_name, {})
        labels = clusters.get('labels', np.zeros(n_channels, dtype=int))
        type_mapping = clusters.get('type_mapping', {0: 'unknown'})
        
        # Invert type_mapping for lookup
        type_to_id = {v: k for k, v in type_mapping.items()}
        
        # Initialize selection
        selected = set()
        protected = set(int(i) for i in (protected_indices or []) if i is not None)
        
        # 1. Apply critical protection constraint
        if self.config.protect_critical_frac < 1.0:
            critical_id = type_to_id.get('critical', -1)
            if critical_id >= 0:
                critical_idx = np.where(labels == critical_id)[0]
                max_prune_critical = int(len(critical_idx) * self.config.protect_critical_frac)
                # Sort critical by score (ascending = low score first to prune)
                critical_sorted = sorted(critical_idx, key=lambda i: scores_np[i])
                # Protect the top (1 - frac) of critical channels
                protected.update(critical_sorted[max_prune_critical:])
        
        # 2. Apply synergy-pair constraint
        if self.config.synergy_pair_constraint:
            metrics = self._metrics_cache.get(layer_name, {})
            synergy_pairs = self._get_top_synergy_pairs(
                metrics, self.config.top_synergy_pairs
            )
            # Don't prune both members of a pair
            # We'll handle this during selection
        else:
            synergy_pairs = []
        
        # 3. Sort channels by score (ascending - low scores get pruned)
        # But prioritize redundant/background
        if self.config.target_redundant:
            redundant_id = type_to_id.get('redundant', -1)
            background_id = type_to_id.get('background', -1)
            
            # Create priority: redundant/background first, then others
            priority = np.zeros(n_channels)
            if redundant_id >= 0:
                priority[labels == redundant_id] = -1  # Higher priority to prune
            if background_id >= 0:
                priority[labels == background_id] = -1
            
            # Combined score: priority first, then actual score
            combined = list(zip(priority, scores_np, range(n_channels)))
            combined.sort()  # Sort by (priority, score)
            sorted_idx = [i for _, _, i in combined]
        else:
            sorted_idx = np.argsort(scores_np).tolist()
        
        # 4. Select channels respecting constraints
        pair_set = set()
        for i, j in synergy_pairs:
            pair_set.add((min(i, j), max(i, j)))
        
        for idx in sorted_idx:
            if len(selected) >= n_prune:
                break
            
            # Skip protected channels
            if idx in protected:
                continue
            
            # Check synergy-pair constraint
            if self.config.synergy_pair_constraint:
                pair_conflict = False
                for i, j in pair_set:
                    if (idx == i and j in selected) or (idx == j and i in selected):
                        pair_conflict = True
                        break
                if pair_conflict:
                    continue
            
            selected.add(idx)
        
        return list(selected)
    
    def _get_metrics(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor],
        outputs: Optional[torch.Tensor],
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        n_channels: int,
        layer_name: str,
    ) -> Dict[str, np.ndarray]:
        """Get or compute per-channel metrics."""
        if layer_name in self._metrics_cache:
            return self._metrics_cache[layer_name]
        
        if self.precomputed_metrics is not None:
            self._metrics_cache[layer_name] = self.precomputed_metrics
            return self.precomputed_metrics
        
        # Compute metrics from scratch
        metrics = {
            'rq': np.ones(n_channels),
            'redundancy': np.zeros(n_channels),
            'synergy': np.zeros(n_channels),
        }
        
        if outputs is None:
            logger.warning("No outputs provided, using default metrics")
            self._metrics_cache[layer_name] = metrics
            return metrics
        
        # Flatten spatial dims if needed
        if outputs.ndim == 4:
            # Conv: [B, C, H, W] -> [B, C]
            acts = outputs.mean(dim=(2, 3)).detach().cpu().numpy()
        else:
            acts = outputs.detach().cpu().numpy()
        
        # 1. RQ proxy (activation variance / weight norm^2)
        var = np.var(acts, axis=0)
        if hasattr(module, 'weight'):
            w = module.weight.detach().cpu().numpy()
            w_flat = w.reshape(w.shape[0], -1)
            w_norm_sq = np.sum(w_flat ** 2, axis=1)
            metrics['rq'] = var / (w_norm_sq[:len(var)] + 1e-10)
        else:
            metrics['rq'] = var
        
        # 2. Redundancy (Gaussian pairwise MI)
        if n_channels > 1:
            corr = np.corrcoef(acts.T)
            corr = np.clip(corr, -0.999, 0.999)
            mi_matrix = -0.5 * np.log(1 - corr ** 2)
            np.fill_diagonal(mi_matrix, 0)
            metrics['redundancy'] = np.mean(mi_matrix, axis=1)
        
        # 3. Synergy with continuous target
        if logits is not None and labels is not None:
            logits_np = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            
            # Logit margin as target
            batch_size = logits_np.shape[0]
            correct_logits = logits_np[np.arange(batch_size), labels_np]
            mask = np.ones_like(logits_np, dtype=bool)
            mask[np.arange(batch_size), labels_np] = False
            max_incorrect = np.where(mask, logits_np, -np.inf).max(axis=1)
            T = correct_logits - max_incorrect
            
            synergy = np.zeros(n_channels)
            for i in range(n_channels):
                mi_i = self._gaussian_mi(T, acts[:, i])
                # Top-k partners by redundancy
                partners = np.argsort(-metrics['redundancy'])[:10]
                syn_vals = []
                for j in partners:
                    if i == j:
                        continue
                    mi_j = self._gaussian_mi(T, acts[:, j])
                    mi_joint = self._gaussian_mi_joint(T, acts[:, i], acts[:, j])
                    s = mi_joint - mi_i - mi_j + min(mi_i, mi_j)
                    syn_vals.append(s)
                synergy[i] = np.mean(syn_vals) if syn_vals else 0.
            metrics['synergy'] = synergy
        
        self._metrics_cache[layer_name] = metrics
        return metrics
    
    def _get_clusters(
        self,
        metrics: Dict[str, np.ndarray],
        n_channels: int,
        layer_name: str,
    ) -> Dict[str, Any]:
        """Get or compute clusters."""
        if layer_name in self._cluster_cache:
            return self._cluster_cache[layer_name]
        
        if self.precomputed_clusters is not None:
            self._cluster_cache[layer_name] = self.precomputed_clusters
            return self.precomputed_clusters
        
        # Cluster using MetricSpaceClustering
        from ...analysis.clustering import MetricSpaceClustering
        
        clusterer = MetricSpaceClustering(
            n_clusters=self.config.n_clusters,
            seed=42,
        )
        result = clusterer.fit(
            metrics['rq'],
            metrics['redundancy'],
            metrics['synergy'],
            layer_name,
        )
        
        clusters = {
            'labels': result.labels,
            'centroids': result.centroids,
            'type_mapping': result.type_mapping,
            'type_counts': result.type_counts,
        }
        
        self._cluster_cache[layer_name] = clusters
        return clusters
    
    def _get_halo_syn(
        self,
        module: nn.Module,
        outputs: Optional[torch.Tensor],
        next_layer_weights: Optional[torch.Tensor],
        next_layer_metrics: Optional[Dict[str, np.ndarray]],
        clusters: Dict[str, Any],
        n_channels: int,
        layer_name: str,
    ) -> np.ndarray:
        """Compute per-channel halo synergy."""
        if layer_name in self._halo_cache:
            return self._halo_cache[layer_name]
        
        if self.precomputed_halos is not None:
            halo_syn = self.precomputed_halos.get('halo_syn', np.zeros(n_channels))
            self._halo_cache[layer_name] = halo_syn
            return halo_syn
        
        # If no next layer info, return zeros
        if next_layer_weights is None or next_layer_metrics is None:
            halo_syn = np.zeros(n_channels)
            self._halo_cache[layer_name] = halo_syn
            return halo_syn
        
        # Compute halo for each channel
        from ...analysis.clustering import CrossLayerHaloAnalysis
        
        halo_analyzer = CrossLayerHaloAnalysis(
            percentile=self.config.halo_percentile,
            use_activation_weight=self.config.use_activation_weight,
        )
        
        # Get weights and compute influence
        w_np = next_layer_weights.detach().cpu().numpy()
        if w_np.ndim == 4:
            # Conv: [out, in, k, k] -> [out, in]
            influence = np.abs(w_np).sum(axis=(2, 3))
        else:
            influence = np.abs(w_np)
        
        # Weight by activation std if available
        if outputs is not None and self.config.use_activation_weight:
            if outputs.ndim == 4:
                acts = outputs.mean(dim=(2, 3)).detach().cpu().numpy()
            else:
                acts = outputs.detach().cpu().numpy()
            std = np.std(acts, axis=0)
            n_in = min(influence.shape[1], len(std))
            influence[:, :n_in] = influence[:, :n_in] * std[:n_in]
        
        # Get next layer synergy
        next_syn = next_layer_metrics.get('synergy', np.zeros(influence.shape[0]))
        
        # Per-channel halo synergy
        halo_syn = np.zeros(n_channels)
        for i in range(min(n_channels, influence.shape[1])):
            # Find receivers that get high influence from this channel
            infl_i = influence[:, i]
            total_infl = influence.sum(axis=1) + 1e-10
            rel_infl = infl_i / total_infl
            thresh = np.percentile(rel_infl, self.config.halo_percentile)
            halo_mask = rel_infl >= thresh
            if halo_mask.sum() > 0:
                halo_syn[i] = np.mean(next_syn[halo_mask])
        
        self._halo_cache[layer_name] = halo_syn
        return halo_syn
    
    def _get_top_synergy_pairs(
        self,
        metrics: Dict[str, np.ndarray],
        top_k: int,
    ) -> List[Tuple[int, int]]:
        """Get top synergy pairs for constraint."""
        synergy = metrics.get('synergy', np.array([]))
        n = len(synergy)
        if n < 2:
            return []
        
        # Simple heuristic: pair high-synergy channels
        top_idx = np.argsort(-synergy)[:min(top_k * 2, n)]
        pairs = []
        for i in range(0, len(top_idx) - 1, 2):
            pairs.append((int(top_idx[i]), int(top_idx[i + 1])))
        return pairs[:top_k]
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return x
    
    def _gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Gaussian MI between two variables."""
        rho = np.corrcoef(x, y)[0, 1]
        rho = np.clip(rho, -0.999, 0.999)
        return max(0, -0.5 * np.log(1 - rho ** 2))
    
    def _gaussian_mi_joint(self, t: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> float:
        """Compute Gaussian MI I(T; [Y1, Y2])."""
        joint = np.column_stack([t, y1, y2])
        cov = np.cov(joint.T) + 1e-8 * np.eye(3)
        var_t = cov[0, 0]
        cov_y = cov[1:, 1:]
        det_all = np.linalg.det(cov)
        det_y = np.linalg.det(cov_y)
        if det_all <= 0 or det_y <= 0 or var_t <= 0:
            return 0.
        return max(0, 0.5 * np.log(var_t * det_y / det_all))


class CompositePruning(ClusterAwarePruning):
    """
    Composite per-channel pruning (baseline without cluster constraints).
    
    Uses the same scoring as ClusterAwarePruning but without:
    - Cluster constraints (protect critical, target redundant)
    - Synergy-pair constraints
    - Halo term (lambda = 0)
    
    This corresponds to a composite-score baseline (same features, no constraints).
    """
    
    def __init__(
        self,
        config: Optional[ClusterAwarePruningConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ClusterAwarePruningConfig()
        
        # Disable cluster constraints and halo
        config.protect_critical_frac = 1.0  # No protection
        config.target_redundant = False
        config.synergy_pair_constraint = False
        config.lambda_halo = 0.0  # No halo term
        
        super().__init__(config, **kwargs)
    
    def select_channels_to_prune(
        self,
        scores: torch.Tensor,
        n_prune: int,
        layer_name: str = "",
        protected_indices: Optional[List[int]] = None,
    ) -> List[int]:
        """Simple selection by score (no constraints)."""
        scores_np = scores.cpu().numpy()
        sorted_idx = np.argsort(scores_np)
        # Respect external protection constraints if provided.
        protected = set(int(i) for i in (protected_indices or []) if i is not None)
        out = []
        for idx in sorted_idx.tolist():
            if idx in protected:
                continue
            out.append(int(idx))
            if len(out) >= int(n_prune):
                break
        return out
