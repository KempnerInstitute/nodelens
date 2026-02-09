"""Metric-space clustering for channels."""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Literal

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SK = True
except ImportError:
    HAS_SK = False


# Ablation modes: which metrics to include in clustering
# Format: (use_first_metric, use_red, use_syn)
# The first metric can be RQ or IXY depending on what's passed to fit()
METRIC_ABLATIONS = {
    "all": (True, True, True),      # RQ, Red, Syn
    "rq_red": (True, True, False),  # RQ + Redundancy only
    "rq_syn": (True, False, True),  # RQ + Synergy only
    "red_syn": (False, True, True), # Redundancy + Synergy only
    "rq_only": (True, False, False),
    "red_only": (False, True, False),
    "syn_only": (False, False, True),
    # IXY variants: when used, the first argument to fit() should be I(X;Y) instead of RQ
    "ixy_all": (True, True, True),      # I(X;Y), Red, Syn
    "ixy_red": (True, True, False),     # I(X;Y) + Redundancy only
    "ixy_syn": (True, False, True),     # I(X;Y) + Synergy only
    "ixy_only": (True, False, False),   # I(X;Y) only
}


@dataclass
class ClusterResult:
    layer_name: str
    n_channels: int
    n_clusters: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float
    type_mapping: Dict[int, str]
    type_counts: Dict[str, int]
    # Additional fields for ablation tracking
    metrics_used: Tuple[bool, bool, bool] = (True, True, True)
    ablation_mode: str = "all"


@dataclass
class AblationResult:
    """Results from metric ablation study."""
    layer_name: str
    ablation_mode: str
    silhouette: float
    cluster_result: ClusterResult
    # Agreement with full 3-metric clustering
    ari_vs_full: float = 0.0
    ami_vs_full: float = 0.0


class MetricSpaceClustering:
    """
    Cluster channels in metric space (log RQ, Redundancy, Synergy).
    
    Supports ablation studies to validate each metric's contribution
    by clustering with subsets of the three metrics.
    """
    
    def __init__(
        self,
        n_clusters: int = 4,
        seed: int = 42,
        *,
        type_mapping_mode: str = "greedy",
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        mode = str(type_mapping_mode or "greedy").lower()
        # Backward-compatibility:
        #   - "global" keeps historical penalized global assignment
        #   - "global_permutation" aliases to "global_penalized"
        if mode in {"global", "global_permutation"}:
            mode = "global_penalized"
        elif mode in {"global_simple", "simple"}:
            mode = "global_simple"
        elif mode in {"global_prototype", "prototype"}:
            mode = "global_prototype"
        else:
            mode = "greedy"
        self.type_mapping_mode: Literal[
            "greedy",
            "global_penalized",
            "global_simple",
            "global_prototype",
        ] = mode  # type: ignore[assignment]

    def fit(
        self, 
        rq, 
        red, 
        syn, 
        name: str = "layer",
        ablation: str = "all",
    ) -> ClusterResult:
        """
        Cluster channels using specified metrics.
        
        Args:
            rq: Rayleigh quotient values per channel
            red: Redundancy values per channel
            syn: Synergy values per channel
            name: Layer name identifier
            ablation: Which metrics to use - one of:
                - "all": Use all 3 metrics (default)
                - "rq_red": RQ + Redundancy only
                - "rq_syn": RQ + Synergy only
                - "red_syn": Redundancy + Synergy only
                - "rq_only", "red_only", "syn_only": Single metrics
        
        Returns:
            ClusterResult with cluster assignments and statistics
        """
        rq = np.asarray(rq).flatten()
        red = np.asarray(red).flatten()
        syn = np.asarray(syn).flatten()
        n = len(rq)
        
        # Get ablation mask
        use_rq, use_red, use_syn = METRIC_ABLATIONS.get(ablation, (True, True, True))
        
        # Build feature matrix based on ablation
        features = []
        feature_names = []
        if use_rq:
            features.append(np.log(np.clip(rq, 1e-10, None)))
            feature_names.append("log_rq")
        if use_red:
            features.append(red)
            feature_names.append("red")
        if use_syn:
            features.append(syn)
            feature_names.append("syn")
        
        if len(features) == 0:
            # Fallback to all metrics if ablation is invalid
            features = [np.log(np.clip(rq, 1e-10, None)), red, syn]
            use_rq, use_red, use_syn = True, True, True
            ablation = "all"
        
        X = np.column_stack(features)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        
        # Adjust n_clusters for reduced feature dimensions
        effective_k = min(self.n_clusters, n - 1) if n > 1 else 1
        effective_k = max(1, effective_k)
        
        if HAS_SK and n >= effective_k and effective_k >= 2:
            km = KMeans(effective_k, random_state=self.seed, n_init=10)
            lab = km.fit_predict(X)
            cen = km.cluster_centers_
            sil = silhouette_score(X, lab) if n > effective_k else 0.
        else:
            lab = np.zeros(n, dtype=int)
            cen = np.zeros((1, len(features)))
            sil = 0.
        
        # Type mapping needs full 3D centroids for consistent labeling
        # Pad centroids with zeros for missing dimensions
        full_cen = np.zeros((len(cen), 3))
        idx = 0
        if use_rq:
            full_cen[:, 0] = cen[:, idx]
            idx += 1
        if use_red:
            full_cen[:, 1] = cen[:, idx]
            idx += 1
        if use_syn:
            full_cen[:, 2] = cen[:, idx]
        
        tm = self._types(full_cen, metrics_used=(use_rq, use_red, use_syn))
        tc = {t: int((lab == k).sum()) for k, t in tm.items()}
        
        return ClusterResult(
            layer_name=name,
            n_channels=n,
            n_clusters=len(cen),
            labels=lab,
            centroids=cen,
            silhouette=sil,
            type_mapping=tm,
            type_counts=tc,
            metrics_used=(use_rq, use_red, use_syn),
            ablation_mode=ablation,
        )
    
    def run_ablation_study(
        self,
        rq,
        red,
        syn,
        name: str = "layer",
        ablations: Optional[List[str]] = None,
    ) -> Dict[str, AblationResult]:
        """
        Run clustering with different metric subsets and compare to full clustering.
        
        Args:
            rq, red, syn: Per-channel metric values
            name: Layer name
            ablations: List of ablation modes to test (default: all 2-metric combinations)
        
        Returns:
            Dict mapping ablation mode to AblationResult
        """
        if ablations is None:
            ablations = ["all", "rq_red", "rq_syn", "red_syn"]
        
        results = {}
        full_result = None
        
        for ablation in ablations:
            result = self.fit(rq, red, syn, name, ablation=ablation)
            
            if ablation == "all":
                full_result = result
            
            results[ablation] = AblationResult(
                layer_name=name,
                ablation_mode=ablation,
                silhouette=result.silhouette,
                cluster_result=result,
            )
        
        # Compute agreement metrics against full clustering
        if full_result is not None and HAS_SK:
            try:
                from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
                for ablation, abl_result in results.items():
                    if ablation != "all":
                        abl_result.ari_vs_full = adjusted_rand_score(
                            full_result.labels, abl_result.cluster_result.labels
                        )
                        abl_result.ami_vs_full = adjusted_mutual_info_score(
                            full_result.labels, abl_result.cluster_result.labels
                        )
            except ImportError:
                pass
        
        return results

    def _types_greedy(self, c: np.ndarray) -> Dict[int, str]:
        """
        Greedy sequential type assignment.

        This mapping is intentionally simple and can be more label-swap prone than
        the global (permutation) assignment, especially when centroids are close.
        """
        if len(c) < 4:
            return {i: "unknown" for i in range(len(c))}
        m: Dict[int, str] = {}
        used = set()

        i = int(np.argmax(c[:, 0] - c[:, 1]))
        m[i] = "critical"
        used.add(i)

        rem = [j for j in range(len(c)) if j not in used]
        i = rem[int(np.argmax([c[j, 1] for j in rem]))]
        m[i] = "redundant"
        used.add(i)

        rem = [j for j in range(len(c)) if j not in used]
        i = rem[int(np.argmax([c[j, 2] for j in rem]))]
        m[i] = "synergistic"
        used.add(i)

        for j in range(len(c)):
            if j not in m:
                m[j] = "background"
        return m

    def _solve_global_assignment(self, scores: np.ndarray) -> Dict[int, str]:
        """
        Solve one-to-one cluster->type assignment by maximizing total score.

        Args:
            scores: [n_clusters, 4] score matrix for
                [critical, redundant, synergistic, background].
        """
        import itertools

        type_names = ["critical", "redundant", "synergistic", "background"]
        n = int(scores.shape[0])
        best = None
        best_score = -1e30

        # Enumerate nP4 assignments (n is small in practice; defaults to 4).
        for perm in itertools.permutations(range(n), 4):
            s = (
                scores[perm[0], 0]
                + scores[perm[1], 1]
                + scores[perm[2], 2]
                + scores[perm[3], 3]
            )
            if s > best_score:
                best_score = float(s)
                best = perm

        mapping: Dict[int, str] = {}
        if best is not None:
            for t_idx, c_idx in enumerate(best):
                mapping[int(c_idx)] = type_names[int(t_idx)]

        # Any extra clusters (if n_clusters > 4) are treated as background.
        for j in range(n):
            if int(j) not in mapping:
                mapping[int(j)] = "background"
        return mapping

    def _scores_global_penalized(
        self,
        c: np.ndarray,
        metrics_used: Tuple[bool, bool, bool],
    ) -> np.ndarray:
        """
        Historical global scoring with mild cross-metric penalties.
        Kept for backward-compatible paper reproduction.
        """
        use_rq, use_red, use_syn = metrics_used
        w_rq = 1.0 if use_rq else 0.0
        w_red = 1.0 if use_red else 0.0
        w_syn = 1.0 if use_syn else 0.0

        scores = np.zeros((len(c), 4), dtype=np.float64)
        # critical: high rq, low red
        scores[:, 0] = (w_rq * c[:, 0]) - (w_red * c[:, 1])
        # redundant: high red (with mild penalty for high rq)
        scores[:, 1] = (w_red * c[:, 1]) - (0.25 * w_rq * c[:, 0])
        # synergistic: high syn (with mild penalty for high red)
        scores[:, 2] = (w_syn * c[:, 2]) - (0.25 * w_red * c[:, 1])
        # background: close to origin
        scores[:, 3] = -(
            (w_rq * np.abs(c[:, 0]))
            + (w_red * np.abs(c[:, 1]))
            + (w_syn * np.abs(c[:, 2]))
        )
        return scores

    def _scores_global_simple(
        self,
        c: np.ndarray,
        metrics_used: Tuple[bool, bool, bool],
    ) -> np.ndarray:
        """
        Definition-aligned simple scoring (no cross-metric penalty weights).
        """
        use_rq, use_red, use_syn = metrics_used
        w_rq = 1.0 if use_rq else 0.0
        w_red = 1.0 if use_red else 0.0
        w_syn = 1.0 if use_syn else 0.0

        scores = np.zeros((len(c), 4), dtype=np.float64)
        # critical: high rq, low red
        scores[:, 0] = (w_rq * c[:, 0]) - (w_red * c[:, 1])
        # redundant: maximize redundancy
        scores[:, 1] = w_red * c[:, 1]
        # synergistic: maximize synergy
        scores[:, 2] = w_syn * c[:, 2]
        # background: low magnitude in active metric dimensions
        scores[:, 3] = -(
            (w_rq * np.abs(c[:, 0]))
            + (w_red * np.abs(c[:, 1]))
            + (w_syn * np.abs(c[:, 2]))
        )
        return scores

    def _scores_global_prototype(
        self,
        c: np.ndarray,
        metrics_used: Tuple[bool, bool, bool],
    ) -> np.ndarray:
        """
        Parameter-free prototype matching in (log_rq, red, syn) space using cosine similarity.
        """
        use_rq, use_red, use_syn = metrics_used
        mask = np.array(
            [
                1.0 if use_rq else 0.0,
                1.0 if use_red else 0.0,
                1.0 if use_syn else 0.0,
            ],
            dtype=np.float64,
        )

        # [critical, redundant, synergistic, background]
        prototypes = np.array(
            [
                [1.0, -1.0, 0.0],   # high rq, low red
                [0.0, 1.0, -1.0],   # high red, low syn
                [0.0, -1.0, 1.0],   # high syn, low red
                [-1.0, -1.0, -1.0], # low on all
            ],
            dtype=np.float64,
        )
        prototypes = prototypes * mask[None, :]
        proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        prototypes = prototypes / proto_norm

        cent = np.asarray(c, dtype=np.float64) * mask[None, :]
        cent_norm = np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
        cent = cent / cent_norm

        # Maximize cosine similarity.
        return cent @ prototypes.T

    def _types(self, c, metrics_used: Tuple[bool, bool, bool] = (True, True, True)):
        """
        Assign cluster types based on centroids.

        Args:
            c: Cluster centroids [n_clusters, 3] (columns: log_rq, red, syn)
            metrics_used: Which metrics are available (rq, red, syn)

        Returns:
            Dict mapping cluster_id to type name
        """
        if self.type_mapping_mode == "greedy":
            return self._types_greedy(c)

        if len(c) < 4:
            return {i: "unknown" for i in range(len(c))}

        if self.type_mapping_mode == "global_simple":
            scores = self._scores_global_simple(c, metrics_used)
        elif self.type_mapping_mode == "global_prototype":
            scores = self._scores_global_prototype(c, metrics_used)
        else:
            # Includes backward-compatible alias "global" normalized to "global_penalized".
            scores = self._scores_global_penalized(c, metrics_used)

        return self._solve_global_assignment(scores)
