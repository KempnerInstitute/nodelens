"""Metric-space clustering for channels."""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SK = True
except ImportError:
    HAS_SK = False


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


class MetricSpaceClustering:
    def __init__(self, n_clusters=4, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed

    def fit(self, rq, red, syn, name="layer"):
        rq = np.asarray(rq).flatten()
        red = np.asarray(red).flatten()
        syn = np.asarray(syn).flatten()
        n = len(rq)
        X = np.column_stack([np.log(np.clip(rq, 1e-10, None)), red, syn])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        if HAS_SK and n >= self.n_clusters:
            km = KMeans(self.n_clusters, random_state=self.seed, n_init=10)
            lab = km.fit_predict(X)
            cen = km.cluster_centers_
            sil = silhouette_score(X, lab) if n > self.n_clusters else 0.
        else:
            lab, cen, sil = np.zeros(n, int), np.zeros((1, 3)), 0.
        tm = self._types(cen)
        tc = {t: int((lab == k).sum()) for k, t in tm.items()}
        return ClusterResult(name, n, len(cen), lab, cen, sil, tm, tc)

    def _types(self, c):
        if len(c) < 4:
            return {i: "unknown" for i in range(len(c))}
        m, used = {}, set()
        i = int(np.argmax(c[:, 0] - c[:, 1]))
        m[i] = "critical"; used.add(i)
        rem = [j for j in range(len(c)) if j not in used]
        i = rem[int(np.argmax([c[j, 1] for j in rem]))]
        m[i] = "redundant"; used.add(i)
        rem = [j for j in range(len(c)) if j not in used]
        i = rem[int(np.argmax([c[j, 2] for j in rem]))]
        m[i] = "synergistic"; used.add(i)
        for j in range(len(c)):
            if j not in m:
                m[j] = "background"
        return m
