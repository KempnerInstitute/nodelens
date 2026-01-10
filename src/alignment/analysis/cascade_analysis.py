"""
Cascade and damage analysis for pruning validation.

Implements:
1. Cascade test: measure downstream disruption when removing channels
2. Damage prediction: correlate scores with true accuracy drop
3. Cluster-specific ablation: compare damage by functional type
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class CascadeResult:
    """Result of cascade test for a cluster type."""
    layer_name: str
    cluster_type: str
    n_removed: int
    accuracy_drop: float
    loss_increase: float


@dataclass
class DamageResult:
    """Result of damage prediction analysis."""
    layer_name: str
    method: str
    spearman: float
    top_k_recall: Dict[int, float]


class CascadeAnalysis:
    """Analyze cascade effects of channel removal."""

    def __init__(self, model, dataloader, device="cuda"):
        self.model = model
        self.loader = dataloader
        self.device = device
        self._baseline = None

    def baseline(self):
        """Compute baseline accuracy/loss."""
        if not HAS_TORCH:
            return {"acc": 0., "loss": 0.}
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.
        crit = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss_sum += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        self._baseline = {"acc": correct/total, "loss": loss_sum/total}
        return self._baseline

    def ablate(self, layer_name: str, indices: List[int]) -> CascadeResult:
        """Remove channels and measure effect."""
        if self._baseline is None:
            self.baseline()
        layer = dict(self.model.named_modules()).get(layer_name)
        if layer is None or not hasattr(layer, 'weight'):
            return CascadeResult(layer_name, "", len(indices), 0., 0.)
        orig_w = layer.weight.data.clone()
        orig_b = layer.bias.data.clone() if layer.bias is not None else None
        layer.weight.data[indices] = 0
        if orig_b is not None:
            layer.bias.data[indices] = 0
        new = self._eval()
        layer.weight.data = orig_w
        if orig_b is not None:
            layer.bias.data = orig_b
        return CascadeResult(layer_name, "", len(indices),
                            self._baseline["acc"] - new["acc"],
                            new["loss"] - self._baseline["loss"])

    def by_cluster(self, layer: str, labels: np.ndarray, 
                   types: Dict[int, str], n_rm: int = 5) -> Dict[str, CascadeResult]:
        """Run cascade test per cluster type."""
        results = {}
        for cid, ctype in types.items():
            idx = np.where(labels == cid)[0]
            if len(idx) == 0:
                continue
            rm = np.random.choice(idx, min(n_rm, len(idx)), replace=False).tolist()
            r = self.ablate(layer, rm)
            r.cluster_type = ctype
            results[ctype] = r
        return results

    def _eval(self):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.
        crit = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss_sum += crit(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return {"acc": correct/total, "loss": loss_sum/total}


class DamagePrediction:
    """Predict damage from importance scores."""

    def __init__(self, cascade: CascadeAnalysis, layer: str):
        self.cascade = cascade
        self.layer = layer
        self._damages = None

    def compute_damages(self, n_ch: int, frac: float = 0.2) -> np.ndarray:
        """Compute true per-channel damage."""
        damages = np.zeros(n_ch)
        test_idx = np.random.choice(n_ch, max(1, int(n_ch * frac)), replace=False)
        for i in test_idx:
            r = self.cascade.ablate(self.layer, [int(i)])
            damages[i] = r.accuracy_drop
        self._damages = damages
        return damages

    def evaluate(self, scores: np.ndarray, method: str = "composite",
                top_ks: List[int] = [10, 20, 50]) -> DamageResult:
        """Evaluate score vs damage correlation."""
        from scipy import stats
        if self._damages is None:
            raise ValueError("Call compute_damages first")
        mask = self._damages != 0
        if mask.sum() < 5:
            return DamageResult(self.layer, method, 0., {})
        d, s = self._damages[mask], scores[mask]
        rho, _ = stats.spearmanr(s, -d)
        recall = {}
        by_d = np.argsort(-d)
        by_s = np.argsort(s)
        for k in top_ks:
            k = min(k, len(d))
            overlap = len(set(by_d[:k]) & set(by_s[:k]))
            recall[k] = overlap / k if k > 0 else 0.
        return DamageResult(self.layer, method, float(rho) if not np.isnan(rho) else 0., recall)
