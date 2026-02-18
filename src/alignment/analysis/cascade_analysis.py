"""
Cascade and damage analysis for pruning validation.

Implements:
1. Cascade test: measure downstream disruption when removing channels
2. Damage prediction: correlate scores with true accuracy drop
3. Cluster-specific ablation: compare damage by functional type
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


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
        # Performance: avoid cloning the entire parameter tensor for each ablation.
        # We only need to restore the ablated output channels (dim=0).
        idx = [int(i) for i in indices]
        orig_w_slice = layer.weight.data[idx].clone()
        orig_b_slice = layer.bias.data[idx].clone() if layer.bias is not None else None
        layer.weight.data[idx] = 0
        if orig_b_slice is not None:
            layer.bias.data[idx] = 0
        new = self._eval()
        layer.weight.data[idx] = orig_w_slice
        if orig_b_slice is not None:
            layer.bias.data[idx] = orig_b_slice
        return CascadeResult(layer_name, "", len(indices),
                            self._baseline["acc"] - new["acc"],
                            new["loss"] - self._baseline["loss"])

    def by_cluster(
        self,
        layer: str,
        labels: np.ndarray,
        types: Dict[int, str],
        n_rm: int = 5,
        seed: int = 0,
    ) -> Dict[str, CascadeResult]:
        """Run cascade test per cluster type.

        Notes:
        - We sample channels *within* each cluster type to make the comparison fair.
        - We use a fixed RNG seed by default for reproducible summaries.
        """
        results = {}
        rng = np.random.default_rng(int(seed))
        for cid, ctype in types.items():
            idx = np.where(labels == cid)[0]
            if len(idx) == 0:
                continue
            rm = rng.choice(idx, min(int(n_rm), len(idx)), replace=False).tolist()
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

    def compute_damages(self, n_ch: int, frac: float = 0.2, seed: int = 0) -> np.ndarray:
        """Compute true per-channel damage.

        Uses a fixed RNG seed by default for reproducible comparisons.
        """
        damages = np.zeros(n_ch)
        rng = np.random.default_rng(int(seed))
        test_idx = rng.choice(int(n_ch), max(1, int(n_ch * frac)), replace=False)
        for i in test_idx:
            r = self.cascade.ablate(self.layer, [int(i)])
            # Use loss increase as a smoother "damage" signal than accuracy drop,
            # especially when evaluating on a small test subset.
            damages[i] = r.loss_increase
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
        # In some pruning workflows we treat `scores` as a *prune score* where higher
        # means "safer to remove". A good prune score should correlate with
        # *lower* damage; we therefore correlate against -d so higher rho is better.
        rho, _ = stats.spearmanr(s, -d)
        recall = {}
        # Recall@k: how well the prune score identifies the least-damaging channels.
        by_d = np.argsort(d)      # least damaging first
        by_s = np.argsort(-s)     # highest prune score first
        for k in top_ks:
            # Keep the dictionary key as the *requested* k for stable downstream
            # table formatting, but clamp the effective k to the number of
            # evaluated channels.
            k_eff = min(int(k), len(d))
            overlap = len(set(by_d[:k_eff]) & set(by_s[:k_eff]))
            recall[int(k)] = overlap / k_eff if k_eff > 0 else 0.0
        return DamageResult(self.layer, method, float(rho) if not np.isnan(rho) else 0., recall)
