"""
Metric-Based Pruning Strategies
===============================

A comprehensive set of pruning methods using individual metrics and their combinations.

Categories:
1. SINGLE METRIC: Prune by one metric (RQ, redundancy, synergy, MI)
2. TAYLOR-WEIGHTED: Combine Taylor sensitivity with each metric
3. LP-OPTIMAL: Learn weights that predict Fisher/Loss-Proxy importance
4. CLUSTER-STRUCTURE: Use cluster membership in scoring (not just selection)

All methods follow the convention: HIGHER score = MORE important (keep).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base import BasePruningStrategy, PruningConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MetricPruningConfig(PruningConfig):
    """Configuration for metric-based pruning."""

    # Which metric(s) to use
    metric: str = "rq"  # rq, redundancy, synergy, mi, composite

    # For composite: weights for each component
    weight_rq: float = 1.0
    weight_redundancy: float = -0.3  # Negative = high redundancy -> low score
    weight_synergy: float = 0.5
    weight_mi: float = 0.0  # MI with task (logit margin)

    # Taylor blending
    taylor_weight: float = 0.0  # 0 = no Taylor, 1 = pure Taylor
    taylor_blend_mode: str = "linear"  # linear, geometric, max

    # LP-optimal learning
    lp_optimal: bool = False  # If True, learn weights from LP correlation

    # Cluster-structure scoring
    use_cluster_structure: bool = False  # Add cluster-type-based bonus/penalty
    critical_bonus: float = 0.5  # Bonus for critical channels
    redundant_penalty: float = 0.3  # Penalty for redundant channels
    synergistic_bonus: float = 0.2  # Bonus for synergistic channels
    background_penalty: float = 0.1  # Penalty for background channels

    # Two-axis score (Rule 1): score = alpha * I_X - beta * I(T;Y)
    # High score = keep; low score = prune.
    # alpha=1, beta=0 recovers pure local I_X ranking.
    # beta>0 penalises task-relevant channels (Q2 danger zone).
    alpha_ix: float = 1.0
    beta_ity: float = 0.0


# =============================================================================
# SINGLE METRIC PRUNING
# =============================================================================


class SingleMetricPruning(BasePruningStrategy):
    """
    Prune based on a single metric.

    Available metrics:
    - 'rq' or 'rayleigh_quotient': log(RQ) - channel uniqueness
    - 'redundancy' or 'mi_redundancy': -redundancy (high redundancy = low score)
    - 'synergy': synergy with other channels for task
    - 'mi' or 'task_mi': MI(channel, logit_margin) - direct task relevance
    - 'mi_in' or 'mi_in_proxy': input-MI proxy = 0.5 * log(1 + RQ * ||w||^2 / sigma0^2)
    - 'composite': weighted combination of (logRQ, -redundancy, synergy, task_mi)
    - 'magnitude': activation/weight magnitude (baseline)
    """

    def __init__(
        self,
        config: Optional[MetricPruningConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(config or MetricPruningConfig())
        self.config: MetricPruningConfig
        self.precomputed_metrics = precomputed_metrics or {}

    def compute_importance_scores(
        self,
        module: nn.Module,
        layer_name: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute scores based on a single metric."""

        n_channels = module.weight.shape[0] if hasattr(module, "weight") else 1
        device = module.weight.device if hasattr(module, "weight") else "cpu"

        metric = self.config.metric.lower()
        metrics = self.precomputed_metrics

        # Get the raw metric values
        if metric in {"rq", "rayleigh_quotient"}:
            raw = metrics.get("rq", metrics.get("rayleigh_quotient", np.ones(n_channels)))
            # Use log(RQ) since RQ can span orders of magnitude
            scores = np.log(np.clip(raw, 1e-10, None))

        elif metric in {"redundancy", "mi_redundancy", "red"}:
            raw = metrics.get("redundancy", np.zeros(n_channels))
            # NEGATIVE because high redundancy = can be pruned
            scores = -raw

        elif metric in {"synergy", "syn"}:
            scores = metrics.get("synergy", np.zeros(n_channels))

        elif metric in {"mi", "task_mi", "mi_task"}:
            # TaskMI is stored as "task_mi" in this codebase; keep "mi_task" as backward-compat alias.
            scores = metrics.get("task_mi", metrics.get("mi_task", np.zeros(n_channels)))

        elif metric in {"mi_in", "mi_in_proxy", "input_mi"}:
            scores = metrics.get("mi_in_proxy", np.zeros(n_channels))

        elif metric in {"composite", "combo"}:
            # Weighted sum in normalized space (higher = more important):
            #   score = w_rq*logRQ + w_syn*Syn + w_red*Red + w_mi*TaskMI
            # NOTE: w_red is typically negative so high redundancy lowers importance.
            raw_rq = metrics.get("rq", metrics.get("rayleigh_quotient", np.ones(n_channels)))
            log_rq = np.log(np.clip(raw_rq, 1e-10, None))
            red = metrics.get("redundancy", np.zeros(n_channels))
            syn = metrics.get("synergy", np.zeros(n_channels))
            tmi = metrics.get("task_mi", metrics.get("mi_task", np.zeros(n_channels)))

            scores = (
                self.config.weight_rq * self._normalize(log_rq)
                + self.config.weight_synergy * self._normalize(syn)
                + self.config.weight_redundancy * self._normalize(red)
                + self.config.weight_mi * self._normalize(tmi)
            )

        elif metric in {"two_axis", "ix_minus_ity", "two_axis_score"}:
            # Rule 1 score: alpha * I_X - beta * I(T;Y).
            # Uses the same mi_in_proxy field already populated by the metric pipeline
            # (Gaussian input-capture proxy, i.e. I_X in the paper notation) and the
            # task_mi field (task-direction MI). Both are normalized before combining
            # so that alpha/beta have comparable scales.
            ix = metrics.get("mi_in_proxy", np.zeros(n_channels))
            ity = metrics.get("task_mi", metrics.get("mi_task", np.zeros(n_channels)))
            alpha = float(getattr(self.config, "alpha_ix", 1.0))
            beta = float(getattr(self.config, "beta_ity", 0.0))
            scores = alpha * self._normalize(ix) - beta * self._normalize(ity)

        elif metric in {"magnitude", "mag"}:
            if hasattr(module, "weight"):
                w = module.weight.detach().view(n_channels, -1)
                scores = w.norm(p=2, dim=1).cpu().numpy()
            else:
                scores = np.ones(n_channels)

        else:
            logger.warning(f"Unknown metric '{metric}', using RQ")
            raw = metrics.get("rq", np.ones(n_channels))
            scores = np.log(np.clip(raw, 1e-10, None))

        # Normalize to [0, 1]
        scores = self._normalize(scores)

        return torch.from_numpy(scores).float().to(device)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1]."""
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)


# =============================================================================
# TAYLOR-WEIGHTED METRIC PRUNING
# =============================================================================


class TaylorWeightedMetricPruning(SingleMetricPruning):
    """
    Combine any metric with Taylor (gradient-based) sensitivity.

    Modes:
    - 'linear': (1-w)*metric + w*taylor
    - 'geometric': sqrt(metric * taylor)
    - 'product': metric * taylor (channels must be both metric-important AND loss-sensitive)
    - 'max': max(metric, taylor) (channels important by either criterion)
    """

    def __init__(
        self,
        config: Optional[MetricPruningConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
        taylor_scores: Optional[np.ndarray] = None,
    ):
        super().__init__(config, precomputed_metrics)
        self.taylor_scores = taylor_scores

    def compute_importance_scores(
        self,
        module: nn.Module,
        layer_name: str = "",
        taylor_scores: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Taylor-weighted metric scores."""

        # Get base metric scores
        metric_scores = super().compute_importance_scores(module, layer_name, **kwargs)

        # Get Taylor scores
        taylor = taylor_scores if taylor_scores is not None else self.taylor_scores
        if taylor is None:
            # Fallback: use magnitude as proxy
            n_channels = module.weight.shape[0]
            w = module.weight.detach().view(n_channels, -1)
            taylor = w.norm(p=2, dim=1).cpu().numpy()

        taylor_norm = self._normalize(np.asarray(taylor))
        taylor_t = torch.from_numpy(taylor_norm).float().to(metric_scores.device)

        # Combine based on mode
        w = self.config.taylor_weight
        mode = self.config.taylor_blend_mode.lower()

        if mode == "linear":
            combined = (1 - w) * metric_scores + w * taylor_t
        elif mode == "geometric":
            combined = torch.sqrt(metric_scores * taylor_t + 1e-8)
        elif mode == "product":
            combined = metric_scores * taylor_t
        elif mode == "max":
            combined = torch.maximum(metric_scores, taylor_t)
        else:
            combined = (1 - w) * metric_scores + w * taylor_t

        return combined


# =============================================================================
# LP-OPTIMAL PRUNING (Learn weights from LP correlation)
# =============================================================================


class LPOptimalPruning(BasePruningStrategy):
    """
    Learn metric weights that best predict Fisher/LP importance.

    Given precomputed LP scores and metrics (RQ, redundancy, synergy), finds
    the linear combination that maximizes correlation with LP:

        score_i = w_rq * log(RQ_i) + w_red * (-red_i) + w_syn * syn_i + ...

    The weights are learned via least-squares regression on the training data.
    """

    def __init__(
        self,
        config: Optional[MetricPruningConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
        lp_scores: Optional[np.ndarray] = None,
    ):
        super().__init__(config or MetricPruningConfig(lp_optimal=True))
        self.config: MetricPruningConfig
        self.precomputed_metrics = precomputed_metrics or {}
        self.lp_scores = lp_scores
        self._learned_weights = None

    def learn_weights(
        self,
        metrics: Dict[str, np.ndarray],
        lp: np.ndarray,
    ) -> Dict[str, float]:
        """
        Learn optimal weights via least-squares regression.

        Args:
            metrics: Dict with 'rq', 'redundancy', 'synergy', etc.
            lp: LP (Fisher) importance scores

        Returns:
            Dict mapping metric name to learned weight
        """
        # Build feature matrix
        features = []
        feature_names = []

        n = len(lp)

        if "rq" in metrics:
            log_rq = np.log(np.clip(metrics["rq"][:n], 1e-10, None))
            features.append(self._normalize(log_rq))
            feature_names.append("rq")

        if "redundancy" in metrics:
            red = metrics["redundancy"][:n]
            features.append(-self._normalize(red))  # Negative
            feature_names.append("redundancy")

        if "synergy" in metrics:
            syn = metrics["synergy"][:n]
            features.append(self._normalize(syn))
            feature_names.append("synergy")

        mi_arr = metrics.get("task_mi", metrics.get("mi_task"))
        if mi_arr is not None:
            mi = mi_arr[:n]
            features.append(self._normalize(mi))
            feature_names.append("task_mi")

        if not features:
            return {}

        X = np.column_stack(features)
        y = self._normalize(lp)

        # Least-squares with regularization
        try:
            from scipy.linalg import lstsq

            weights, _, _, _ = lstsq(X, y)
        except ImportError:
            # Fallback: normal equations
            XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
            weights = np.linalg.solve(XtX, X.T @ y)

        self._learned_weights = dict(zip(feature_names, weights))

        # Compute correlation for logging
        pred = X @ weights
        corr = np.corrcoef(pred, y)[0, 1]
        logger.info(f"LP-optimal weights learned: {self._learned_weights}, correlation: {corr:.3f}")

        return self._learned_weights

    def compute_importance_scores(
        self,
        module: nn.Module,
        layer_name: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute scores using learned or default weights."""

        n_channels = module.weight.shape[0] if hasattr(module, "weight") else 1
        device = module.weight.device if hasattr(module, "weight") else "cpu"

        metrics = self.precomputed_metrics

        # Use learned weights if available
        if self._learned_weights is not None:
            weights = self._learned_weights
        else:
            # Default weights from config
            weights = {
                "rq": self.config.weight_rq,
                "redundancy": self.config.weight_redundancy,
                "synergy": self.config.weight_synergy,
                "task_mi": self.config.weight_mi,
            }

        # Compute weighted sum
        scores = np.zeros(n_channels)

        if "rq" in metrics and "rq" in weights:
            log_rq = np.log(np.clip(metrics["rq"][:n_channels], 1e-10, None))
            scores += weights["rq"] * self._normalize(log_rq)

        if "redundancy" in metrics and "redundancy" in weights:
            red = metrics["redundancy"][:n_channels]
            scores += weights["redundancy"] * self._normalize(red)  # Weight already includes sign

        if "synergy" in metrics and "synergy" in weights:
            syn = metrics["synergy"][:n_channels]
            scores += weights["synergy"] * self._normalize(syn)

        mi_arr = metrics.get("task_mi", metrics.get("mi_task"))
        if mi_arr is not None and "task_mi" in weights:
            mi = mi_arr[:n_channels]
            scores += weights["task_mi"] * self._normalize(mi)

        return torch.from_numpy(scores).float().to(device)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)


# =============================================================================
# CLUSTER-STRUCTURE SCORING
# =============================================================================


class ClusterStructurePruning(BasePruningStrategy):
    """
    Use cluster membership directly in scoring (not just selection constraints).

    Score = base_metric_score + cluster_bonus/penalty

    Where:
    - Critical channels get a bonus
    - Redundant channels get a penalty
    - Synergistic channels get a small bonus
    - Background channels get a small penalty

    This is different from constraint-based cluster-aware pruning because:
    - Constraints BLOCK certain channels from being pruned
    - This method ADJUSTS scores to make cluster membership affect ordering
    """

    def __init__(
        self,
        config: Optional[MetricPruningConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
        precomputed_clusters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config or MetricPruningConfig(use_cluster_structure=True))
        self.config: MetricPruningConfig
        self.precomputed_metrics = precomputed_metrics or {}
        self.precomputed_clusters = precomputed_clusters or {}

    def compute_importance_scores(
        self,
        module: nn.Module,
        layer_name: str = "",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute scores with cluster-structure bonuses/penalties."""

        n_channels = module.weight.shape[0] if hasattr(module, "weight") else 1
        device = module.weight.device if hasattr(module, "weight") else "cpu"

        metrics = self.precomputed_metrics
        clusters = self.precomputed_clusters

        # Base score from composite metric
        scores = np.zeros(n_channels)

        if "rq" in metrics:
            log_rq = np.log(np.clip(metrics["rq"][:n_channels], 1e-10, None))
            scores += self.config.weight_rq * self._normalize(log_rq)

        if "redundancy" in metrics:
            red = metrics["redundancy"][:n_channels]
            scores += self.config.weight_redundancy * self._normalize(red)

        if "synergy" in metrics:
            syn = metrics["synergy"][:n_channels]
            scores += self.config.weight_synergy * self._normalize(syn)

        # Normalize base scores to [0, 1]
        scores = self._normalize(scores)

        # Add cluster-structure bonuses/penalties
        labels = np.asarray(clusters.get("labels", np.zeros(n_channels, dtype=int)))[:n_channels]
        type_mapping = clusters.get("type_mapping", {})

        # Invert mapping: type_name -> cluster_id
        type_to_id = {v: int(k) for k, v in type_mapping.items()}

        for channel_type, adjustment in [
            ("critical", self.config.critical_bonus),
            ("redundant", -self.config.redundant_penalty),
            ("synergistic", self.config.synergistic_bonus),
            ("background", -self.config.background_penalty),
        ]:
            cluster_id = type_to_id.get(channel_type, -1)
            if cluster_id >= 0:
                mask = labels == cluster_id
                scores[mask] += adjustment

        return torch.from_numpy(scores).float().to(device)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64).ravel()
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_metric_pruning_strategy(
    method: str,
    precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
    precomputed_clusters: Optional[Dict[str, Any]] = None,
    taylor_scores: Optional[np.ndarray] = None,
    lp_scores: Optional[np.ndarray] = None,
    **config_kwargs,
) -> BasePruningStrategy:
    """
    Factory function to create metric-based pruning strategies.

    Method names:
    - Single metric: 'rq', 'redundancy', 'synergy', 'mi', 'magnitude'
    - Taylor-weighted: 'taylor_rq', 'taylor_redundancy', 'taylor_synergy', etc.
    - Taylor-act-weighted: 'taylor_act_rq', 'taylor_act_redundancy', ... (same blends; different Taylor source)
    - LP-optimal: 'lp_optimal'
    - Cluster-structure: 'cluster_structure'
    - Composite: 'composite' (default linear combination)
    """
    method = method.lower()

    # Single metric methods
    single_metrics = {
        "rq",
        "redundancy",
        "synergy",
        "mi",
        "mi_in",
        "magnitude",
        "composite",
        "two_axis",
        "ix_minus_ity",
        "two_axis_score",
    }

    # Two-axis variants with embedded alpha/beta so configs can select them by
    # name alone (no custom kwarg threading through the experiment runner).
    # Convention: two_axis_aA_bB encodes alpha=A, beta=B with 'p' as decimal point.
    # Examples: two_axis_a1_b0p25 (alpha=1, beta=0.25), two_axis_a1_b1 (alpha=1, beta=1).
    if method.startswith("two_axis_a"):
        try:
            remainder = method[len("two_axis_a") :]
            a_str, b_str = remainder.split("_b")
            alpha = float(a_str.replace("p", "."))
            beta = float(b_str.replace("p", "."))
            config = MetricPruningConfig(
                metric="two_axis",
                alpha_ix=alpha,
                beta_ity=beta,
                **config_kwargs,
            )
            return SingleMetricPruning(config, precomputed_metrics)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse two-axis variant '{method}', falling through")

    if method in single_metrics:
        config = MetricPruningConfig(metric=method, **config_kwargs)
        return SingleMetricPruning(config, precomputed_metrics)

    # Taylor-act-weighted methods (identical logic; expects taylor_scores to be activation-based)
    if method.startswith("taylor_act_"):
        base_metric = method[len("taylor_act_") :]  # Remove 'taylor_act_' prefix
        if base_metric not in single_metrics:
            base_metric = "rq"
        config = MetricPruningConfig(
            metric=base_metric,
            taylor_weight=config_kwargs.pop("taylor_weight", 0.5),
            taylor_blend_mode=config_kwargs.pop("taylor_blend_mode", "geometric"),
            **config_kwargs,
        )
        return TaylorWeightedMetricPruning(config, precomputed_metrics, taylor_scores)

    # Taylor-weighted methods
    if method.startswith("taylor_"):
        base_metric = method[7:]  # Remove 'taylor_' prefix
        if base_metric not in single_metrics:
            base_metric = "rq"
        config = MetricPruningConfig(
            metric=base_metric,
            taylor_weight=config_kwargs.pop("taylor_weight", 0.5),
            taylor_blend_mode=config_kwargs.pop("taylor_blend_mode", "geometric"),
            **config_kwargs,
        )
        return TaylorWeightedMetricPruning(config, precomputed_metrics, taylor_scores)

    # LP-optimal
    if method == "lp_optimal":
        config = MetricPruningConfig(lp_optimal=True, **config_kwargs)
        strategy = LPOptimalPruning(config, precomputed_metrics, lp_scores)
        # Learn weights if LP scores provided
        if lp_scores is not None and precomputed_metrics:
            strategy.learn_weights(precomputed_metrics, lp_scores)
        return strategy

    # Cluster-structure
    if method in {"cluster_structure", "cluster_scoring"}:
        config = MetricPruningConfig(use_cluster_structure=True, **config_kwargs)
        return ClusterStructurePruning(config, precomputed_metrics, precomputed_clusters)

    # Default: composite
    config = MetricPruningConfig(metric="composite", **config_kwargs)
    return SingleMetricPruning(config, precomputed_metrics)
