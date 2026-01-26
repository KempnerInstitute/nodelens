"""
Generalized Taylor-like Pruning Strategies
===========================================

Taylor importance is defined as: |∂L/∂a · a|

This module generalizes Taylor in several ways:

1. STRUCTURE-WEIGHTED TAYLOR: Taylor × f(structural_metric)
   - The gradient tells us loss-sensitivity, but we weight by structure

2. STRUCTURAL TAYLOR: Replace activation with structural importance
   - |∂L/∂a| × structural_score (gradient sensitivity × structural importance)

3. TAYLOR AS OPTIMAL WEIGHT: Include Taylor in the LP-optimal regression
   - Find: w_t*Taylor + w_rq*RQ + w_red*(-red) + w_syn*syn that best predicts LP

4. CLUSTER-TYPE TAYLOR: Weight Taylor by cluster membership
   - Critical channels get Taylor×1.5, redundant get Taylor×0.5, etc.

5. MI-TAYLOR: Taylor × MI(channel, task)
   - Channels must be both loss-sensitive AND task-informative
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..base import PruningConfig, BasePruningStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# ANALYTICAL DEFINITIONS
# =============================================================================

"""
TAYLOR IMPORTANCE - Formal Definition
======================================

Given:
- L: loss function
- a_i: activation of channel i (scalar or aggregated from spatial dims)
- θ: model parameters

Standard Taylor:
    Taylor_i = |∂L/∂a_i · a_i|
    
    Intuition: First-order approximation of loss change when a_i → 0
    L(a_i=0) - L(a_i) ≈ -∂L/∂a_i · a_i

    In practice, we compute:
    - Forward pass: get a_i
    - Backward pass: get ∂L/∂a_i
    - Score: |∂L/∂a_i · a_i|, often averaged over batch/spatial dims

GENERALIZED TAYLOR
==================

1. RQ-Weighted Taylor:
    score_i = |∂L/∂a_i · a_i| × log(RQ_i)
    
    Intuition: Channels that are loss-sensitive AND unique (high RQ) are more important.
    A redundant channel might have high Taylor but can be replaced by correlated channels.

2. Redundancy-Discounted Taylor:
    score_i = |∂L/∂a_i · a_i| / (1 + β·redundancy_i)
    
    Intuition: High-redundancy channels have their Taylor score discounted because
    their information can be recovered from other channels.

3. Synergy-Boosted Taylor:
    score_i = |∂L/∂a_i · a_i| × (1 + γ·synergy_i)
    
    Intuition: Channels that cooperate with others for the task get a boost.

4. Structural Taylor (replaces activation with structural importance):
    score_i = |∂L/∂a_i| × structural_score_i
    
    where structural_score = α·log(RQ) + β·synergy - γ·redundancy
    
    Intuition: Instead of "gradient × activation", use "gradient × how structurally 
    important this channel is". This separates gradient sensitivity from magnitude.

5. MI-Taylor:
    score_i = |∂L/∂a_i · a_i| × MI(a_i; T)
    
    where T is the task target (logit margin).
    
    Intuition: Channels must be both loss-sensitive AND directly informative about T.

6. Full Generalized Taylor:
    score_i = |∂L/∂a_i|^α × |a_i|^β × f(RQ_i, red_i, syn_i, type_i)
    
    This is the most general form, allowing independent weighting of:
    - Gradient magnitude (loss sensitivity)
    - Activation magnitude  
    - Structural metrics
"""


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeneralizedTaylorConfig(PruningConfig):
    """Configuration for generalized Taylor pruning."""
    
    # Which variant to use
    variant: str = "structural_taylor"  # See VARIANTS below
    
    # Structural metric weights (for variants that use them)
    weight_rq: float = 1.0
    weight_redundancy: float = 0.3  # Will be used as penalty
    weight_synergy: float = 0.5
    
    # Exponents for the full generalized form
    gradient_exponent: float = 1.0  # α in |∇L|^α
    activation_exponent: float = 1.0  # β in |a|^β
    
    # For redundancy-discounted Taylor
    redundancy_discount_beta: float = 1.0
    
    # For synergy-boosted Taylor
    synergy_boost_gamma: float = 0.5
    
    # For cluster-type Taylor
    critical_multiplier: float = 1.5
    redundant_multiplier: float = 0.5
    synergistic_multiplier: float = 1.2
    background_multiplier: float = 0.8

    # Numerical stability / scale parameters (kept explicit so they can be config-driven)
    rq_log_eps: float = 1e-10  # clip floor for log(RQ)
    structural_eps: float = 0.1  # additive eps used in multiplicative structural factors (non-gated variants)
    grad_over_act_eps: float = 1e-8  # eps for grad≈taylor/|act| approximation
    lp_optimal_l2_reg: float = 0.01  # ridge term for taylor_optimal_combo least-squares

    # For metric-gated Taylor (Taylor * gate(metrics[, clusters]))
    gate_mode: str = "sigmoid"  # "linear" | "sigmoid"
    gate_temperature: float = 6.0  # slope for sigmoid gate
    gate_bias: float = 0.5  # center for sigmoid gate in normalized structural space
    gate_eps: float = 0.05  # avoid exact zeros when multiplying
    gate_min: float = 0.0  # clamp lower bound after gating
    gate_include_cluster_multiplier: bool = True
    
    # For LP-optimal with Taylor
    include_taylor_in_lp_optimal: bool = True


# Variant names
VARIANTS = {
    "taylor": "Standard Taylor: |∂L/∂a · a|",
    "rq_weighted_taylor": "Taylor × log(RQ)",
    "redundancy_discounted_taylor": "Taylor / (1 + β·redundancy)",
    "synergy_boosted_taylor": "Taylor × (1 + γ·synergy)",
    "structural_taylor": "|∂L/∂a| × structural_score",
    "metric_gated_taylor": "Taylor × gate(structural_score[, cluster_type])",
    "mi_taylor": "Taylor × MI(channel, task)",
    "cluster_type_taylor": "Taylor × type_multiplier",
    "full_generalized": "|∇L|^α × |a|^β × f(metrics)",
    "taylor_optimal_combo": "Learn: w_t·Taylor + w_rq·RQ + w_r·(-red) + w_s·syn",
}


# =============================================================================
# IMPLEMENTATION
# =============================================================================

class GeneralizedTaylorPruning(BasePruningStrategy):
    """
    Generalized Taylor-like pruning that combines gradient sensitivity with 
    structural metrics.
    """
    
    def __init__(
        self,
        config: Optional[GeneralizedTaylorConfig] = None,
        precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
        precomputed_clusters: Optional[Dict[str, Any]] = None,
        taylor_scores: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
        activations: Optional[np.ndarray] = None,
    ):
        super().__init__(config or GeneralizedTaylorConfig())
        self.config: GeneralizedTaylorConfig
        self.precomputed_metrics = precomputed_metrics or {}
        self.precomputed_clusters = precomputed_clusters or {}
        self.taylor_scores = taylor_scores
        self.gradients = gradients  # |∂L/∂a| per channel
        self.activations = activations  # |a| per channel
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        layer_name: str = "",
        taylor_scores: Optional[np.ndarray] = None,
        gradients: Optional[np.ndarray] = None,
        activations: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute generalized Taylor importance scores."""
        
        n_channels = module.weight.shape[0] if hasattr(module, 'weight') else 1
        device = module.weight.device if hasattr(module, 'weight') else 'cpu'
        
        # Get Taylor scores (or compute proxy)
        taylor = taylor_scores if taylor_scores is not None else self.taylor_scores
        if taylor is None:
            # Fallback: use magnitude as Taylor proxy
            if hasattr(module, 'weight'):
                w = module.weight.detach().view(n_channels, -1)
                taylor = w.norm(p=2, dim=1).cpu().numpy()
            else:
                taylor = np.ones(n_channels)
        taylor = np.asarray(taylor)[:n_channels]
        
        # Get gradient/activation if provided
        grad = gradients if gradients is not None else self.gradients
        act = activations if activations is not None else self.activations
        
        # Get structural metrics
        metrics = self.precomputed_metrics
        rq = np.log(np.clip(metrics.get('rq', np.ones(n_channels))[:n_channels], float(self.config.rq_log_eps), None))
        red = metrics.get('redundancy', np.zeros(n_channels))[:n_channels]
        syn = metrics.get('synergy', np.zeros(n_channels))[:n_channels]
        # NOTE: in this codebase, TaskMI is stored as "task_mi" in layer metrics.
        mi_task = metrics.get('task_mi', metrics.get('mi_task', np.zeros(n_channels)))[:n_channels]
        
        # Normalize for stable combination
        taylor_norm = self._normalize(taylor)
        rq_norm = self._normalize(rq)
        red_norm = self._normalize(red)
        syn_norm = self._normalize(syn)
        mi_norm = self._normalize(mi_task)
        
        # Dispatch to variant
        variant = self.config.variant.lower()
        
        if variant == "taylor":
            scores = taylor_norm
            
        elif variant == "rq_weighted_taylor":
            # Taylor × log(RQ): unique AND loss-sensitive
            scores = taylor_norm * (rq_norm + float(self.config.structural_eps))
            
        elif variant == "redundancy_discounted_taylor":
            # Taylor / (1 + β·redundancy): discount redundant channels
            beta = self.config.redundancy_discount_beta
            scores = taylor_norm / (1 + beta * red_norm)
            
        elif variant == "synergy_boosted_taylor":
            # Taylor × (1 + γ·synergy): boost synergistic channels
            gamma = self.config.synergy_boost_gamma
            scores = taylor_norm * (1 + gamma * syn_norm)
            
        elif variant == "structural_taylor":
            # |∂L/∂a| × structural_score
            # Use gradient if available, else use Taylor / activation proxy
            if grad is not None:
                grad_norm = self._normalize(np.asarray(grad)[:n_channels])
            else:
                # Approximate: Taylor ≈ grad × activation, so grad ≈ Taylor / activation
                if act is not None:
                    act_arr = np.asarray(act)[:n_channels]
                    grad_norm = self._normalize(taylor / (np.abs(act_arr) + float(self.config.grad_over_act_eps)))
                else:
                    grad_norm = taylor_norm  # Use Taylor as proxy
            
            # Compute structural score
            structural = (
                self.config.weight_rq * rq_norm +
                self.config.weight_synergy * syn_norm -
                self.config.weight_redundancy * red_norm
            )
            structural_norm = self._normalize(structural)
            
            # Combine: gradient sensitivity × structural importance
            scores = grad_norm * (structural_norm + float(self.config.structural_eps))

        elif variant in {"metric_gated_taylor", "gated_taylor"}:
            # Taylor × gate(structural_score[, cluster_type])
            #
            # This is the "gate-based Taylor" hybrid:
            #   score_i = |∂L/∂g_i · g_i| where g_i = gate(structural metrics)
            # Under a multiplicative gate on channel output, ∂L/∂g_i is proportional
            # to the standard Taylor term |∂L/∂a_i · a_i|, so we implement:
            #   score_i = Taylor_i × gate_i
            structural = (
                self.config.weight_rq * rq_norm
                + self.config.weight_synergy * syn_norm
                - self.config.weight_redundancy * red_norm
            )
            structural_norm = self._normalize(structural)

            gate_mode = str(self.config.gate_mode).lower()
            if gate_mode == "sigmoid":
                z = self.config.gate_temperature * (structural_norm - float(self.config.gate_bias))
                gate = 1.0 / (1.0 + np.exp(-z))
            else:
                # "linear" (or unknown): just use normalized structural score as gate
                gate = structural_norm

            gate = np.clip(gate, float(self.config.gate_min), None)

            if bool(self.config.gate_include_cluster_multiplier):
                clusters = self.precomputed_clusters or {}
                labels = np.asarray(clusters.get("labels", np.zeros(n_channels, dtype=int)))[:n_channels]
                type_mapping = clusters.get("type_mapping", {}) or {}

                # type_mapping usually maps cluster_id -> type_name
                type_to_id = {v: int(k) for k, v in type_mapping.items()}
                multipliers = np.ones(n_channels, dtype=np.float64)
                for type_name, mult in [
                    ("critical", self.config.critical_multiplier),
                    ("redundant", self.config.redundant_multiplier),
                    ("synergistic", self.config.synergistic_multiplier),
                    ("background", self.config.background_multiplier),
                ]:
                    cid = type_to_id.get(type_name, -1)
                    if cid >= 0:
                        multipliers[labels == cid] = float(mult)
                gate = gate * multipliers

            scores = taylor_norm * (gate + float(self.config.gate_eps))
            
        elif variant == "mi_taylor":
            # Taylor × MI(channel, task)
            scores = taylor_norm * (mi_norm + float(self.config.structural_eps))
            
        elif variant == "cluster_type_taylor":
            # Taylor × type_multiplier based on cluster membership
            clusters = self.precomputed_clusters
            labels = np.asarray(clusters.get('labels', np.zeros(n_channels, dtype=int)))[:n_channels]
            type_mapping = clusters.get('type_mapping', {})
            
            # Build type-to-multiplier map
            type_to_id = {v: int(k) for k, v in type_mapping.items()}
            multipliers = np.ones(n_channels)
            
            for type_name, mult in [
                ('critical', self.config.critical_multiplier),
                ('redundant', self.config.redundant_multiplier),
                ('synergistic', self.config.synergistic_multiplier),
                ('background', self.config.background_multiplier),
            ]:
                cluster_id = type_to_id.get(type_name, -1)
                if cluster_id >= 0:
                    mask = labels == cluster_id
                    multipliers[mask] = mult
            
            scores = taylor_norm * multipliers
            
        elif variant == "full_generalized":
            # |∇L|^α × |a|^β × f(metrics)
            alpha = self.config.gradient_exponent
            beta = self.config.activation_exponent
            
            if grad is not None:
                grad_term = np.abs(np.asarray(grad)[:n_channels]) ** alpha
            else:
                grad_term = np.ones(n_channels)
            
            if act is not None:
                act_term = np.abs(np.asarray(act)[:n_channels]) ** beta
            else:
                act_term = np.ones(n_channels)
            
            # f(metrics) = structural score
            structural = (
                self.config.weight_rq * rq_norm +
                self.config.weight_synergy * syn_norm -
                self.config.weight_redundancy * red_norm
            )
            
            scores = (
                self._normalize(grad_term)
                * self._normalize(act_term)
                * (self._normalize(structural) + float(self.config.structural_eps))
            )
            
        elif variant == "taylor_optimal_combo":
            # Learn: w_t·Taylor + w_rq·RQ + w_r·(-red) + w_s·syn from LP
            lp = metrics.get('loss_proxy', metrics.get('lp', metrics.get('fisher')))
            
            if lp is not None:
                # Build feature matrix
                X = np.column_stack([taylor_norm, rq_norm, -red_norm, syn_norm])
                y = self._normalize(lp[:n_channels])
                
                # Least-squares fit
                try:
                    XtX = X.T @ X + float(self.config.lp_optimal_l2_reg) * np.eye(4)
                    weights = np.linalg.solve(XtX, X.T @ y)
                    scores = X @ weights
                    logger.info(f"Taylor-optimal weights: Taylor={weights[0]:.3f}, RQ={weights[1]:.3f}, "
                               f"Red={weights[2]:.3f}, Syn={weights[3]:.3f}")
                except Exception:
                    scores = taylor_norm
            else:
                # Fallback: equal-weight combination
                scores = 0.4 * taylor_norm + 0.3 * rq_norm - 0.2 * red_norm + 0.1 * syn_norm
        
        else:
            logger.warning(f"Unknown variant '{variant}', using standard Taylor")
            scores = taylor_norm
        
        # Final normalization
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
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_generalized_taylor(
    variant: str,
    precomputed_metrics: Optional[Dict[str, np.ndarray]] = None,
    precomputed_clusters: Optional[Dict[str, Any]] = None,
    taylor_scores: Optional[np.ndarray] = None,
    **config_kwargs,
) -> GeneralizedTaylorPruning:
    """
    Factory function to create generalized Taylor pruning.
    
    Variants:
    - 'taylor': Standard Taylor
    - 'rq_weighted_taylor': Taylor × log(RQ)
    - 'redundancy_discounted_taylor': Taylor / (1 + β·redundancy)
    - 'synergy_boosted_taylor': Taylor × (1 + γ·synergy)
    - 'structural_taylor': |∂L/∂a| × structural_score
    - 'metric_gated_taylor': Taylor × gate(structural_score[, cluster_type])
    - 'mi_taylor': Taylor × MI(channel, task)
    - 'cluster_type_taylor': Taylor × type_multiplier
    - 'full_generalized': |∇L|^α × |a|^β × f(metrics)
    - 'taylor_optimal_combo': Learn optimal combination
    """
    config = GeneralizedTaylorConfig(variant=variant, **config_kwargs)
    return GeneralizedTaylorPruning(
        config=config,
        precomputed_metrics=precomputed_metrics,
        precomputed_clusters=precomputed_clusters,
        taylor_scores=taylor_scores,
    )


# =============================================================================
# ANALYTICAL DERIVATION NOTES
# =============================================================================

"""
WHY THESE GENERALIZATIONS MAKE SENSE
=====================================

1. RQ-Weighted Taylor: Taylor × log(RQ)
   ----------------------------------------
   - Taylor tells us: "how much does removing this channel affect the loss?"
   - RQ tells us: "how unique is this channel's representation?"
   - Problem with pure Taylor: a redundant channel might have high Taylor because 
     it carries important information, but that information can be recovered from 
     correlated channels after fine-tuning.
   - Solution: Weight Taylor by RQ so we prefer channels that are BOTH loss-sensitive 
     AND carry unique information.

2. Redundancy-Discounted Taylor: Taylor / (1 + β·redundancy)
   ----------------------------------------------------------
   - Explicitly discount channels with high redundancy.
   - Even if Taylor is high, if redundancy is high, the channel's information
     can be recovered from others.
   - β controls how much to discount.

3. Structural Taylor: |∂L/∂a| × structural_score
   -----------------------------------------------
   - Standard Taylor = |gradient| × |activation|
   - Structural Taylor = |gradient| × (structural importance)
   - This separates:
     - Gradient: how sensitive is the loss to this channel?
     - Structural importance: is this channel unique/synergistic/non-redundant?
   - Key insight: activation magnitude might not reflect actual importance.
     A high-activation channel could be redundant.

4. Cluster-Type Taylor: Taylor × type_multiplier
   -----------------------------------------------
   - Uses cluster semantics to weight Taylor:
     - Critical: high multiplier (we really don't want to prune these)
     - Redundant: low multiplier (OK to prune even if high Taylor)
     - Synergistic: moderate-high multiplier (cooperates for task)
     - Background: moderate-low multiplier (low variance, less important)

5. Taylor-Optimal Combo: w_t·Taylor + w_rq·RQ + w_r·(-red) + w_s·syn
   -------------------------------------------------------------------
   - Instead of heuristically combining, LEARN the optimal weights
   - Target: LP (Fisher importance) or validation loss after pruning
   - This finds the combination that best predicts true importance.
"""
