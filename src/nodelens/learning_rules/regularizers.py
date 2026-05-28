"""Differentiable regularizers for replaceability-aware training."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .statistics import channel_correlation, flatten_channel_samples


def _offdiag(matrix: torch.Tensor) -> torch.Tensor:
    return matrix - torch.diag_embed(torch.diagonal(matrix))


def task_aware_redundancy_loss(
    activations: torch.Tensor,
    task_gate: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
    normalize: bool = True,
    correlation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the BP-TARD correlation penalty for one layer.

    ``task_gate`` should be a nonnegative vector with one value per channel. If
    omitted, this reduces to a DeCov-style off-diagonal correlation penalty.
    """
    corr = channel_correlation(activations, eps=eps) if correlation is None else correlation
    n_channels = corr.shape[0]

    if task_gate is None:
        weighted_corr = corr
    else:
        if task_gate.ndim != 1 or task_gate.numel() != n_channels:
            raise ValueError("task_gate must be a 1D tensor with one value per channel")
        gate = task_gate.to(device=corr.device, dtype=corr.dtype).clamp_min(0.0)
        sqrt_gate = torch.sqrt(gate + eps)
        weighted_corr = sqrt_gate[:, None] * corr * sqrt_gate[None, :]

    loss = _offdiag(weighted_corr).square().sum()
    if normalize and n_channels > 1:
        loss = loss / (n_channels * (n_channels - 1))
    return loss


def peer_reconstructability_penalty(
    task_gate: torch.Tensor,
    peer_reconstructability: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Penalize task-gated peer reconstructability, the BP-RTP objective."""
    if task_gate.shape != peer_reconstructability.shape:
        raise ValueError("task_gate and peer_reconstructability must have the same shape")
    penalty = task_gate.to(peer_reconstructability).clamp_min(0.0) * peer_reconstructability
    return penalty.mean() if normalize else penalty.sum()


def variance_floor_loss(
    activations: torch.Tensor,
    *,
    min_std: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize channels whose minibatch standard deviation falls below a floor."""
    samples = flatten_channel_samples(activations).float()
    centered = samples - samples.mean(dim=0, keepdim=True)
    denom = max(samples.shape[0] - 1, 1)
    std = torch.sqrt(centered.square().sum(dim=0) / denom + eps)
    return F.relu(min_std - std).square().mean()


def synergy_pair_penalty(
    pooled: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_pairs: Optional[int] = 256,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward pairwise synergy: information about T from (Y_i, Y_j) beyond singletons.

    For Gaussian variables the squared-correlation analogue of mutual
    information gives, for a pair ``(i, j)`` with target ``T``,

        R^2(T | Y_i, Y_j) = (r_iT^2 + r_jT^2 - 2 r_iT r_jT r_ij) / (1 - r_ij^2)

    where ``r_iT = corr(Y_i, T)`` and ``r_ij = corr(Y_i, Y_j)``. The pairwise
    synergy excess is

        Syn_ij = R^2(T | Y_i, Y_j) - max(r_iT^2, r_jT^2)

    Returns ``-mean(Syn_ij over sampled pairs)`` so that gradient descent on
    this loss *increases* group-coded task information beyond what singletons
    provide. The target ``T`` is detached: this regularizer only flows
    gradient through the channel activations.
    """
    if pooled.ndim != 2:
        raise ValueError("pooled must be [batch, channels]")
    batch, n_channels = pooled.shape
    if n_channels < 2 or batch < 2:
        return pooled.sum() * 0.0

    p = pooled - pooled.mean(dim=0, keepdim=True)
    p_norm = p.norm(dim=0, keepdim=True).clamp_min(eps)
    p_unit = p / p_norm  # each channel column has unit L2 norm

    t = target.detach().float() - target.detach().float().mean()
    t_unit = t / t.norm().clamp_min(eps)

    rho_iT = (p_unit * t_unit[:, None]).sum(dim=0)  # [C]
    rho_ij = p_unit.T @ p_unit  # [C, C], unit diagonal

    triu_i, triu_j = torch.triu_indices(n_channels, n_channels, offset=1, device=pooled.device)
    n_pairs = int(triu_i.numel())
    if sample_pairs is not None and n_pairs > int(sample_pairs):
        perm = torch.randperm(n_pairs, device=pooled.device)[: int(sample_pairs)]
        triu_i = triu_i[perm]
        triu_j = triu_j[perm]

    r_ij = rho_ij[triu_i, triu_j]
    r_iT = rho_iT[triu_i]
    r_jT = rho_iT[triu_j]

    denom = (1.0 - r_ij.square()).clamp_min(1e-3)
    r2_pair = (r_iT.square() + r_jT.square() - 2.0 * r_iT * r_jT * r_ij) / denom
    r2_pair = r2_pair.clamp(min=0.0, max=1.0)
    max_single = torch.maximum(r_iT.square(), r_jT.square())
    synergy = (r2_pair - max_single).clamp_min(0.0)

    return -synergy.mean()


def anti_decoupling_penalty(
    pooled: torch.Tensor,
    target: torch.Tensor,
    *,
    target_rho: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize over-decoupling of the local and target axes within a layer.

    Computes the within-layer Spearman-style correlation between two proxy
    quantities defined per channel:

    - ``I_X(i)``: a local input-capture proxy, the log of per-channel
      activation variance ``log(1 + Var(Y_i))``.
    - ``I_T(i)``: a target-relevance proxy, ``corr(Y_i_pooled, T)^2``.

    Let ``rho_l = corr_i(I_X(i), I_T(i))`` be the across-channel correlation.
    The penalty is

        L_antidecouple = (rho_l - target_rho)^2.

    Minimizing this drives ``rho_l`` toward ``target_rho``. With
    ``target_rho > 0`` the rule prevents the two axes from fully decoupling,
    pushing back against the orthogonal regime that BP naturally produces.
    The target ``T`` is detached.
    """
    if pooled.ndim != 2:
        raise ValueError("pooled must be [batch, channels]")
    batch, n_channels = pooled.shape
    if n_channels < 2 or batch < 2:
        return pooled.sum() * 0.0

    p = pooled - pooled.mean(dim=0, keepdim=True)
    p_var = (p.square().sum(dim=0) / max(batch - 1, 1)).clamp_min(eps)
    i_x = torch.log1p(p_var)

    p_norm = p.norm(dim=0, keepdim=True).clamp_min(eps)
    p_unit = p / p_norm
    t = target.detach().float() - target.detach().float().mean()
    t_unit = t / t.norm().clamp_min(eps)
    rho_iT = (p_unit * t_unit[:, None]).sum(dim=0)
    i_t = rho_iT.square()

    ix_c = i_x - i_x.mean()
    it_c = i_t - i_t.mean()
    denom = (ix_c.norm() * it_c.norm()).clamp_min(eps)
    rho = (ix_c * it_c).sum() / denom

    return (rho - float(target_rho)).square()


def compact_hull_penalty(
    task_gate: torch.Tensor,
    hull_score: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Penalize task-relevant channels that have a high compact-hull score.

    Compact-hull score (``E_i^full / max(1, |H_i|)``) is the original two-axis
    paper's strongest within-bin lesion-damage predictor: higher score means a
    channel is reconstructable from a small peer set, i.e. more locally
    replaceable. Penalizing ``gate_i * hull_score_i`` pushes task-relevant
    channels toward larger hulls (broader local support) or lower task gate.
    """
    if task_gate.shape != hull_score.shape:
        raise ValueError("task_gate and hull_score must have the same shape")
    penalty = task_gate.to(hull_score).clamp_min(0.0) * hull_score.clamp_min(0.0)
    return penalty.mean() if normalize else penalty.sum()


def cross_layer_weights(
    layer_stats: List[dict],
    *,
    mode: str = "uniform",
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> List[float]:
    """Allocate a per-layer scalar weight from forward-only statistics.

    ``layer_stats`` is a list of dicts with at least ``duplicate_task_mass`` and
    ``non_replaceable_task_mass`` keys (one per regularized layer, ordered by
    forward depth). Returns a list of nonnegative floats normalized to sum to
    ``len(layer_stats)``, so the average per-layer weight stays at 1.0 and the
    user-facing ``lambda`` retains its existing scale across modes.

    Modes:
    - ``"uniform"``: returns 1.0 per layer (existing behavior).
    - ``"dtm_share"``: weight proportional to layer DTM (concentrates budget on
      layers carrying the most duplicate task mass).
    - ``"ntm_share"``: weight proportional to layer NTM (treats NTM as a
      sensitivity proxy; opposite intuition to dtm_share).
    - ``"depth"``: weight proportional to ``(1 + alpha * depth_index)``
      (CAP-QGW-style depth bias; downstream layers get more budget).
    - ``"dtm_depth"``: ``DTM_l * (1 + alpha * depth_index)`` (combine the
      cross-layer Taylor allocator from the original paper with DTM).
    """
    n = len(layer_stats)
    if n == 0:
        return []
    mode_l = str(mode).lower()
    if mode_l in {"", "uniform", "none", "off"}:
        return [1.0] * n

    depth_factor = [1.0 + float(alpha) * (i / max(n - 1, 1)) for i in range(n)]
    if mode_l in {"depth"}:
        raw = depth_factor
    else:
        dtm = [max(0.0, float(s.get("duplicate_task_mass", 0.0))) for s in layer_stats]
        ntm = [max(0.0, float(s.get("non_replaceable_task_mass", 0.0))) for s in layer_stats]
        if mode_l == "dtm_share":
            raw = dtm
        elif mode_l == "ntm_share":
            raw = ntm
        elif mode_l == "dtm_depth":
            raw = [d * f for d, f in zip(dtm, depth_factor)]
        else:
            raise ValueError(f"Unknown cross_layer_alloc mode: {mode}")

    total = sum(raw)
    if total <= eps:
        return [1.0] * n
    scale = float(n) / total
    return [v * scale for v in raw]
