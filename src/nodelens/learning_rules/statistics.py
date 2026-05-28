"""Core statistics for replaceability-aware learning rules.

These functions are intentionally small and tensor-native so they can be used
inside training loops before the full experiment pipeline grows around them.
"""

from __future__ import annotations

from typing import Tuple

import torch


def flatten_channel_samples(activations: torch.Tensor) -> torch.Tensor:
    """Return activations as ``[samples, channels]``.

    Accepted shapes:
    - ``[samples, channels]`` for already flattened activations.
    - ``[batch, channels, ...]`` for Conv/sequence activations.
    """
    if activations.ndim < 2:
        raise ValueError("activations must have at least 2 dimensions")
    if activations.ndim == 2:
        return activations
    channels = activations.shape[1]
    return activations.movedim(1, -1).reshape(-1, channels)


def channel_correlation(activations: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Compute a channel correlation matrix from activations.

    The returned matrix is differentiable with respect to ``activations`` and is
    suitable for minibatch regularizers such as BP-TARD.
    """
    samples = flatten_channel_samples(activations).float()
    n_samples = samples.shape[0]
    if n_samples == 0:
        raise ValueError("activations contain no samples")

    centered = samples - samples.mean(dim=0, keepdim=True)
    denom = max(n_samples - 1, 1)
    variance = centered.square().sum(dim=0, keepdim=True) / denom
    normalized = centered / torch.sqrt(variance + eps)
    corr = normalized.T @ normalized / denom
    return corr.clamp(min=-1.0, max=1.0)


def gaussian_mi_from_correlation(correlation: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Convert Pearson correlations to scalar Gaussian MI values."""
    rho2 = correlation.square().clamp(max=1.0 - eps)
    return -0.5 * torch.log1p(-rho2)


def peer_reconstructability_from_correlation(correlation: torch.Tensor, *, ridge: float = 1e-3) -> torch.Tensor:
    """Estimate per-channel peer reconstructability from a correlation matrix.

    For each channel ``i``, this computes the linear-Gaussian regression
    explained variance ``R^2(Y_i <- Y_-i)`` using the channel correlation matrix.
    """
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a square matrix")

    n_channels = correlation.shape[0]
    if n_channels == 1:
        return correlation.new_zeros(1)

    eye = torch.eye(n_channels - 1, device=correlation.device, dtype=correlation.dtype)
    indices = torch.arange(n_channels, device=correlation.device)
    scores = []
    for channel in range(n_channels):
        mask = indices != channel
        peer_corr = correlation[mask][:, mask]
        target_corr = correlation[mask, channel]
        beta = torch.linalg.solve(peer_corr + ridge * eye, target_corr.unsqueeze(-1)).squeeze(-1)
        r2 = target_corr @ beta
        scores.append(r2.clamp(min=0.0, max=1.0))
    return torch.stack(scores)


def average_squared_peer_correlation(correlation: torch.Tensor) -> torch.Tensor:
    """Cheap per-channel peer-reconstructability proxy from average ``corr^2``."""
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a square matrix")
    n_channels = correlation.shape[0]
    if n_channels <= 1:
        return correlation.new_zeros(n_channels)
    offdiag = correlation.square() - torch.diag_embed(torch.diagonal(correlation.square()))
    return offdiag.sum(dim=1) / float(n_channels - 1)


def capacity_masses(task_relevance: torch.Tensor, peer_reconstructability: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return duplicate task mass and non-replaceable task mass."""
    if task_relevance.shape != peer_reconstructability.shape:
        raise ValueError("task_relevance and peer_reconstructability must have the same shape")
    duplicate_task_mass = task_relevance * peer_reconstructability
    non_replaceable_task_mass = task_relevance * (1.0 - peer_reconstructability)
    return duplicate_task_mass.sum(), non_replaceable_task_mass.sum()


def _peer_r2_topk(correlation: torch.Tensor, channel: int, peer_indices: torch.Tensor, ridge: float) -> torch.Tensor:
    """Linear-Gaussian explained variance R^2(Y_channel <- Y_{peer_indices})."""
    n = peer_indices.numel()
    if n == 0:
        return correlation.new_zeros(())
    eye = torch.eye(n, device=correlation.device, dtype=correlation.dtype)
    peer_corr = correlation[peer_indices][:, peer_indices]
    target = correlation[peer_indices, channel]
    beta = torch.linalg.solve(peer_corr + ridge * eye, target.unsqueeze(-1)).squeeze(-1)
    return (target @ beta).clamp(min=0.0, max=1.0)


def compact_hull_from_correlation(
    correlation: torch.Tensor,
    *,
    max_size: int = 10,
    eps: float = 0.05,
    ridge: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedy compact-hull statistics per channel (vectorized).

    For each channel ``i`` compute the smallest peer set ``H_i`` (size capped at
    ``max_size``) such that the linear-Gaussian peer-explained variance reaches
    ``(1 - eps)`` of the full-peer R^2. Returns ``(hull_size, full_r2, hull_score)``
    where ``hull_score = full_r2 / max(1, hull_size)``. Higher ``hull_score``
    means the channel is reconstructable from a small peer set, i.e. more
    locally replaceable in the original-paper sense.

    Implementation: vectorized across channels. For each ``k in 1..max_size``,
    we solve a batched ridge system ``(P_k + ridge*I) beta = t_k`` of shape
    ``[C, k, k]`` simultaneously for every channel, where the peer sets are the
    top-``k`` absolute correlations per channel. We pick the smallest ``k`` per
    channel for which R^2 >= (1-eps) * full_r2. Cost is dominated by
    ``torch.linalg.solve`` on ``[C, k, k]`` systems, k <= max_size, which is
    much cheaper than the previous per-channel Python loop.

    All tensors are detached: this is a diagnostic / gating quantity, not part
    of the gradient graph. ``correlation`` is expected to have unit diagonal.
    """
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a square matrix")
    n_channels = correlation.shape[0]
    device = correlation.device
    dtype = correlation.dtype
    if n_channels <= 1:
        zeros = correlation.new_zeros(n_channels)
        return zeros.long(), zeros, zeros

    corr = correlation.detach()
    full_r2 = peer_reconstructability_from_correlation(corr, ridge=ridge).detach()
    cap = int(max(1, min(max_size, n_channels - 1)))
    target_eps = float(max(0.0, min(1.0, eps)))
    targets_thresh = (1.0 - target_eps) * full_r2  # [C]

    abs_corr = corr.abs().clone()
    abs_corr.fill_diagonal_(-1.0)
    topk_all = torch.topk(abs_corr, k=cap, dim=1).indices  # [C, cap]
    channel_idx = torch.arange(n_channels, device=device)

    hull_size = torch.full((n_channels,), cap, device=device, dtype=torch.long)
    resolved = torch.zeros(n_channels, device=device, dtype=torch.bool)
    # Channels with no usable target (full_r2 <= 0) get hull size 0.
    zero_target = targets_thresh <= 0.0
    hull_size[zero_target] = 0
    resolved[zero_target] = True

    for k in range(1, cap + 1):
        if bool(resolved.all()):
            break
        active = ~resolved
        if not bool(active.any()):
            break
        idx_active = channel_idx[active]
        peers = topk_all[active, :k]  # [Ca, k]
        rows = idx_active.unsqueeze(1).expand(-1, k)  # [Ca, k]
        t_vec = corr[rows, peers]  # [Ca, k]
        # Gather peer correlation submatrix per active channel: corr[peers, peers]
        row_p = peers.unsqueeze(2).expand(-1, k, k)
        col_p = peers.unsqueeze(1).expand(-1, k, k)
        peer_corr = corr[row_p, col_p]  # [Ca, k, k]
        eye = torch.eye(k, device=device, dtype=dtype).expand(idx_active.numel(), k, k)
        sys_mat = peer_corr + ridge * eye
        beta = torch.linalg.solve(sys_mat, t_vec.unsqueeze(-1)).squeeze(-1)  # [Ca, k]
        r2_k = (t_vec * beta).sum(dim=1).clamp(min=0.0, max=1.0)  # [Ca]
        thresh = targets_thresh[active]
        hit = r2_k >= thresh
        if bool(hit.any()):
            hit_global = idx_active[hit]
            hull_size[hit_global] = k
            resolved[hit_global] = True

    hull_score = full_r2 / hull_size.clamp_min(1).to(dtype)
    return hull_size, full_r2, hull_score
