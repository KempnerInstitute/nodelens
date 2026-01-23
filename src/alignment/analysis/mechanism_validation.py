"""
General-purpose mechanism validation utilities.

This module provides reusable analysis code for validating:
1) Synergy predictions via non-additive pair lesions
2) Halo/influence predictions via downstream receiver disruption

Paper-specific plotting scripts should live under drafts/, but the core computations
belong here so they can be reused across projects and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation with scipy fallback."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(a, b)
        return float(rho) if rho == rho else 0.0
    except Exception:
        # Pearson on ranks
        ra = a.argsort().argsort().astype(np.float64)
        rb = b.argsort().argsort().astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
        return float((ra @ rb) / denom)


def logit_margin(logits, labels):
    """T = correct_logit - max_incorrect_logit."""
    import torch

    bsz = logits.size(0)
    correct = logits[torch.arange(bsz, device=logits.device), labels]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(bsz, device=logits.device), labels] = False
    max_incorrect = logits.masked_fill(~mask, float("-inf")).max(dim=1)[0]
    return (correct - max_incorrect).detach()


def _bn_for_conv(modules: Dict[str, Any], conv_name: str):
    """Best-effort BN lookup matching common conv->bn naming conventions."""
    try:
        import torch.nn as nn
    except Exception:
        return None
    candidates = [
        conv_name.replace("conv", "bn"),
        conv_name.replace(".conv", ".bn"),
        conv_name + "_bn",
    ]
    if "downsample.0" in conv_name:
        candidates.append(conv_name.replace("downsample.0", "downsample.1"))
    for name in candidates:
        m = modules.get(name)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return m
    return None


@contextmanager
def mask_conv_output_channels(model, conv_name: str, indices: Sequence[int], *, mask_bn: bool = True):
    """
    Temporarily zero out the specified Conv2d output channels.

    If a matching BatchNorm exists and mask_bn=True, also zero BN affine params for
    those channels so the post-BN signal is exactly zero.
    """
    import torch

    modules = dict(model.named_modules())
    conv = modules.get(conv_name)
    if conv is None or not hasattr(conv, "weight"):
        raise ValueError(f"Layer not found or has no weights: {conv_name}")
    bn = _bn_for_conv(modules, conv_name) if mask_bn else None

    idx = torch.as_tensor(list(indices), dtype=torch.long, device=conv.weight.device)
    # Save only touched indices (cheap; avoids cloning full tensors each eval).
    saved = {
        "conv_w": conv.weight.data.index_select(0, idx).clone(),
        "conv_b": conv.bias.data.index_select(0, idx).clone() if getattr(conv, "bias", None) is not None else None,
        "bn_w": bn.weight.data.index_select(0, idx).clone() if bn is not None and getattr(bn, "weight", None) is not None else None,
        "bn_b": bn.bias.data.index_select(0, idx).clone() if bn is not None and getattr(bn, "bias", None) is not None else None,
    }

    conv.weight.data.index_fill_(0, idx, 0.0)
    if saved["conv_b"] is not None:
        conv.bias.data.index_fill_(0, idx, 0.0)
    if bn is not None and saved["bn_w"] is not None and saved["bn_b"] is not None:
        bn.weight.data.index_fill_(0, idx, 0.0)
        bn.bias.data.index_fill_(0, idx, 0.0)

    try:
        yield
    finally:
        conv.weight.data.index_copy_(0, idx, saved["conv_w"])
        if saved["conv_b"] is not None:
            conv.bias.data.index_copy_(0, idx, saved["conv_b"])
        if bn is not None and saved["bn_w"] is not None and saved["bn_b"] is not None:
            bn.weight.data.index_copy_(0, idx, saved["bn_w"])
            bn.bias.data.index_copy_(0, idx, saved["bn_b"])


def eval_loss_acc(model, loader, *, device: str) -> Tuple[float, float]:
    """Evaluate mean CE loss and accuracy on loader."""
    import torch
    import torch.nn as nn

    model.eval()
    crit = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += float(loss.item()) * int(x.size(0))
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.size(0))
    return loss_sum / max(1, total), correct / max(1, total)


class _CovAccumulator:
    """Streaming covariance accumulator for (T, Y) with Y in R^C."""

    def __init__(self, n_channels: int):
        self.n = 0
        self.sum_y = np.zeros(n_channels, dtype=np.float64)
        self.sum_yy = np.zeros((n_channels, n_channels), dtype=np.float64)
        self.sum_t = 0.0
        self.sum_tt = 0.0
        self.sum_ty = np.zeros(n_channels, dtype=np.float64)

    def update(self, y: np.ndarray, t: np.ndarray) -> None:
        # y: [N, C], t: [N]
        if y.size == 0:
            return
        y = y.astype(np.float64, copy=False)
        t = t.astype(np.float64, copy=False)
        n = int(y.shape[0])
        self.n += n
        self.sum_y += y.sum(axis=0)
        self.sum_yy += y.T @ y
        self.sum_t += float(t.sum())
        self.sum_tt += float((t * t).sum())
        self.sum_ty += (t[:, None] * y).sum(axis=0)

    def finalize(self) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return (var_t, var_y[C], cov_yy[C,C], cov_ty[C])."""
        if self.n < 2:
            c = self.sum_y.shape[0]
            return 0.0, np.zeros(c), np.zeros((c, c)), np.zeros(c)

        n = float(self.n)
        mean_y = self.sum_y / n
        mean_t = self.sum_t / n

        cov_yy = (self.sum_yy - n * np.outer(mean_y, mean_y)) / (n - 1.0)
        var_y = np.clip(np.diag(cov_yy), 1e-12, None)

        var_t = float((self.sum_tt - n * mean_t * mean_t) / (n - 1.0))
        var_t = max(var_t, 1e-12)

        cov_ty = (self.sum_ty - n * mean_t * mean_y) / (n - 1.0)
        return var_t, var_y, cov_yy, cov_ty


def gaussian_mi_joint_from_stats(
    *,
    var_t: float,
    var_i: float,
    var_j: float,
    cov_t_i: float,
    cov_t_j: float,
    cov_i_j: float,
) -> float:
    """Gaussian MI I(T; [Y_i, Y_j]) from covariance statistics (no raw samples)."""
    cov = np.array(
        [
            [var_t, cov_t_i, cov_t_j],
            [cov_t_i, var_i, cov_i_j],
            [cov_t_j, cov_i_j, var_j],
        ],
        dtype=np.float64,
    )
    cov += 1e-10 * np.eye(3)
    cov_y = np.array([[var_i, cov_i_j], [cov_i_j, var_j]], dtype=np.float64)
    cov_y += 1e-10 * np.eye(2)

    det_all = float(np.linalg.det(cov))
    det_y = float(np.linalg.det(cov_y))
    if det_all <= 0.0 or det_y <= 0.0 or var_t <= 0.0:
        return 0.0
    return max(0.0, 0.5 * float(np.log(var_t * det_y / det_all)))


def compute_synergy_pairs_from_loader(
    *,
    model,
    loader,
    layer_name: str,
    device: str,
    activation_samples: str = "flatten_spatial",
    spatial_samples_per_image: int = 16,
    seed: int = 123,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pair synergy scores for all pairs (i<j) in a conv layer.

    Returns:
      - pairs: list of (i,j) with i<j
      - syn: predicted Gaussian synergy proxy per pair
      - mi_i: TaskMI per channel, I_G(T; Y_i) (Gaussian MI)
      - mi_ij: Redundancy matrix between channels, I_G(Y_i; Y_j) (Gaussian MI)

    Synergy is defined as:
      S(T; Yi, Yj) = I(T; [Yi, Yj]) - max(I(T; Yi), I(T; Yj))
    using Gaussian MI from covariances of (T, Yi, Yj).
    """
    import torch

    modules = dict(model.named_modules())
    layer = modules.get(layer_name)
    if layer is None:
        raise ValueError(f"Layer not found: {layer_name}")

    rng = np.random.default_rng(int(seed))
    p = max(1, int(spatial_samples_per_image))
    mode = str(activation_samples).lower()

    batch_out: Dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, out):
        batch_out["y"] = out.detach()

    handle = layer.register_forward_hook(_hook)
    acc: Optional[_CovAccumulator] = None

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            batch_out.clear()
            logits = model(x)
            T = logit_margin(logits, y).detach().cpu().numpy()  # [B]

            out = batch_out.get("y")
            if out is None or out.ndim != 4:
                continue

            out_cpu = out.detach().cpu()  # [B,C,H,W]
            b, c, h, w = out_cpu.shape
            if acc is None:
                acc = _CovAccumulator(n_channels=int(c))

            if mode in {"gap", "global", "global_avg", "global_average"}:
                y_s = out_cpu.mean(dim=(2, 3)).numpy()  # [B,C]
                t_s = T
            else:
                hw = int(h * w)
                pp = min(p, hw)
                y_hw = out_cpu.permute(0, 2, 3, 1).reshape(b, hw, c).numpy()  # [B,HW,C]
                if pp < hw:
                    idx = rng.integers(0, hw, size=(b, pp), endpoint=False)
                    row = np.arange(b)[:, None]
                    y_s = y_hw[row, idx, :].reshape(b * pp, c)
                    t_s = np.repeat(T, pp)
                else:
                    y_s = y_hw.reshape(b * hw, c)
                    t_s = np.repeat(T, hw)

            acc.update(y_s, t_s)

    handle.remove()
    if acc is None or acc.n < 2:
        raise RuntimeError("Failed to collect activations for synergy computation")

    var_t, var_y, cov_yy, cov_ty = acc.finalize()
    n_channels = int(var_y.shape[0])

    # I(T; Yi) from corr(T, Yi)
    corr_ty = cov_ty / (np.sqrt(var_t * var_y) + 1e-12)
    corr_ty = np.clip(corr_ty, -0.999, 0.999)
    mi_i = np.maximum(0.0, -0.5 * np.log(1.0 - corr_ty**2))

    # I(Yi;Yj) redundancy matrix from correlations
    denom = np.sqrt(np.outer(var_y, var_y)) + 1e-12
    corr = cov_yy / denom
    corr = np.clip(corr, -0.999, 0.999)
    mi_ij = -0.5 * np.log(1.0 - corr**2)
    np.fill_diagonal(mi_ij, 0.0)

    pairs: List[Tuple[int, int]] = []
    syn = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            mi_joint = gaussian_mi_joint_from_stats(
                var_t=var_t,
                var_i=float(var_y[i]),
                var_j=float(var_y[j]),
                cov_t_i=float(cov_ty[i]),
                cov_t_j=float(cov_ty[j]),
                cov_i_j=float(cov_yy[i, j]),
            )
            pairs.append((i, j))
            syn.append(float(mi_joint - max(float(mi_i[i]), float(mi_i[j]))))

    return pairs, np.asarray(syn, dtype=np.float64), mi_i.astype(np.float64), mi_ij.astype(np.float64)


@dataclass
class SynergyPairLesionResult:
    layer_name: str
    baseline_loss: float
    baseline_acc: float
    top_pairs: List[Tuple[int, int]]
    top_synergy: np.ndarray
    matched_control_pairs: List[Tuple[int, int]]
    matched_control_synergy: np.ndarray
    excess_damage_top: np.ndarray
    excess_damage_control: np.ndarray
    spearman_rho: float


def validate_synergy_pair_lesions(
    *,
    model,
    calib_loader,
    eval_loader,
    layer_name: str,
    device: str,
    top_pairs: int = 20,
    pool_size: int = 120,
    activation_samples: str = "flatten_spatial",
    spatial_samples_per_image: int = 16,
    seed: int = 0,
    mask_bn: bool = True,
) -> SynergyPairLesionResult:
    """
    Validate synergy with pair lesions.

    - Compute predicted synergy for all pairs from calibration activations.
    - Select top-N pairs by predicted synergy.
    - Build matched control pairs with similar (single-channel damage, task MI, redundancy).
    - Evaluate excess damage Δ_ij - max(Δ_i, Δ_j) on eval set (no fine-tuning).
    """
    rng = np.random.default_rng(int(seed))

    # 1) Predicted synergies
    pairs, syn, mi_i, mi_ij = compute_synergy_pairs_from_loader(
        model=model,
        loader=calib_loader,
        layer_name=layer_name,
        device=device,
        activation_samples=activation_samples,
        spatial_samples_per_image=spatial_samples_per_image,
        seed=seed + 123,
    )
    if syn.size == 0:
        raise RuntimeError("Synergy computation produced no pairs")

    top_n = max(1, min(int(top_pairs), len(pairs)))
    top_idx = np.argsort(-syn)[:top_n]
    top_pairs_list = [pairs[int(k)] for k in top_idx.tolist()]
    top_synergy = syn[top_idx]
    # Map all computed pairs to synergy so we can look up control-pair scores without re-running stats.
    syn_all = {pairs[i]: float(syn[i]) for i in range(len(pairs))}

    # 2) Channel pool for matching controls
    pool_size = int(max(2 * top_n, pool_size))
    pool_size = min(pool_size, int(max(i for ij in pairs for i in ij)) + 1)
    pool: List[int] = sorted({i for ij in top_pairs_list for i in ij})
    if len(pool) < pool_size:
        remaining = [i for i in range(pool_size) if i not in set(pool)]
        if remaining:
            extra = rng.choice(len(remaining), size=(pool_size - len(pool)), replace=False)
            pool.extend([remaining[int(k)] for k in extra.tolist()])
            pool = sorted(pool)

    # 3) Baseline eval
    base_loss, base_acc = eval_loss_acc(model, eval_loader, device=device)

    # 4) Single-channel damages over pool (loss increase)
    delta: Dict[int, float] = {}
    for i in pool:
        with mask_conv_output_channels(model, layer_name, [int(i)], mask_bn=mask_bn):
            loss_i, _acc_i = eval_loss_acc(model, eval_loader, device=device)
        delta[int(i)] = float(loss_i - base_loss)

    def max_delta(pair: Tuple[int, int]) -> float:
        i, j = pair
        return max(delta.get(int(i), 0.0), delta.get(int(j), 0.0))

    def max_task_mi(pair: Tuple[int, int]) -> float:
        i, j = int(pair[0]), int(pair[1])
        if mi_i is None or mi_i.size == 0:
            return 0.0
        if i >= mi_i.size or j >= mi_i.size:
            return 0.0
        return float(max(float(mi_i[i]), float(mi_i[j])))

    def redundancy_mi(pair: Tuple[int, int]) -> float:
        i, j = int(pair[0]), int(pair[1])
        if mi_ij is None or mi_ij.size == 0:
            return 0.0
        if i >= mi_ij.shape[0] or j >= mi_ij.shape[1]:
            return 0.0
        return float(mi_ij[i, j])

    # 5) Candidate control pairs sampled from pool
    top_set = set(top_pairs_list)
    cand_pairs: List[Tuple[int, int]] = []
    for _ in range(20000):
        i, j = rng.choice(pool, size=2, replace=False).tolist()
        a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        if (a, b) in top_set:
            continue
        cand_pairs.append((a, b))
        if len(cand_pairs) >= 50 * top_n:
            break
    if not cand_pairs:
        raise RuntimeError("Failed to sample control pairs")

    # 6) Greedy matching with multiple controls:
    #   - max single-channel damage (on eval set)
    #   - max task MI (on calibration set)
    #   - within-layer redundancy I(Yi;Yj) (on calibration set)
    #
    # We match in a robustly-scaled feature space so one dimension doesn't dominate
    # solely due to units.
    used: set[Tuple[int, int]] = set()
    matched_controls: List[Tuple[int, int]] = []

    # Precompute candidate features for speed
    cand_feat = {}
    for cp in cand_pairs:
        cand_feat[cp] = np.asarray(
            [max_delta(cp), max_task_mi(cp), redundancy_mi(cp)],
            dtype=np.float64,
        )
    feat_mat = np.stack(list(cand_feat.values()), axis=0) if cand_feat else np.zeros((0, 3), dtype=np.float64)
    if feat_mat.shape[0] < 10:
        raise RuntimeError("Not enough control candidates to match; increase pool_size/eval size.")
    q25 = np.percentile(feat_mat, 25, axis=0)
    q75 = np.percentile(feat_mat, 75, axis=0)
    scale = (q75 - q25)
    scale = np.where(scale > 1e-12, scale, (np.std(feat_mat, axis=0) + 1e-12))

    for sp in top_pairs_list:
        target = np.asarray([max_delta(sp), max_task_mi(sp), redundancy_mi(sp)], dtype=np.float64)
        best = None
        best_gap = None
        for cp in cand_pairs:
            if cp in used:
                continue
            v = cand_feat.get(cp)
            if v is None:
                continue
            gap = float(np.abs((v - target) / scale).sum())
            if best is None or (best_gap is not None and gap < best_gap):
                best = cp
                best_gap = gap
        if best is None:
            break
        used.add(best)
        matched_controls.append(best)
    if len(matched_controls) < max(5, top_n // 2):
        raise RuntimeError("Not enough matched control pairs; increase pool_size/eval size.")

    # 7) Pair damages and excess damage
    def pair_damage(pair: Tuple[int, int]) -> float:
        i, j = pair
        with mask_conv_output_channels(model, layer_name, [int(i), int(j)], mask_bn=mask_bn):
            loss_ij, _acc_ij = eval_loss_acc(model, eval_loader, device=device)
        return float(loss_ij - base_loss)

    top_used = top_pairs_list[: len(matched_controls)]
    excess_top = []
    for (i, j) in top_used:
        dij = pair_damage((i, j))
        excess_top.append(float(dij - max(delta[int(i)], delta[int(j)])))
    excess_ctl = []
    for (i, j) in matched_controls:
        dij = pair_damage((i, j))
        excess_ctl.append(float(dij - max(delta[int(i)], delta[int(j)])))

    excess_top_arr = np.asarray(excess_top, dtype=np.float64)
    excess_ctl_arr = np.asarray(excess_ctl, dtype=np.float64)

    # Correlation on evaluated top pairs
    syn_x = np.asarray([float(syn_all.get(p, 0.0)) for p in top_used], dtype=np.float64)
    syn_ctl = np.asarray([float(syn_all.get(p, 0.0)) for p in matched_controls], dtype=np.float64)
    rho = spearman(syn_x, excess_top_arr)

    return SynergyPairLesionResult(
        layer_name=layer_name,
        baseline_loss=float(base_loss),
        baseline_acc=float(base_acc),
        top_pairs=top_used,
        top_synergy=syn_x,
        matched_control_pairs=matched_controls,
        matched_control_synergy=syn_ctl,
        excess_damage_top=excess_top_arr,
        excess_damage_control=excess_ctl_arr,
        spearman_rho=float(rho),
    )


@dataclass
class HaloReceiverDisruptionResult:
    src_layer: str
    tgt_layer: str
    source_channels: List[int]
    per_source_spearman: List[float]
    per_source_recall_at_k: List[float]
    representative_source: int
    representative_r: np.ndarray
    representative_disruption: np.ndarray
    representative_spearman: float
    k: int


def receiver_mean_abs(
    *,
    model,
    loader,
    device: str,
    layer_name: str,
) -> np.ndarray:
    """Mean |activation| per channel of a conv layer output, aggregated over the loader."""
    import torch

    modules = dict(model.named_modules())
    layer = modules.get(layer_name)
    if layer is None:
        raise ValueError(f"Layer not found: {layer_name}")

    sums = None
    n = 0
    batch_out: Dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, out):
        batch_out["y"] = out.detach()

    h = layer.register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                batch_out.clear()
                _ = model(x)
                out = batch_out.get("y")
                if out is None or out.ndim != 4:
                    continue
                v = out.abs().mean(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)
                if sums is None:
                    sums = np.zeros_like(v)
                sums += v
                n += 1
    finally:
        h.remove()
    if sums is None or n == 0:
        raise RuntimeError("No receiver activations captured")
    return sums / float(n)


def source_sigma_from_loader(
    *,
    model,
    loader,
    device: str,
    layer_name: str,
) -> np.ndarray:
    """Compute per-channel std of GAP-pooled conv outputs over the loader."""
    import torch

    modules = dict(model.named_modules())
    layer = modules.get(layer_name)
    if layer is None:
        raise ValueError(f"Layer not found: {layer_name}")

    sum_y = None
    sum_y2 = None
    n = 0
    batch_out: Dict[str, torch.Tensor] = {}

    def _hook(_m, _inp, out):
        batch_out["y"] = out.detach()

    h = layer.register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device)
                batch_out.clear()
                _ = model(x)
                out = batch_out.get("y")
                if out is None or out.ndim != 4:
                    continue
                v = out.mean(dim=(2, 3))  # [B,C]
                v = v.detach().cpu().numpy().astype(np.float64)
                if sum_y is None:
                    sum_y = np.zeros(v.shape[1], dtype=np.float64)
                    sum_y2 = np.zeros(v.shape[1], dtype=np.float64)
                sum_y += v.sum(axis=0)
                sum_y2 += (v * v).sum(axis=0)
                n += int(v.shape[0])
    finally:
        h.remove()

    if sum_y is None or sum_y2 is None or n < 2:
        raise RuntimeError("Failed to compute sigma (no activations captured)")

    mean = sum_y / float(n)
    mean2 = sum_y2 / float(n)
    var = np.clip(mean2 - mean * mean, 1e-12, None)
    return np.sqrt(var)


def influence_vector_r_j_i(
    *,
    tgt_layer,
    sigma_src: np.ndarray,
    src_idx: int,
) -> np.ndarray:
    """Compute r_{j<-i} across receivers j for a fixed source channel i."""
    w = tgt_layer.weight.detach().cpu().numpy().astype(np.float64)
    infl = np.abs(w).sum(axis=(2, 3)) if w.ndim == 4 else np.abs(w)  # [C_out,C_in]
    n_in = min(infl.shape[1], sigma_src.shape[0])
    infl[:, :n_in] = infl[:, :n_in] * sigma_src[:n_in][None, :]
    denom = infl.sum(axis=1) + 1e-12
    i = int(min(max(src_idx, 0), infl.shape[1] - 1))
    return infl[:, i] / denom


def validate_halo_receiver_disruption(
    *,
    model,
    loader,
    src_layer_name: str,
    tgt_layer_name: str,
    source_channels: Sequence[int],
    device: str,
    sigma_src: Optional[np.ndarray] = None,
    top_frac: float = 0.1,
    mask_bn: bool = True,
) -> HaloReceiverDisruptionResult:
    """
    Validate whether r_{j<-i} predicts receiver disruption.
    """
    modules = dict(model.named_modules())
    tgt_layer = modules.get(tgt_layer_name)
    if tgt_layer is None or not hasattr(tgt_layer, "weight"):
        raise ValueError(f"Target layer not found or has no weights: {tgt_layer_name}")

    if sigma_src is None:
        sigma_src = source_sigma_from_loader(model=model, loader=loader, device=device, layer_name=src_layer_name)

    base_recv = receiver_mean_abs(model=model, loader=loader, device=device, layer_name=tgt_layer_name)
    k = max(5, int(float(top_frac) * base_recv.shape[0]))

    corrs: List[float] = []
    recalls: List[float] = []

    src_list = [int(i) for i in source_channels]
    for i in src_list:
        r = influence_vector_r_j_i(tgt_layer=tgt_layer, sigma_src=sigma_src, src_idx=i)
        with mask_conv_output_channels(model, src_layer_name, [i], mask_bn=mask_bn):
            recv = receiver_mean_abs(model=model, loader=loader, device=device, layer_name=tgt_layer_name)
        disruption = (base_recv - recv) / (base_recv + 1e-12)
        rho = spearman(r, disruption)
        corrs.append(float(rho))

        top_pred = set(np.argsort(-r)[:k].tolist())
        top_obs = set(np.argsort(-disruption)[:k].tolist())
        recalls.append(len(top_pred & top_obs) / float(k))

    if not corrs:
        raise RuntimeError("No source channels evaluated")

    med_i = int(np.argsort(np.asarray(corrs))[len(corrs) // 2])
    rep_src = src_list[med_i]
    r_rep = influence_vector_r_j_i(tgt_layer=tgt_layer, sigma_src=sigma_src, src_idx=rep_src)
    with mask_conv_output_channels(model, src_layer_name, [rep_src], mask_bn=mask_bn):
        recv_rep = receiver_mean_abs(model=model, loader=loader, device=device, layer_name=tgt_layer_name)
    dis_rep = (base_recv - recv_rep) / (base_recv + 1e-12)
    rho_rep = spearman(r_rep, dis_rep)

    return HaloReceiverDisruptionResult(
        src_layer=src_layer_name,
        tgt_layer=tgt_layer_name,
        source_channels=src_list,
        per_source_spearman=corrs,
        per_source_recall_at_k=recalls,
        representative_source=int(rep_src),
        representative_r=r_rep,
        representative_disruption=dis_rep,
        representative_spearman=float(rho_rep),
        k=int(k),
    )

