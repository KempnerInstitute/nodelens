"""
Mechanism diagnostic plots for SCAR-style LLM pruning experiments.

These are intentionally lightweight and deterministic, meant to produce:
- Loss-proxy concentration plots (supernode heavy-tail)
- Halo structure plots (Conn vs redundancy/protection)
- Summary plots for the mechanism evidence section
- A simple schematic diagram of the SCAR pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _save(fig: plt.Figure, save_path: Union[str, Path], dpi: int = 300) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    logger.info(f"[Saved] {save_path}")


def plot_loss_proxy_concentration(
    loss_proxy: Any,
    rho: float = 0.01,
    layer_label: str = "",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot (ICML figure* friendly):
      (a) sorted LP values (heavy tail)
      (b) cumulative proxy mass vs fraction of channels kept
    """
    lp = _to_numpy(loss_proxy).astype(np.float64).reshape(-1)
    lp = lp[np.isfinite(lp)]
    lp = np.maximum(lp, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    if lp.size == 0:
        for ax in axes:
            ax.axis("off")
        return fig

    rho = float(rho)
    rho = min(max(rho, 1e-6), 0.5)

    lp_sorted = np.sort(lp)[::-1]
    n = lp_sorted.size
    k = max(1, int(round(rho * n)))

    total = float(lp_sorted.sum()) if float(lp_sorted.sum()) > 0 else 1.0
    cum_mass = np.cumsum(lp_sorted) / total
    frac = (np.arange(n) + 1) / float(n)
    top_mass = float(cum_mass[k - 1])

    # Panel A: sorted values
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(frac, lp_sorted, color="#2c3e50", linewidth=1.5)
    ax.axvline(x=rho, color="#c0392b", linestyle="--", linewidth=2, label=f"Top {rho*100:.1f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of channels (sorted by LP)")
    ax.set_ylabel("Loss proxy (LP)")
    title = "Loss-proxy heavy tail"
    if layer_label:
        title += f"\n{layer_label}"
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel B: cumulative mass
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(frac, cum_mass, color="#2980b9", linewidth=2.0)
    ax.axvline(x=rho, color="#c0392b", linestyle="--", linewidth=2)
    ax.scatter([rho], [top_mass], color="#c0392b", zorder=5)
    ax.set_xlabel("Fraction of channels kept (top by LP)")
    ax.set_ylabel("Cumulative LP mass")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Top {rho*100:.1f}% mass = {top_mass*100:.1f}%", fontsize=10.5)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_halo_structure(
    conn: Any,
    redundancy_to_core: Any,
    protect: Any,
    super_mask: Any,
    halo_mask: Any,
    layer_label: str = "",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    max_points: int = 60000,
) -> plt.Figure:
    """
    Three-panel plot (ICML figure* friendly):
      (a) Conn vs redundancy-to-core (halo channels)
      (b) Redundancy-to-core distribution: halo vs non-halo (sample where defined)
      (c) Protect vs Conn (all channels; halo emphasized)
    """
    conn_np = _to_numpy(conn).astype(np.float64).reshape(-1)
    red_np = _to_numpy(redundancy_to_core).astype(np.float64).reshape(-1)
    prot_np = _to_numpy(protect).astype(np.float64).reshape(-1)
    super_np = _to_numpy(super_mask).astype(bool).reshape(-1)
    halo_np = _to_numpy(halo_mask).astype(bool).reshape(-1)

    n = int(conn_np.size)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    if n == 0:
        for ax in axes:
            ax.axis("off")
        return fig

    # Downsample for plotting stability
    idx_all = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx_all = rng.choice(idx_all, size=max_points, replace=False)

    idx_halo = idx_all[halo_np[idx_all] & (~super_np[idx_all])]
    idx_non = idx_all[(~halo_np[idx_all]) & (~super_np[idx_all])]
    idx_sup = idx_all[super_np[idx_all]]

    # (a) Conn vs redundancy-to-core (halo only)
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    x = conn_np[idx_halo]
    y = red_np[idx_halo]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    ax.scatter(x, y, s=8, alpha=0.35, color="#1f77b4", edgecolors="none")
    ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
    ax.set_ylabel(r"Red.\ to core $\mathrm{Red}^{\rightarrow \mathcal{M}}$")
    title = "Halo redundancy structure"
    if layer_label:
        title += f"\n{layer_label}"
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.25)
    if y.size > 0 and np.nanmin(y) > 0:
        ax.set_yscale("log")

    # (b) Halo vs non-halo redundancy-to-core distribution
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    y_h = red_np[idx_halo]
    y_n = red_np[idx_non]
    y_h = y_h[np.isfinite(y_h)]
    y_n = y_n[np.isfinite(y_n)]
    if y_h.size == 0 or y_n.size == 0:
        ax.text(
            0.5,
            0.5,
            "Red-to-core\n(non-halo sample unavailable)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9.5,
            color="#2c3e50",
        )
        ax.set_axis_off()
    else:
        bp = ax.boxplot(
            [y_h, y_n],
            vert=True,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#2c3e50", linewidth=2),
            boxprops=dict(linewidth=1.2, color="#2c3e50"),
            whiskerprops=dict(linewidth=1.2, color="#2c3e50"),
            capprops=dict(linewidth=1.2, color="#2c3e50"),
        )
        colors = ["#1f77b4", "#7f8c8d"]
        for patch, c in zip(bp.get("boxes", []), colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.set_xticklabels([f"Halo\n(n={y_h.size})", f"Non-halo\n(sample, n={y_n.size})"], fontsize=8.5)
        ax.set_ylabel(r"Red.\ to core $\mathrm{Red}^{\rightarrow \mathcal{M}}$")
        ax.set_title("Halo vs non-halo", fontsize=10.5)
        ax.grid(True, alpha=0.25)
        if np.nanmin(np.concatenate([y_h, y_n])) > 0:
            ax.set_yscale("log")

    # (c) Protect vs Conn
    ax = axes[2]
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.scatter(conn_np[idx_non], prot_np[idx_non], s=5, alpha=0.15, color="#7f8c8d", label="Non-halo", edgecolors="none")
    ax.scatter(conn_np[idx_halo], prot_np[idx_halo], s=7, alpha=0.35, color="#1f77b4", label="Halo", edgecolors="none")
    if idx_sup.size > 0:
        ax.scatter(conn_np[idx_sup], prot_np[idx_sup], s=10, alpha=0.7, color="#c0392b", label="Supernodes", edgecolors="none")
    ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
    ax.set_ylabel(r"Protection $\mathrm{Protect}$")
    ax.set_title("Protection vs Conn", fontsize=10.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_supernode_halo_summary(
    layer_indices: Sequence[int],
    top_mass_ratios: Sequence[float],
    halo_aggregate: Optional[Dict[str, Any]] = None,
    halo_per_layer: Optional[Dict[str, Any]] = None,
    rho: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot:
      (a) top-rho LP mass ratio across layers
      (b) halo/non-halo redundancy summary (from halo_analysis.per_layer if available)
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    ratios = np.asarray(list(top_mass_ratios), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, ratios, "o-", color="#2c3e50", linewidth=2, markersize=3.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Top-{rho*100:.1f}% LP mass ratio")
    ax.set_ylim(0, 1.02)
    ax.set_title("Supernode concentration", fontsize=10.5)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    groups = [
        ("Within-Halo", "halo_halo", "#1f77b4"),
        ("Within-Non-Halo", "non_halo", "#7f8c8d"),
        ("Cross", "cross", "#2ecc71"),
    ]

    # Prefer per-layer medians (more robust for heavy tails).
    if isinstance(halo_per_layer, dict) and halo_per_layer:
        data = []
        for _, key, _ in groups:
            vals: List[float] = []
            for _, rec in halo_per_layer.items():
                if not isinstance(rec, dict):
                    continue
                g = rec.get(key)
                if not isinstance(g, dict):
                    continue
                m = g.get("median")
                try:
                    mf = float(m)
                except Exception:
                    continue
                if np.isfinite(mf) and mf > 0:
                    vals.append(mf)
            data.append(np.asarray(vals, dtype=np.float64))

        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#2c3e50", linewidth=2),
            boxprops=dict(linewidth=1.2, color="#2c3e50"),
            whiskerprops=dict(linewidth=1.2, color="#2c3e50"),
            capprops=dict(linewidth=1.2, color="#2c3e50"),
        )
        for patch, (_, _, color) in zip(bp.get("boxes", []), groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(np.arange(1, len(groups) + 1))
        ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right", fontsize=8.5)
        ax.set_ylabel("Redundancy (Gaussian MI, nats)\n(per-layer median)")
        ax.set_title("Halo redundancy", fontsize=10.5)
        ax.grid(True, alpha=0.25, axis="y")
        ax.set_yscale("log")
    else:
        halo_aggregate = halo_aggregate or {}
        means = []
        cis = []
        for _, key, _ in groups:
            rec = halo_aggregate.get(key) or {}
            mu = float(rec.get("mean", 0.0))
            sd = float(rec.get("std", 0.0))
            n = float(rec.get("count", 0.0) or 0.0)
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            means.append(mu)
            cis.append(1.96 * sem)
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=cis, capsize=3, color=[g[2] for g in groups], alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right", fontsize=8.5)
        ax.set_ylabel("Redundancy (Gaussian MI, nats)\n(mean ± 95% CI)")
        ax.set_title("Halo redundancy", fontsize=10.5)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_supernode_outlier_profile(
    layer_indices: Sequence[int],
    outlier_ratios: Sequence[float],
    z_scores_activation: Sequence[float],
    z_scores_loss_proxy: Sequence[float],
    z_scores_max_activation: Sequence[float],
    rho: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot:
      (a) activation outlier ratio (supernode mean / population mean), log scale.
      (b) z-scores across layers (activation and loss-proxy), plus max-neuron z.
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    ratios = np.asarray(list(outlier_ratios), dtype=np.float64)
    z_act = np.asarray(list(z_scores_activation), dtype=np.float64)
    z_lp = np.asarray(list(z_scores_loss_proxy), dtype=np.float64)
    z_max = np.asarray(list(z_scores_max_activation), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, ratios, "o-", color="#8e44ad", linewidth=2.0, markersize=3.5)
    ax.set_yscale("log")
    ax.axhline(10.0, color="#f39c12", linestyle="--", linewidth=1.4, label="10×")
    ax.axhline(100.0, color="#c0392b", linestyle="--", linewidth=1.4, label="100×")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Activation outlier ratio")
    ax.set_title(f"Outlier ratio (top {rho*100:.0f}% by LP)", fontsize=10.5)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, z_act, "o-", color="#e67e22", linewidth=2.0, markersize=3.5, label="Activation z (supernode mean)")
    ax.plot(layers, z_lp, "o-", color="#2980b9", linewidth=2.0, markersize=3.5, label="LP z (supernode mean)")
    ax.axhline(2.0, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axhline(3.0, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Z-score (supernode mean)")
    ax.set_title("Outlier z-scores", fontsize=10.5)
    ax.grid(True, alpha=0.25, axis="y")

    ax2 = ax.twinx()
    ax2.plot(layers, z_max, "^-", color="#2c3e50", linewidth=1.6, markersize=4, label="Activation z (max neuron)")
    ax2.set_ylabel("Z-score (max neuron)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_sparsity_perplexity_curves(
    sparsities: Sequence[float],
    ppl_by_method: Dict[str, Sequence[Optional[float]]],
    baseline_ppl: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    xs = np.asarray(list(sparsities), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(3.45, 2.6))

    for label in sorted(ppl_by_method.keys()):
        ys_raw = ppl_by_method[label]
        ys = np.asarray([np.nan if v is None else float(v) for v in ys_raw], dtype=np.float64)
        finite = np.isfinite(ys)
        if not np.any(finite):
            continue
        ax.plot(xs[finite], ys[finite], "o-", linewidth=2.0, markersize=4, label=label, alpha=0.9)

    if baseline_ppl is not None:
        try:
            b = float(baseline_ppl)
            if np.isfinite(b):
                ax.axhline(b, color="#2c3e50", linestyle=":", linewidth=2.0, label=f"Unpruned ({b:.1f})")
        except Exception:
            pass

    ax.set_xlabel("Structured FFN channel sparsity", fontsize=10)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=10)
    ax.set_title("Perplexity vs sparsity", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # Use log if the dynamic range is large.
    all_vals: List[float] = []
    for vs in ppl_by_method.values():
        for v in vs:
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue
            if np.isfinite(vf) and vf > 0:
                all_vals.append(vf)
    if all_vals:
        mn = min(all_vals)
        mx = max(all_vals)
        if mx / max(mn, 1e-9) > 20:
            ax.set_yscale("log")

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_sparsity_accuracy_curves(
    sparsities: Sequence[float],
    acc_by_method: Dict[str, Sequence[Optional[float]]],
    baseline_acc: Optional[float] = None,
    *,
    ylabel: str = "Accuracy (%)",
    title: str = "Accuracy vs sparsity",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    xs = np.asarray(list(sparsities), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(3.45, 2.6))

    for label in sorted(acc_by_method.keys()):
        ys_raw = acc_by_method[label]
        ys = np.asarray([np.nan if v is None else float(v) for v in ys_raw], dtype=np.float64)
        finite = np.isfinite(ys)
        if not np.any(finite):
            continue
        ax.plot(xs[finite], ys[finite], "o-", linewidth=2.0, markersize=4, label=label, alpha=0.9)

    if baseline_acc is not None:
        try:
            b = float(baseline_acc)
            if np.isfinite(b):
                ax.axhline(b, color="#2c3e50", linestyle=":", linewidth=2.0, label=f"Unpruned ({b:.1f}%)")
        except Exception:
            pass

    ax.set_xlabel("Structured FFN channel sparsity", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_scar_schematic(
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate a schematic of SCAR (supernodes + halos) as a flowchart.
    This is model-agnostic and can be generated during artifact collection.
    """
    fig = plt.figure(figsize=(12, 3.8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#ecf0f1", ec="#2c3e50", lw: float = 1.6):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)

    def arrow(x1, y1, x2, y2, color="#2c3e50"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", linewidth=1.6, color=color, mutation_scale=12)
        ax.add_patch(a)

    x0 = 0.03
    col_w = 0.22
    gap = 0.035
    y_top = 0.58
    h_top = 0.32
    y_bot = 0.15
    h_bot = 0.30

    C_SUP = "#c0392b"
    C_STEP = "#2c3e50"
    C_CAL = "#d35400"

    # Col 1
    box(x0, y_top, col_w, h_top, "Calibration\n(tokens)", fc="#fdf2e9", ec=C_CAL)
    box(
        x0,
        y_bot,
        col_w,
        h_bot,
        r"Loss proxy\n$\mathrm{LP}_i=\frac{1}{2}\,\mathbb{E}[(u_i s_i)^2]$",
        fc="#fdf2e9",
        ec=C_CAL,
    )
    ax.text(x0 + col_w / 2, y_top + 0.07, "fwd + bwd", ha="center", va="center", fontsize=9.5, color=C_STEP)

    # Col 2
    x1 = x0 + col_w + gap
    box(x1, y_top, col_w, h_top, r"Supernodes\n(top-$\rho$ by LP)\n\bf protect", fc="#fdebd0", ec=C_SUP)
    box(x1, y_bot, col_w, h_bot, "FFN channels\n(sorted by LP)", fc="#f8f9f9", ec=C_STEP)

    # Col 3
    x2 = x1 + col_w + gap
    box(x2, y_top, col_w, h_top, r"Halo (Conn)\n(top-$\eta$)", fc="#eaf2f8", ec="#1f77b4")
    box(x2, y_bot, col_w, h_bot, r"Red-to-core\n$\max_{s\in\mathcal{M}}\mathrm{Red}(j,s)$", fc="#eaf2f8", ec="#1f77b4")

    # Col 4
    x3 = x2 + col_w + gap
    box(x3, y_top, col_w, h_top, r"Protect\n(rank-power)", fc="#f8f9f9", ec=C_STEP)
    box(x3, y_bot, col_w, h_bot, r"Prune\n(redundant followers)", fc="#f8f9f9", ec=C_STEP)

    # Arrows
    arrow(x0 + col_w, y_top + h_top / 2, x1, y_top + h_top / 2, color=C_STEP)
    arrow(x0 + col_w, y_bot + h_bot / 2, x1, y_bot + h_bot / 2, color=C_STEP)
    arrow(x1 + col_w, y_top + h_top / 2, x2, y_top + h_top / 2, color=C_STEP)
    arrow(x1 + col_w, y_bot + h_bot / 2, x2, y_bot + h_bot / 2, color=C_STEP)
    arrow(x2 + col_w, y_top + h_top / 2, x3, y_top + h_top / 2, color=C_STEP)
    arrow(x2 + col_w, y_bot + h_bot / 2, x3, y_bot + h_bot / 2, color=C_STEP)

    ax.text(0.5, 0.98, "SCAR pipeline overview", ha="center", va="top", fontsize=12, fontweight="bold", color=C_STEP)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def _spearman_np(a: Any, b: Any) -> float:
    a = _to_numpy(a).astype(np.float64).reshape(-1)
    b = _to_numpy(b).astype(np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    rho = float((ra @ rb) / denom)
    return rho if np.isfinite(rho) else 0.0


def plot_lp_vs_magnitude_controls(
    *,
    loss_proxy: Any,
    activation_power: Any,
    downproj_col_norm: Optional[Any] = None,
    upproj_row_norm: Optional[Any] = None,
    gateproj_row_norm: Optional[Any] = None,
    super_mask: Optional[Any] = None,
    layer_label: str = "",
    rho: float = 0.01,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot:
      (a) log-log scatter: activation power vs loss proxy (supernodes highlighted)
      (b) rank correlations between LP and simple magnitude controls
    """
    lp = _to_numpy(loss_proxy).astype(np.float64).reshape(-1)
    ap = _to_numpy(activation_power).astype(np.float64).reshape(-1)
    n = int(min(lp.size, ap.size))
    lp = lp[:n]
    ap = ap[:n]

    eps = 1e-12
    lp = np.maximum(lp, 0.0)
    ap = np.maximum(ap, 0.0)

    if super_mask is None:
        # Default: supernodes = top-rho by LP.
        k = max(1, int(round(float(rho) * float(n))))
        idx = np.argsort(lp)[::-1]
        super_mask_np = np.zeros(n, dtype=bool)
        super_mask_np[idx[:k]] = True
    else:
        super_mask_np = _to_numpy(super_mask).astype(bool).reshape(-1)[:n]

    x = np.log10(ap + eps)
    y = np.log10(lp + eps)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # (a) scatter
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    idx_non = np.where(~super_mask_np)[0]
    idx_sup = np.where(super_mask_np)[0]
    ax.scatter(x[idx_non], y[idx_non], s=6, alpha=0.18, color="#7f8c8d", edgecolors="none", label="Non-supernode")
    ax.scatter(x[idx_sup], y[idx_sup], s=10, alpha=0.75, color="#c0392b", edgecolors="none", label=f"Supernode (top {rho*100:.1f}%)")
    ax.set_xlabel(r"$\log_{10}\, \mathbb{E}[u_i^2]$ (activation power)")
    ax.set_ylabel(r"$\log_{10}\, \mathrm{LP}_i$")
    title = "LP vs activation magnitude"
    if layer_label:
        title += f"\n{layer_label}"
    ax.set_title(title, fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    # (b) correlation summary (Spearman on log space)
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    rows: List[Tuple[str, float]] = []
    rows.append(("ρ(LP, ActPower)", _spearman_np(y, x)))

    if downproj_col_norm is not None:
        dn = _to_numpy(downproj_col_norm).astype(np.float64).reshape(-1)[:n]
        dn = np.log10(np.maximum(dn, 0.0) + eps)
        rows.append(("ρ(LP, ||v_i||)", _spearman_np(y, dn)))
    if upproj_row_norm is not None:
        un = _to_numpy(upproj_row_norm).astype(np.float64).reshape(-1)[:n]
        un = np.log10(np.maximum(un, 0.0) + eps)
        rows.append(("ρ(LP, ||W_up[i]||)", _spearman_np(y, un)))
    if gateproj_row_norm is not None:
        gn = _to_numpy(gateproj_row_norm).astype(np.float64).reshape(-1)[:n]
        gn = np.log10(np.maximum(gn, 0.0) + eps)
        rows.append(("ρ(LP, ||W_gate[i]||)", _spearman_np(y, gn)))

    ax.axis("off")
    txt = "\n".join([f"{name}: {val:+.3f}" for name, val in rows])
    ax.text(
        0.02,
        0.90,
        txt,
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#2c3e50", alpha=0.9),
    )
    ax.set_title("Rank correlation controls", fontsize=10.5)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_bus_concentration(
    *,
    layer_indices: Sequence[int],
    d_eff_super: Sequence[float],
    d_eff_random: Optional[Sequence[float]] = None,
    curves: Optional[Dict[int, Dict[str, Any]]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot:
      (a) Cumulative write-mass curves for selected layers (supernodes vs random baseline)
      (b) Effective dimension d_eff vs depth

    `curves` (optional) is a dict: layer_idx -> { "frac": [...], "cum_super": [...], "cum_rand": [...] }.
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    deff_s = np.asarray(list(d_eff_super), dtype=np.float64)
    deff_r = None if d_eff_random is None else np.asarray(list(d_eff_random), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    if isinstance(curves, dict) and curves:
        # Plot up to 3 layers for readability
        show = list(sorted(curves.keys()))
        if len(show) > 3:
            show = [show[0], show[len(show) // 2], show[-1]]
        colors = ["#2980b9", "#8e44ad", "#16a085"]
        for c, li in zip(colors, show):
            rec = curves.get(li) or {}
            frac = np.asarray(rec.get("frac", []), dtype=np.float64)
            cs = np.asarray(rec.get("cum_super", []), dtype=np.float64)
            cr = np.asarray(rec.get("cum_rand", []), dtype=np.float64)
            if frac.size and cs.size:
                ax.plot(frac, cs, color=c, linewidth=2.0, label=f"Layer {li} (super)")
            if frac.size and cr.size:
                ax.plot(frac, cr, color=c, linewidth=1.6, linestyle="--", alpha=0.9, label=f"Layer {li} (rand)")
    else:
        ax.text(0.5, 0.5, "No curves provided", ha="center", va="center", transform=ax.transAxes, fontsize=9.5)
    ax.set_xlabel("Residual dims kept (sorted by write mass)")
    ax.set_ylabel("Cumulative write mass")
    ax.set_ylim(0, 1.02)
    ax.set_title("Bus concentration (examples)", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, deff_s, "o-", color="#2c3e50", linewidth=2.0, markersize=3.5, label="Supernodes")
    if deff_r is not None and deff_r.size == deff_s.size:
        ax.plot(layers, deff_r, "o--", color="#7f8c8d", linewidth=1.8, markersize=3.0, label="Random")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Effective dimension $d_{\mathrm{eff}}$")
    ax.set_title("Low-dimensional write support", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_read_halo_dependence_summary(
    *,
    layer_indices: Sequence[int],
    spearman_rho: Sequence[float],
    read_halo_mean_abs_delta_u: Sequence[float],
    random_mean_abs_delta_u: Sequence[float],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Two-panel summary of read-halo dependence across depth."""
    layers = np.asarray(list(layer_indices), dtype=int)
    rho = np.asarray(list(spearman_rho), dtype=np.float64)
    mh = np.asarray(list(read_halo_mean_abs_delta_u), dtype=np.float64)
    mr = np.asarray(list(random_mean_abs_delta_u), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, rho, "o-", color="#2980b9", linewidth=2.0, markersize=3.5)
    ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Layer index (target)")
    ax.set_ylabel("Spearman ρ(ReadConn, mean|Δu|)")
    ax.set_title("ReadConn predicts bus dependence", fontsize=10.5)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, mh, "o-", color="#f39c12", linewidth=2.0, markersize=3.5, label="Read-halo")
    ax.plot(layers, mr, "o--", color="#95a5a6", linewidth=1.8, markersize=3.0, label="Random")
    ax.set_xlabel("Layer index (target)")
    ax.set_ylabel(r"Mean $|\Delta u_j|$")
    ax.set_title("Dependence gap", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_conditional_halo_ablation(
    *,
    layer_indices: Sequence[int],
    delta_nll_halo: Sequence[float],
    delta_nll_matched: Sequence[float],
    delta_nll_supernodes: Optional[Sequence[float]] = None,
    delta_nll_halo_plus_supernodes: Optional[Sequence[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot for the conditional causal test:
      (a) Ablate halo subset vs matched non-halo subset (supernodes intact)
      (b) Ablate supernodes (and optionally supernodes + halo)
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    dh = np.asarray(list(delta_nll_halo), dtype=np.float64)
    dm = np.asarray(list(delta_nll_matched), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, dh, "o-", color="#1f77b4", linewidth=2.0, markersize=3.5, label="Ablate halo subset")
    ax.plot(layers, dm, "o--", color="#7f8c8d", linewidth=1.8, markersize=3.0, label="Ablate matched non-halo")
    ax.axhline(0.0, color="#2c3e50", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$\Delta$NLL (per token)")
    ax.set_title("Conditional halo redundancy", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    if delta_nll_supernodes is not None:
        ds = np.asarray(list(delta_nll_supernodes), dtype=np.float64)
        ax.plot(layers, ds, "o-", color="#c0392b", linewidth=2.0, markersize=3.5, label="Ablate supernodes")
    if delta_nll_halo_plus_supernodes is not None:
        db = np.asarray(list(delta_nll_halo_plus_supernodes), dtype=np.float64)
        ax.plot(layers, db, "o--", color="#d35400", linewidth=1.8, markersize=3.0, label="Ablate supernodes + halo")
    ax.axhline(0.0, color="#2c3e50", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$\Delta$NLL (per token)")
    ax.set_title("Supernodes as loss-critical hubs", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig

