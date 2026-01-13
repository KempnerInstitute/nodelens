"""
Paper-oriented plots for the SCAR LLM pruning draft.

These are intentionally lightweight and deterministic, meant to produce:
- Loss-proxy concentration plots (supernode heavy-tail)
- Halo structure plots (Conn vs redundancy/protection)
- Summary plots for the mechanism evidence section
- A simple schematic diagram of the SCAR pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

# Non-interactive backend for cluster jobs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

logger = logging.getLogger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _save(fig: plt.Figure, save_path: Union[str, Path], dpi: int = 300) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"[Saved] {save_path}")


def plot_loss_proxy_concentration(
    loss_proxy: Any,
    rho: float = 0.01,
    layer_label: str = "",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Two-panel plot:
      (Left) sorted LP values (heavy tail)
      (Right) cumulative proxy mass vs fraction of channels kept
    """
    lp = _to_numpy(loss_proxy).astype(np.float64).reshape(-1)
    lp = lp[np.isfinite(lp)]
    lp = np.maximum(lp, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    if lp.size == 0:
        for ax in axes:
            ax.axis("off")
        return fig

    rho = float(rho)
    rho = min(max(rho, 1e-6), 0.5)

    lp_sorted = np.sort(lp)[::-1]
    n = lp_sorted.size
    k = max(1, int(round(rho * n)))
    threshold = lp_sorted[k - 1]

    total = float(lp_sorted.sum()) if float(lp_sorted.sum()) > 0 else 1.0
    cum_mass = np.cumsum(lp_sorted) / total
    frac = (np.arange(n) + 1) / float(n)
    top_mass = float(cum_mass[k - 1])

    # Panel A: sorted values
    ax = axes[0]
    ax.plot(frac, lp_sorted, color="#2c3e50", linewidth=1.5)
    ax.axvline(x=rho, color="#c0392b", linestyle="--", linewidth=2, label=f"Top {rho*100:.1f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Fraction of channels (sorted by LP)")
    ax.set_ylabel("Loss proxy (LP)")
    title = "Loss-proxy heavy tail"
    if layer_label:
        title += f"\n{layer_label}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    # Panel B: cumulative mass
    ax = axes[1]
    ax.plot(frac, cum_mass, color="#2980b9", linewidth=2.0)
    ax.axvline(x=rho, color="#c0392b", linestyle="--", linewidth=2)
    ax.scatter([rho], [top_mass], color="#c0392b", zorder=5)
    ax.set_xlabel("Fraction of channels kept (top by LP)")
    ax.set_ylabel("Cumulative LP mass")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Top {rho*100:.1f}% mass = {top_mass*100:.1f}%")
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
    Two-panel plot:
      (Left) Conn vs redundancy-to-core (halo channels)
      (Right) Protect vs Conn (all channels; halo emphasized)
    """
    conn_np = _to_numpy(conn).astype(np.float64).reshape(-1)
    red_np = _to_numpy(redundancy_to_core).astype(np.float64).reshape(-1)
    prot_np = _to_numpy(protect).astype(np.float64).reshape(-1)
    super_np = _to_numpy(super_mask).astype(bool).reshape(-1)
    halo_np = _to_numpy(halo_mask).astype(bool).reshape(-1)

    n = int(conn_np.size)
    if n == 0:
        fig, _ = plt.subplots(figsize=(10, 4))
        return fig

    # Downsample for plotting stability
    idx_all = np.arange(n)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx_all = rng.choice(idx_all, size=max_points, replace=False)

    idx_halo = idx_all[halo_np[idx_all] & (~super_np[idx_all])]
    idx_non = idx_all[(~halo_np[idx_all]) & (~super_np[idx_all])]
    idx_sup = idx_all[super_np[idx_all]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel A: Conn vs redundancy-to-core (halo only, since redundancy is defined there)
    ax = axes[0]
    x = conn_np[idx_halo]
    y = red_np[idx_halo]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    ax.scatter(x, y, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
    ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
    ax.set_ylabel(r"Redundancy to core $\mathrm{Red}^{\rightarrow \mathcal{M}}$")
    title = "Halo redundancy structure"
    if layer_label:
        title += f"\n{layer_label}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if y.size > 0 and np.nanmin(y) > 0:
        ax.set_yscale("log")

    # Panel B: Protect vs Conn (all channels)
    ax = axes[1]
    ax.scatter(conn_np[idx_non], prot_np[idx_non], s=6, alpha=0.15, color="#7f8c8d", label="Non-halo", edgecolors="none")
    ax.scatter(conn_np[idx_halo], prot_np[idx_halo], s=10, alpha=0.35, color="#1f77b4", label="Halo", edgecolors="none")
    if idx_sup.size > 0:
        ax.scatter(conn_np[idx_sup], prot_np[idx_sup], s=14, alpha=0.7, color="#c0392b", label="Supernodes", edgecolors="none")
    ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
    ax.set_ylabel(r"Protection $\mathrm{Protect}$")
    ax.set_title("Protection vs connectivity")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

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
      (Left) top-rho LP mass ratio across layers
      (Right) halo/non-halo redundancy summary bars (from halo_analysis.aggregate)
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    ratios = np.asarray(list(top_mass_ratios), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    ax = axes[0]
    ax.plot(layers, ratios, "o-", color="#2c3e50", linewidth=2)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Top-{rho*100:.1f}% LP mass ratio")
    ax.set_ylim(0, 1.02)
    ax.set_title("Supernode concentration across layers")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    groups = [("Within-Halo", "halo_halo", "#1f77b4"), ("Within-Non-Halo", "non_halo", "#7f8c8d"), ("Cross", "cross", "#2ecc71")]

    # Prefer per-layer distributions (much clearer than mean±std when the MI distribution is heavy-tailed).
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
        # Color the boxes
        for patch, (_, _, color) in zip(bp.get("boxes", []), groups):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # Overlay jittered per-layer medians for transparency
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data, start=1):
            if vals.size == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=0.05, size=vals.size)
            ax.scatter(
                np.full(vals.shape, i, dtype=float) + jitter,
                vals,
                s=14,
                alpha=0.35,
                color="#2c3e50",
                edgecolors="none",
            )

        ax.set_xticks(np.arange(1, len(groups) + 1))
        ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right")
        ax.set_ylabel("Redundancy (Gaussian MI, nats)\n(per-layer median)")
        ax.set_title("Halo redundancy across layers")
        ax.grid(True, alpha=0.25, axis="y")
        # MI is positive and often heavy-tailed; log helps readability.
        ax.set_yscale("log")
    else:
        # Fallback: show mean ± 95% CI of the mean (std can be huge for heavy tails).
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
        ax.bar(x, means, yerr=cis, capsize=4, color=[g[2] for g in groups], alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right")
        ax.set_ylabel("Redundancy (Gaussian MI, nats)\n(mean ± 95% CI)")
        ax.set_title("Halo redundancy (aggregate)")
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
    Two-panel plot for supernode outlier strength across depth.

    - Left: activation outlier ratio (supernode mean / population mean), log scale.
    - Right: z-scores across layers (activation and loss-proxy); plus max-neuron activation z on a secondary axis.
    """
    layers = np.asarray(list(layer_indices), dtype=int)
    ratios = np.asarray(list(outlier_ratios), dtype=np.float64)
    z_act = np.asarray(list(z_scores_activation), dtype=np.float64)
    z_lp = np.asarray(list(z_scores_loss_proxy), dtype=np.float64)
    z_max = np.asarray(list(z_scores_max_activation), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    # Panel A: outlier ratio (log)
    ax = axes[0]
    ax.plot(layers, ratios, "o-", color="#8e44ad", linewidth=2.0, markersize=4, label="Supernode mean / population mean")
    ax.set_yscale("log")
    ax.axhline(10.0, color="#f39c12", linestyle="--", linewidth=1.8, label="10×")
    ax.axhline(100.0, color="#c0392b", linestyle="--", linewidth=1.8, label="100×")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Activation outlier ratio (log scale)")
    ax.set_title(f"Supernode outlier ratio (top {rho*100:.0f}% by LP)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(loc="upper right", frameon=True)

    # Panel B: z-scores (dual axis)
    ax = axes[1]
    ax.plot(layers, z_act, "o-", color="#e67e22", linewidth=2.0, markersize=4, label="Activation z (supernode mean)")
    ax.plot(layers, z_lp, "o-", color="#2980b9", linewidth=2.0, markersize=4, label="Loss-proxy z (supernode mean)")
    ax.axhline(2.0, color="#7f8c8d", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(3.0, color="#7f8c8d", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Z-score (supernode mean)")
    ax.set_title("Outlier z-scores across layers")
    ax.grid(True, alpha=0.25, axis="y")

    ax2 = ax.twinx()
    ax2.plot(layers, z_max, "^-", color="#2c3e50", linewidth=1.8, markersize=5, label="Activation z (max neuron)")
    ax2.set_ylabel("Z-score (max neuron, activation)")

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

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
    """
    Paper-facing plot: perplexity vs structured sparsity for multiple methods.

    Inputs are already "paper-ready" (i.e., only the intended pruning direction, typically low-mode).
    """
    xs = np.asarray(list(sparsities), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Stable ordering for legend
    for label in sorted(ppl_by_method.keys()):
        ys_raw = ppl_by_method[label]
        ys = np.asarray([np.nan if v is None else float(v) for v in ys_raw], dtype=np.float64)
        finite = np.isfinite(ys)
        if not np.any(finite):
            continue
        ax.plot(xs[finite], ys[finite], "o-", linewidth=2.0, markersize=5, label=label, alpha=0.9)

    if baseline_ppl is not None:
        try:
            b = float(baseline_ppl)
            if np.isfinite(b):
                ax.axhline(b, color="#2c3e50", linestyle=":", linewidth=2.0, label=f"Unpruned ({b:.1f})")
        except Exception:
            pass

    ax.set_xlabel("Structured FFN channel sparsity", fontsize=11)
    ax.set_ylabel("Perplexity (WikiText-2)", fontsize=11)
    ax.set_title("Perplexity vs sparsity (low-mode)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # Use log if the dynamic range is large.
    all_vals = []
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
    title: str = "Accuracy vs sparsity (low-mode)",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Paper-facing plot: downstream accuracy vs structured sparsity for multiple methods.

    Notes:
    - Accuracies are expected to already be in percent units (e.g., 58.0 for 58%).
    - Inputs should be filtered to the intended pruning direction (typically low-mode).
    """
    xs = np.asarray(list(sparsities), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for label in sorted(acc_by_method.keys()):
        ys_raw = acc_by_method[label]
        ys = np.asarray([np.nan if v is None else float(v) for v in ys_raw], dtype=np.float64)
        finite = np.isfinite(ys)
        if not np.any(finite):
            continue
        ax.plot(xs[finite], ys[finite], "o-", linewidth=2.0, markersize=5, label=label, alpha=0.9)

    if baseline_acc is not None:
        try:
            b = float(baseline_acc)
            if np.isfinite(b):
                ax.axhline(b, color="#2c3e50", linestyle=":", linewidth=2.0, label=f"Unpruned ({b:.1f}%)")
        except Exception:
            pass

    ax.set_xlabel("Structured FFN channel sparsity", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, frameon=True)

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
    # Keep this figure intentionally clean + legible in a 1-column ICML layout.
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

    # ------------------------------------------------------------------
    # Column layout
    # ------------------------------------------------------------------
    x0 = 0.03
    col_w = 0.22
    gap = 0.035
    y_top = 0.58
    h_top = 0.32
    y_bot = 0.15
    h_bot = 0.30

    # Colors (match paper narrative)
    C_SUP = "#c0392b"   # supernodes
    C_HALO = "#1f77b4"  # halo
    C_STEP = "#2c3e50"  # neutral
    C_CAL = "#d35400"   # calibration/loss-proxy compute

    # --- Col 1: Calibration + proxy ---
    box(x0, y_top, col_w, h_top, "Calibration\n(tokens)", fc="#fdf2e9", ec=C_CAL)
    box(
        x0,
        y_bot,
        col_w,
        h_bot,
        # NOTE: Use \frac{1}{2} (not \frac12) for broad compatibility with matplotlib mathtext.
        r"Loss proxy\n$\mathrm{LP}_i=\frac{1}{2}\,\mathbb{E}[(u_i s_i)^2]$",
        fc="#fdf2e9",
        ec=C_CAL,
    )

    # Tiny icon: forward/backward arrows
    ax.text(x0 + col_w / 2, y_top + 0.07, "fwd + bwd", ha="center", va="center", fontsize=9.5, color=C_STEP)

    # --- Col 2: Supernodes ---
    x1 = x0 + col_w + gap
    box(x1, y_top, col_w, h_top, r"Supernodes\n(top-$\rho$ by LP)\n\bf protect", fc="#fdebd0", ec=C_SUP)
    box(x1, y_bot, col_w, h_bot, "FFN channels\n(sorted by LP)", fc="#f8f9f9", ec=C_STEP)

    # Draw a stylized heavy-tail: a few bars, with 2 red outliers
    xs = np.linspace(x1 + 0.03, x1 + col_w - 0.03, 10)
    heights = np.array([0.06, 0.05, 0.04, 0.035, 0.03, 0.028, 0.025, 0.022, 0.18, 0.24])
    for i, (xx, hh) in enumerate(zip(xs, heights)):
        c = C_SUP if i >= 8 else "#7f8c8d"
        ax.plot([xx, xx], [y_bot + 0.06, y_bot + 0.06 + hh], color=c, linewidth=4, solid_capstyle="round")
    ax.text(x1 + col_w / 2, y_bot + 0.03, "rare outliers", ha="center", va="center", fontsize=9.0, color=C_STEP)

    # --- Col 3: Halo + redundancy ---
    x2 = x1 + col_w + gap
    box(x2, y_top, col_w, h_top, r"Halo\n(high Conn to core)", fc="#e8f6ff", ec=C_HALO)
    box(
        x2,
        y_bot,
        col_w,
        h_bot,
        r"Redundancy to core\n$\mathrm{Red}^{\rightarrow\mathcal{M}}_j=\max_{m\in\mathcal{M}} I(q_j;q_m)$",
        fc="#eafaf1",
        ec="#27ae60",
    )
    ax.text(x2 + col_w / 2, y_bot + 0.03, r"$q=u\odot s$", ha="center", va="center", fontsize=9.0, color=C_STEP)

    # --- Col 4: Structured pruning ---
    x3 = x2 + col_w + gap
    box(
        x3,
        y_top,
        col_w,
        h_top,
        "Score + prune\n(non-core only)\nlayer caps",
        fc="#f8f9f9",
        ec=C_STEP,
    )
    box(x3, y_bot, col_w, h_bot, r"Result:\nstructured FFN\nchannel sparsity", fc="#f8f9f9", ec=C_STEP)

    # Arrows across columns (top row)
    arrow(x0 + col_w, y_top + h_top / 2, x1, y_top + h_top / 2, color=C_STEP)
    arrow(x1 + col_w, y_top + h_top / 2, x2, y_top + h_top / 2, color=C_STEP)
    arrow(x2 + col_w, y_top + h_top / 2, x3, y_top + h_top / 2, color=C_STEP)

    # Vertical arrows within columns
    arrow(x0 + col_w / 2, y_top, x0 + col_w / 2, y_bot + h_bot, color=C_STEP)
    arrow(x2 + col_w / 2, y_top, x2 + col_w / 2, y_bot + h_bot, color=C_STEP)

    ax.text(
        0.02,
        0.98,
        "SCAR: supernodes + halos for structured FFN channel pruning",
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=C_STEP,
    )

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig

