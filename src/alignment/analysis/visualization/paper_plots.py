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
    halo_aggregate: Dict[str, Any],
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
    groups = [("Within-Halo", "halo_halo"), ("Within-Non-Halo", "non_halo"), ("Cross", "cross")]
    means = []
    stds = []
    for _, key in groups:
        rec = halo_aggregate.get(key) or {}
        means.append(float(rec.get("mean", 0.0)))
        stds.append(float(rec.get("std", 0.0)))

    x = np.arange(len(groups))
    ax.bar(x, means, yerr=stds, capsize=4, color=["#1f77b4", "#7f8c8d", "#2ecc71"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("Redundancy (Gaussian MI, nats)")
    ax.set_title("Halo redundancy vs non-halo (avg.)")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_scar_schematic(
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate a simple schematic of SCAR (supernodes + halos) as a flowchart.
    This is model-agnostic and can be generated during artifact collection.
    """
    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#ecf0f1", ec="#2c3e50"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.6,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2, color="#2c3e50"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", linewidth=1.6, color=color, mutation_scale=12)
        ax.add_patch(a)

    # Left: FFN depiction (conceptual)
    box(0.02, 0.62, 0.18, 0.26, "FFN layer\n(MLP channels)", fc="#f8f9f9")
    ax.text(0.11, 0.71, "u channels", ha="center", va="center", fontsize=9)
    # Draw "channels": a few vertical ticks, color some as supernodes/halo
    for i, x in enumerate(np.linspace(0.05, 0.19, 9)):
        c = "#7f8c8d"
        if i in (2, 3):
            c = "#c0392b"  # supernodes
        if i in (5, 6):
            c = "#1f77b4"  # halo
        ax.plot([x, x], [0.66, 0.84], color=c, linewidth=3)
    ax.text(0.03, 0.58, "Supernodes (red): high LP\nHalo (blue): high Conn + redundant", fontsize=9, ha="left", va="top")

    # Middle: compute steps
    box(0.28, 0.70, 0.20, 0.20, "Calibration\nforward+backward", fc="#fdf2e9", ec="#d35400")
    box(0.52, 0.70, 0.22, 0.20, r"Loss proxy\n$\mathrm{LP}_i=\frac12\mathbb{E}[(u_i s_i)^2]$", fc="#fdf2e9", ec="#d35400")
    box(0.78, 0.70, 0.20, 0.20, r"Supernodes\n(top-$\rho$ by LP)\nprotect core", fc="#fdebd0", ec="#c0392b")

    arrow(0.20, 0.80, 0.28, 0.80)
    arrow(0.48, 0.80, 0.52, 0.80)
    arrow(0.74, 0.80, 0.78, 0.80)

    # Bottom: halo + redundancy + pruning
    box(0.28, 0.35, 0.22, 0.20, r"Connectivity\n$\mathrm{Conn}_j$ from $|v_j|$ overlap", fc="#e8f6ff", ec="#2980b9")
    box(0.54, 0.35, 0.20, 0.20, r"Halo\n(top-$\eta$ non-core by Conn)", fc="#e8f6ff", ec="#2980b9")
    box(0.78, 0.35, 0.20, 0.20, r"Redundancy\n$\mathrm{Red}^{\rightarrow\mathcal{M}}$ from $q=u\!\odot\!s$", fc="#eafaf1", ec="#27ae60")
    box(0.52, 0.06, 0.46, 0.20, r"Score + prune\n(prune low-$\mathrm{LP}$ first,\nboost halo followers; respect caps)", fc="#f8f9f9", ec="#2c3e50")

    arrow(0.62, 0.70, 0.39, 0.55)
    arrow(0.50, 0.45, 0.54, 0.45)
    arrow(0.74, 0.45, 0.78, 0.45)
    arrow(0.88, 0.35, 0.75, 0.26)
    arrow(0.64, 0.35, 0.64, 0.26)

    ax.text(0.02, 0.97, "SCAR schematic (supernodes + halos for structured FFN channel pruning)", fontsize=12, fontweight="bold", ha="left", va="top")

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig

