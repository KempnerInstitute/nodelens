"""
Mechanism diagnostic plots for LLM pruning experiments.

These are general-purpose visualization utilities for:
- Loss-proxy concentration plots (heavy-tail analysis)
- Halo structure plots (connectivity vs redundancy)
- Sparsity-performance curves
- Schematic diagrams for FFN pruning pipelines
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

logger = logging.getLogger(__name__)

# Default color palette for pruning methods (can be overridden)
DEFAULT_METHOD_COLORS = {
    "method_a": "#c0392b",
    "method_b": "#e74c3c",
    "method_c": "#27ae60",
    "baseline_1": "#3498db",
    "baseline_2": "#f39c12",
    "baseline_3": "#9b59b6",
    "magnitude": "#e67e22",
    "random": "#95a5a6",
    "unpruned": "#2c3e50",
}


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
    Two-panel plot sized for a typical two-column figure:
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
    Three-panel plot sized for a typical two-column figure:
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


def plot_halo_structure_improved(
    *,
    conn_values: Any,
    redundancy_values: Any,
    is_halo: Any,
    per_layer_halo_means: Optional[Sequence[float]] = None,
    per_layer_nonhalo_means: Optional[Sequence[float]] = None,
    aggregate_halo_mean: Optional[float] = None,
    aggregate_nonhalo_mean: Optional[float] = None,
    layer_indices: Optional[Sequence[int]] = None,
    per_layer_ratios: Optional[Sequence[float]] = None,
    n_bins: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Improved halo structure visualization with 4 panels:
      (A) Binned Conn vs Redundancy (means + 95% CI) - cleaner than raw scatter
      (B) Per-layer halo/non-halo ratio bars
      (C) Aggregate comparison
      (D) Ratio distribution across layers
    
    Args:
        conn_values: Connectivity scores for halo channels
        redundancy_values: Redundancy-to-core values for halo channels
        is_halo: Boolean mask indicating halo membership
        per_layer_halo_means: Mean redundancy for halo per layer
        per_layer_nonhalo_means: Mean redundancy for non-halo per layer
        aggregate_halo_mean: Overall mean redundancy for halo
        aggregate_nonhalo_mean: Overall mean redundancy for non-halo
        layer_indices: Layer indices (for panel B)
        per_layer_ratios: Pre-computed halo/non-halo ratios per layer
        n_bins: Number of bins for connectivity in panel A
        save_path: Optional path to save figure
        dpi: Resolution
    """
    import scipy.stats as stats
    
    conn_np = _to_numpy(conn_values).astype(np.float64).ravel()
    red_np = _to_numpy(redundancy_values).astype(np.float64).ravel()
    halo_np = _to_numpy(is_halo).astype(bool).ravel()
    
    # Filter to finite values in halo region
    valid = np.isfinite(conn_np) & np.isfinite(red_np) & (red_np > 0) & halo_np
    conn_h = conn_np[valid]
    red_h = red_np[valid]
    
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6), gridspec_kw={'width_ratios': [1.2, 1, 0.8, 1]})
    
    # ========== Panel A: Binned Conn vs Redundancy ==========
    ax = axes[0]
    ax.text(0.02, 0.98, "(A)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if conn_h.size >= 20:
        # Bin by connectivity percentiles
        bin_edges = np.percentile(conn_h, np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        bin_means = []
        bin_ci_low = []
        bin_ci_high = []
        
        for i in range(n_bins):
            mask = (conn_h >= bin_edges[i]) & (conn_h < bin_edges[i+1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (conn_h >= bin_edges[i]) & (conn_h <= bin_edges[i+1])
            
            if mask.sum() >= 3:
                bin_red = red_h[mask]
                bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                bin_means.append(np.mean(bin_red))
                # 95% CI via bootstrap or t-distribution
                sem = np.std(bin_red, ddof=1) / np.sqrt(len(bin_red))
                t_crit = 1.96  # Approx for large n
                bin_ci_low.append(np.mean(bin_red) - t_crit * sem)
                bin_ci_high.append(np.mean(bin_red) + t_crit * sem)
        
        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_ci_low = np.array(bin_ci_low)
        bin_ci_high = np.array(bin_ci_high)
        
        ax.fill_between(bin_centers, bin_ci_low, bin_ci_high, alpha=0.3, color="#1f77b4")
        ax.plot(bin_centers, bin_means, 'o-', color="#1f77b4", linewidth=2, markersize=5)
        ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
        ax.set_ylabel(r"Redundancy (mean $\pm$ 95% CI)")
        ax.set_title("Conn vs Redundancy\n(binned means)", fontsize=10)
    else:
        # Fallback: raw scatter if too few points
        ax.scatter(conn_h, red_h, s=8, alpha=0.35, color="#1f77b4", edgecolors="none")
        ax.set_xlabel(r"Connectivity $\mathrm{Conn}$")
        ax.set_ylabel(r"Redundancy")
        ax.set_title("Conn vs Redundancy", fontsize=10)
    ax.grid(True, alpha=0.25)
    
    # ========== Panel B: Per-layer ratio bars ==========
    ax = axes[1]
    ax.text(0.02, 0.98, "(B)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if per_layer_halo_means is not None and per_layer_nonhalo_means is not None:
        halo_arr = np.asarray(per_layer_halo_means, dtype=np.float64)
        nonhalo_arr = np.asarray(per_layer_nonhalo_means, dtype=np.float64)
        layers = np.asarray(layer_indices if layer_indices is not None else np.arange(len(halo_arr)))
        
        # Use pre-computed ratios if available, else compute
        if per_layer_ratios is not None:
            ratios = np.asarray(per_layer_ratios, dtype=np.float64)
        else:
            ratios = halo_arr / np.maximum(nonhalo_arr, 1e-12)
        
        valid_mask = np.isfinite(ratios) & (ratios > 0)
        
        colors = ['#ff7f0e' if r > 1.0 else '#7f8c8d' for r in ratios[valid_mask]]
        ax.bar(layers[valid_mask], ratios[valid_mask], color=colors, alpha=0.8, edgecolor='none')
        ax.axhline(y=1.0, color='#c0392b', linestyle='--', linewidth=1.5, label='No enrichment')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Halo/Non-halo ratio")
        ax.set_title("Ratio by Layer", fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
    else:
        ax.text(0.5, 0.5, "No per-layer data", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.25, axis='y')
    
    # ========== Panel C: Aggregate comparison ==========
    ax = axes[2]
    ax.text(0.02, 0.98, "(C)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if aggregate_halo_mean is not None and aggregate_nonhalo_mean is not None:
        x_pos = [0, 1]
        vals = [aggregate_halo_mean, aggregate_nonhalo_mean]
        colors = ['#ff7f0e', '#7f8c8d']
        bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, width=0.6, edgecolor='none')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Halo', 'Non-halo'], fontsize=9)
        ax.set_ylabel("Mean Redundancy")
        ax.set_title("Aggregate", fontsize=10)
        
        # Annotate ratio
        if aggregate_nonhalo_mean > 0:
            ratio = aggregate_halo_mean / aggregate_nonhalo_mean
            ax.text(0.5, 0.95, f"{ratio:.2f}×", ha='center', va='top', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold', color='#2c3e50')
    else:
        ax.text(0.5, 0.5, "No aggregate data", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.25, axis='y')
    
    # ========== Panel D: Ratio distribution ==========
    ax = axes[3]
    ax.text(0.02, 0.98, "(D)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if per_layer_ratios is not None or (per_layer_halo_means is not None and per_layer_nonhalo_means is not None):
        if per_layer_ratios is not None:
            ratios = np.asarray(per_layer_ratios, dtype=np.float64)
        else:
            halo_arr = np.asarray(per_layer_halo_means, dtype=np.float64)
            nonhalo_arr = np.asarray(per_layer_nonhalo_means, dtype=np.float64)
            ratios = halo_arr / np.maximum(nonhalo_arr, 1e-12)
        
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        
        if ratios.size > 0:
            ax.hist(ratios, bins=15, color='#ff7f0e', alpha=0.7, edgecolor='white')
            ax.axvline(x=1.0, color='#2c3e50', linestyle=':', linewidth=2, label='Baseline (1×)')
            ax.axvline(x=np.mean(ratios), color='#c0392b', linestyle='-', linewidth=2, 
                      label=f'Mean: {np.mean(ratios):.2f}×')
            ax.axvline(x=np.median(ratios), color='#3498db', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(ratios):.2f}×')
            ax.set_xlabel("Halo/Non-Halo Ratio")
            ax.set_ylabel("Count (layers)")
            ax.set_title("Ratio Distribution", fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
    else:
        ax.text(0.5, 0.5, "No ratio data", ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.25, axis='y')
    
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
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

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

    ax.set_xlabel("FFN channel sparsity", fontsize=9)
    ax.set_ylabel("PPL (WikiText-2)", fontsize=9)
    # Titles are often redundant with captions; keep typography compact.
    ax.set_title("")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        mode="expand",
        ncol=2,
        fontsize=6.8,
        frameon=True,
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=2.0,
    )

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
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

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

    ax.set_xlabel("FFN channel sparsity", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    # Titles are often redundant with captions; keep this small.
    ax.set_title(title, fontsize=9, fontweight="normal")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
        mode="expand",
        ncol=2,
        fontsize=6.8,
        frameon=True,
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=2.0,
    )

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
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.0)

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
        "Loss proxy\n$\\mathrm{LP}_i=\\frac{1}{2}\\,\\mathbb{E}[(u_i s_i)^2]$",
        fc="#fdf2e9",
        ec=C_CAL,
    )
    ax.text(x0 + col_w / 2, y_top + 0.07, "fwd + bwd", ha="center", va="center", fontsize=9.5, color=C_STEP)

    # Col 2
    x1 = x0 + col_w + gap
    box(x1, y_top, col_w, h_top, "Supernodes\n(top-$\\rho$ by LP)\nprotect", fc="#fdebd0", ec=C_SUP)
    box(x1, y_bot, col_w, h_bot, "FFN channels\n(sorted by LP)", fc="#f8f9f9", ec=C_STEP)

    # Col 3
    x2 = x1 + col_w + gap
    box(x2, y_top, col_w, h_top, "Halo (Conn)\n(top-$\\eta$)", fc="#eaf2f8", ec="#1f77b4")
    box(
        x2,
        y_bot,
        col_w,
        h_bot,
        "Red-to-core\n$\\max_{s\\in\\mathcal{M}}\\mathrm{Red}(j,s)$",
        fc="#eaf2f8",
        ec="#1f77b4",
    )

    # Col 4
    x3 = x2 + col_w + gap
    box(x3, y_top, col_w, h_top, "Protect\n(rank-power)", fc="#f8f9f9", ec=C_STEP)
    box(x3, y_bot, col_w, h_bot, "Prune\n(redundant followers)", fc="#f8f9f9", ec=C_STEP)

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


def plot_main_schematic(
    *,
    ppl_wanda: Optional[float] = None,
    ppl_scar: Optional[float] = None,
    supernode_pruned_pct_wanda: Optional[float] = None,
    supernode_pruned_pct_scar: Optional[float] = None,
    sparsity_pct: int = 50,
    d_model: int = 4096,
    d_mlp: int = 14336,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Main schematic:
      (A) SwiGLU FFN block with a few highlighted channels
      (B) Supernode/halo write overlap via W_down
      (C) Headline pruning result at a target sparsity
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    # Give subplot titles a bit more breathing room (avoid overlap/cropping).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10, wspace=0.40)
    for ax in axes:
        ax.set_axis_off()

    C_SUP = "#c0392b"
    C_HALO = "#f39c12"
    C_REG = "#bdc3c7"
    C_INK = "#2c3e50"

    # -------------------------
    # (A) SwiGLU FFN block
    # -------------------------
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.98, "(A) SwiGLU FFN", ha="left", va="top", fontsize=10.0, fontweight="bold")

    ax.add_patch(Circle((0.07, 0.50), 0.06, facecolor="white", edgecolor=C_INK, linewidth=2.0))
    ax.text(0.07, 0.50, "x", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.07, 0.33, f"Input\n({d_model})", ha="center", va="top", fontsize=8, color=C_INK)

    ax.add_patch(Circle((0.93, 0.50), 0.06, facecolor="white", edgecolor=C_INK, linewidth=2.0))
    ax.text(0.93, 0.50, "y", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.93, 0.33, f"Output\n({d_model})", ha="center", va="top", fontsize=8, color=C_INK)

    def _box(x, y, w, h, label):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=1.8,
                edgecolor=C_INK,
                facecolor="white",
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, fontweight="bold")

    _box(0.22, 0.62, 0.18, 0.22, "Gate")
    _box(0.22, 0.16, 0.18, 0.22, "Up")
    _box(0.62, 0.39, 0.18, 0.22, "Down")

    ax.add_patch(Circle((0.48, 0.50), 0.035, facecolor="white", edgecolor=C_INK, linewidth=1.6))
    ax.text(0.48, 0.50, "⊙", ha="center", va="center", fontsize=12)

    def _arrow(p1, p2, ls="-", lw=1.6, color=C_INK):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", linewidth=lw, linestyle=ls, color=color, mutation_scale=10))

    _arrow((0.13, 0.50), (0.22, 0.73))
    _arrow((0.13, 0.50), (0.22, 0.27))
    _arrow((0.40, 0.73), (0.45, 0.53))
    _arrow((0.40, 0.27), (0.45, 0.47))
    _arrow((0.515, 0.50), (0.62, 0.50))
    _arrow((0.80, 0.50), (0.87, 0.50))

    # Stylized intermediate channels u
    xs = np.linspace(0.40, 0.56, 14)
    for i, xi in enumerate(xs):
        color = C_REG
        lw = 2.0
        if i in (3, 10):
            color = C_SUP
            lw = 3.0
        elif i in (2, 4, 9, 11):
            color = C_HALO
            lw = 2.6
        ax.plot([xi, xi], [0.26, 0.74], color=color, linewidth=lw, solid_capstyle="round", alpha=0.95)
    ax.text(0.48, 0.18, f"$u\\in\\mathbb{{R}}^{{{d_mlp}}}$", ha="center", va="center", fontsize=8.5, color="#7f8c8d")

    # -------------------------
    # (B) Supernode bus structure: write halo + read halo
    # -------------------------
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.98, "(B) Bus structure", ha="left", va="top", fontsize=10.0, fontweight="bold")

    C_READ = "#3498db"  # Blue for read halo

    # Layer L: supernodes and write halo (left side)
    left_y = [0.80, 0.65, 0.50, 0.35]
    left_c = [C_SUP, C_HALO, C_SUP, C_HALO]
    left_labels = ["S", "W", "S", "W"]
    
    # Shared support / residual stream (center)
    center_y = [0.72, 0.52, 0.32]
    
    # Layer L+1: read halo (right side)
    right_y = [0.75, 0.55, 0.35]
    right_c = [C_READ, C_REG, C_READ]
    right_labels = ["R", "", "R"]

    # Draw Layer L channels
    for y, c, lbl in zip(left_y, left_c, left_labels):
        ax.add_patch(Circle((0.12, y), 0.032, facecolor=c, edgecolor="white", linewidth=1.0))
        if lbl:
            ax.text(0.12, y, lbl, ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")

    # Draw shared write support (residual stream)
    for y in center_y:
        ax.add_patch(Circle((0.50, y), 0.025, facecolor="#ecf0f1", edgecolor="#95a5a6", linewidth=1.0))
    for y, c, lbl in zip(right_y, right_c, right_labels):
        ax.add_patch(Circle((0.88, y), 0.032, facecolor=c, edgecolor="white" if c != C_REG else "#95a5a6", linewidth=1.0))
        if lbl:
            ax.text(0.88, y, lbl, ha="center", va="center", fontsize=6.5, fontweight="bold", color="white")

    # Draw write connections (left to center)
    for y, c in zip(left_y, left_c):
        ls = "-" if c == C_SUP else "--"
        lw = 1.8 if c == C_SUP else 1.3
        for yy in center_y:
            ax.add_patch(FancyArrowPatch((0.16, y), (0.47, yy), arrowstyle="->", linewidth=lw, linestyle=ls, color=c, alpha=0.5, mutation_scale=8))

    # Draw read connections (center to right)
    for y, c in zip(right_y, right_c):
        if c == C_READ:
            for yy in center_y:
                ax.add_patch(FancyArrowPatch((0.53, yy), (0.84, y), arrowstyle="->", linewidth=1.3, linestyle="-", color=c, alpha=0.5, mutation_scale=8))

    # Labels
    ax.text(0.31, 0.18, r"$W_{\mathrm{down}}$", ha="center", va="center", fontsize=7.5, color=C_INK)
    ax.text(0.69, 0.18, r"$W_{\mathrm{up/gate}}$", ha="center", va="center", fontsize=7.5, color=C_INK)

    # Mini legend
    ax.add_patch(Circle((0.12, 0.12), 0.015, facecolor=C_SUP, edgecolor="none"))
    ax.text(0.15, 0.12, "Supernode", ha="left", va="center", fontsize=6.5)
    ax.add_patch(Circle((0.12, 0.05), 0.015, facecolor=C_HALO, edgecolor="none"))
    ax.text(0.15, 0.05, "Write halo", ha="left", va="center", fontsize=6.5)
    ax.add_patch(Circle((0.55, 0.12), 0.015, facecolor=C_READ, edgecolor="none"))
    ax.text(0.58, 0.12, "Read halo", ha="left", va="center", fontsize=6.5)

    # -------------------------
    # (C) Result callout
    # -------------------------
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.98, "(C) Pruning result", ha="left", va="top", fontsize=10.0, fontweight="bold")

    ax.add_patch(
        FancyBboxPatch(
            (0.10, 0.22),
            0.80,
            0.56,
            boxstyle="round,pad=0.03,rounding_size=0.03",
            linewidth=2.0,
            edgecolor="#27ae60",
            facecolor="#ecf9f1",
        )
    )

    def _fmt(x: Optional[float]) -> str:
        if x is None:
            return "--"
        try:
            v = float(x)
        except Exception:
            return "--"
        return f"{v:.1f}" if np.isfinite(v) else "--"

    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "--"
        try:
            v = float(x)
        except Exception:
            return "--"
        return f"{v:.1f}%" if np.isfinite(v) else "--"

    ax.text(0.50, 0.71, f"At {sparsity_pct}% sparsity:", ha="center", va="center", fontsize=11)
    ax.text(0.50, 0.55, f"Wanda PPL = {_fmt(ppl_wanda)}", ha="center", va="center", fontsize=11)
    ax.text(0.50, 0.40, f"SCAR  PPL = {_fmt(ppl_scar)}", ha="center", va="center", fontsize=11)
    if supernode_pruned_pct_wanda is not None or supernode_pruned_pct_scar is not None:
        def _fmt_pct_num(x: Optional[float]) -> str:
            if x is None:
                return "--"
            try:
                v = float(x)
            except Exception:
                return "--"
            return f"{v:.1f}" if np.isfinite(v) else "--"

        txt = f"SN pruned (W/S): {_fmt_pct_num(supernode_pruned_pct_wanda)} / {_fmt_pct_num(supernode_pruned_pct_scar)}"
        ax.text(
            0.50,
            0.28,
            txt,
            ha="center",
            va="center",
            fontsize=8.6,
            color=C_INK,
        )

    # Use manual layout (subplots_adjust above) for stable spacing.
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_supernode_hit_rate_vs_ppl(
    *,
    labels: Sequence[str],
    supernode_pruned_pct: Sequence[float],
    perplexity: Sequence[float],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    annotate: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Scatter diagnostic: how many supernodes a method prunes vs resulting PPL.

    Intended as a compact, reviewer-friendly figure explaining catastrophic pruning failures.
    """
    labs = list(labels)
    xs = np.asarray(list(supernode_pruned_pct), dtype=np.float64)
    ys = np.asarray(list(perplexity), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    # Filter valid points
    finite = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    labs_f = [l for l, ok in zip(labs, finite) if ok]
    xs = xs[finite]
    ys = ys[finite]

    def _style(label: str) -> Tuple[str, str, float]:
        # (color, marker, size)
        if label.startswith("SCAR"):
            return "#c0392b", "o", 60.0
        if "Wanda" in label:
            return "#e67e22", "o", 55.0
        if "SparseGPT" in label:
            return "#8e44ad", "o", 55.0
        if "Act" in label:
            return "#2980b9", "o", 55.0
        if "Magnitude" in label:
            return "#2c3e50", "o", 55.0
        return "#95a5a6", "o", 35.0

    # Plot in stable order: background (gray) first, then highlighted.
    order = np.argsort(ys)
    for i in order:
        label = labs_f[i]
        c, m, s = _style(label)
        z = 3 if c != "#95a5a6" else 2
        ax.scatter(xs[i], ys[i], s=s, marker=m, color=c, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=z)

    ax.set_yscale("log")
    ax.set_xlabel("Supernodes pruned (%)", fontsize=9)
    ax.set_ylabel("PPL (WikiText-2)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # Annotate only a small, pre-chosen subset (avoids clutter).
    annotate = list(annotate) if annotate is not None else [
        "SCAR-Prot",
        "Act-L2 (channel)",
        "Wanda (channel)",
        "SparseGPT (channel)",
        "Magnitude (channel)",
    ]
    for i, label in enumerate(labs_f):
        if label not in annotate:
            continue
        # Small offset that alternates to reduce overlap.
        dx = 1.5 if (i % 2 == 0) else -1.5
        dy = 1.15 if (i % 3 == 0) else 0.90
        ax.annotate(
            label.replace(" (channel)", ""),
            xy=(xs[i], ys[i]),
            xytext=(xs[i] + dx, ys[i] * dy),
            fontsize=7.5,
            color="#2c3e50",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#7f8c8d", alpha=0.8),
        )

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_supernode_hit_rate_dose_response(
    *,
    supernode_pruned_pct: Sequence[float],
    perplexity: Sequence[float],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    x_round: float = 1.0,
) -> plt.Figure:
    """
    Dose–response diagnostic: evaluate multiple random pruning masks conditioned on a
    target supernode hit-rate, then plot degradation as a function of hit-rate.

    This is intentionally more "causal-control" than `plot_supernode_hit_rate_vs_ppl`:
    it groups points by (rounded) hit-rate and draws mean ± std on log(PPL).
    """
    xs = np.asarray(list(supernode_pruned_pct), dtype=np.float64)
    ys = np.asarray(list(perplexity), dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
    xs = xs[finite]
    ys = ys[finite]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    if xs.size == 0:
        ax.text(0.5, 0.5, "No valid points", ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        if save_path is not None:
            _save(fig, save_path, dpi=dpi)
        return fig

    # Light scatter of raw points
    ax.scatter(xs, ys, s=18, color="#95a5a6", alpha=0.35, edgecolor="none", zorder=1)

    # Group by rounded hit-rate bins
    if x_round <= 0:
        x_round = 1.0
    x_bin = np.round(xs / x_round) * x_round
    uniq = np.unique(x_bin)

    mean_x = []
    mean_y = []
    yerr_low = []
    yerr_high = []
    for xb in sorted(uniq.tolist()):
        mask = x_bin == xb
        if not np.any(mask):
            continue
        yb = ys[mask]
        logy = np.log10(yb)
        mu = float(np.mean(logy))
        sd = float(np.std(logy)) if logy.size > 1 else 0.0
        y_mu = 10 ** mu
        y_lo = 10 ** (mu - sd)
        y_hi = 10 ** (mu + sd)
        mean_x.append(float(xb))
        mean_y.append(float(y_mu))
        yerr_low.append(float(y_mu - y_lo))
        yerr_high.append(float(y_hi - y_mu))

    ax.errorbar(
        mean_x,
        mean_y,
        yerr=[yerr_low, yerr_high],
        fmt="o-",
        color="#2c3e50",
        ecolor="#2c3e50",
        elinewidth=1.0,
        capsize=2.5,
        markersize=4.0,
        linewidth=1.2,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Supernodes pruned (%)", fontsize=9)
    ax.set_ylabel("PPL (WikiText-2)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_lp_vs_ablation_validation(
    *,
    lp: Sequence[float],
    delta_nll: Sequence[float],
    layer_label: str = "",
    rho: float = 0.01,
    spearman_by_layer: Optional[Sequence[float]] = None,
    layer_indices: Optional[Sequence[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Validate LP as an instrument: compare LP to true Δloss from single-channel ablation.

    Two-panel figure:
      (a) scatter: log LP vs log ΔNLL (representative layer)
      (b) Spearman correlations across layers (if provided)
    """
    lp_arr = np.asarray(list(lp), dtype=np.float64).reshape(-1)
    dn_arr = np.asarray(list(delta_nll), dtype=np.float64).reshape(-1)
    m = min(lp_arr.size, dn_arr.size)
    lp_arr = lp_arr[:m]
    dn_arr = dn_arr[:m]

    # Only plot points with positive values (log-scale).
    mask = np.isfinite(lp_arr) & np.isfinite(dn_arr) & (lp_arr > 0) & (dn_arr > 0)
    lp_arr = lp_arr[mask]
    dn_arr = dn_arr[mask]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    # (a) Scatter
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    if lp_arr.size == 0:
        ax.axis("off")
    else:
        x = np.log10(lp_arr)
        y = np.log10(dn_arr)
        thr = np.quantile(lp_arr, 1.0 - float(rho)) if (0.0 < float(rho) < 1.0) else np.quantile(lp_arr, 0.99)
        super_mask = lp_arr >= thr

        ax.scatter(x[~super_mask], y[~super_mask], s=10, color="#95a5a6", alpha=0.35, linewidth=0)
        ax.scatter(x[super_mask], y[super_mask], s=16, color="#c0392b", alpha=0.85, linewidth=0)

        # Spearman on log-log (rank correlation of x and y)
        rho_s = _spearman_np(x, y)
        ax.set_xlabel(r"$\log_{10}\,\mathrm{LP}_i$", fontsize=10)
        ax.set_ylabel(r"$\log_{10}\,\Delta\mathrm{NLL}_i$", fontsize=10)
        ax.set_title(f"LP vs true ablation loss {layer_label}".strip(), fontsize=10.5)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.text(
            0.04,
            0.06,
            f"Spearman ρ = {rho_s:+.2f}\nN = {int(lp_arr.size)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
        )

    # (b) Across-layer correlations
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    if spearman_by_layer is None or layer_indices is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Across-layer\ncorrelations\n(not provided)", ha="center", va="center", fontsize=10, color="#7f8c8d")
    else:
        xs = np.asarray(list(layer_indices), dtype=np.float64)
        ys = np.asarray(list(spearman_by_layer), dtype=np.float64)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[ok]
        ys = ys[ok]
        if xs.size == 0:
            ax.axis("off")
        else:
            ax.plot(xs, ys, "o-", color="#2980b9", linewidth=2.0, markersize=4, alpha=0.9)
            ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.7)
            med = float(np.median(ys)) if ys.size else float("nan")
            if np.isfinite(med):
                ax.axhline(med, color="#2c3e50", linestyle=":", linewidth=1.6, alpha=0.9, label=f"Median {med:+.2f}")
                ax.legend(loc="lower right", fontsize=8, frameon=True)
            ax.set_xlabel("Layer index", fontsize=10)
            ax.set_ylabel("Spearman ρ", fontsize=10)
            ax.set_title("LP vs ΔNLL rank correlation", fontsize=10.5)
            ax.grid(True, alpha=0.25, linewidth=0.6)

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


def plot_lp_retrieval_validation(
    *,
    lp: Sequence[float],
    delta_nll: Sequence[float],
    activation_power: Optional[Sequence[float]] = None,
    layer_label: str = "",
    k_values: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Validate LP as an instrument using retrieval metrics (Precision@k, Recall@k).

    Three-panel figure:
      (a) Precision@k curves: LP vs activation power vs random baseline
      (b) Recall@k curves: LP vs activation power vs random baseline
      (c) Summary: AUC or top-1% retrieval statistics

    This is more appropriate than correlation when the goal is to identify the tail.
    """
    lp_arr = np.asarray(list(lp), dtype=np.float64).reshape(-1)
    dn_arr = np.asarray(list(delta_nll), dtype=np.float64).reshape(-1)
    n = min(lp_arr.size, dn_arr.size)
    lp_arr = lp_arr[:n]
    dn_arr = dn_arr[:n]

    # Filter valid values
    mask = np.isfinite(lp_arr) & np.isfinite(dn_arr) & (lp_arr > 0) & (dn_arr > 0)
    lp_arr = lp_arr[mask]
    dn_arr = dn_arr[mask]

    ap_arr = None
    if activation_power is not None:
        ap_arr = np.asarray(list(activation_power), dtype=np.float64).reshape(-1)
        ap_arr = ap_arr[:n][mask]

    n = lp_arr.size
    if n < 10:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))
        ax.text(0.5, 0.5, "Insufficient data for retrieval analysis", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # Define "true positives" as top k% by true ΔNLL
    k_vals = np.asarray(list(k_values), dtype=np.float64)
    k_vals = k_vals[(k_vals > 0) & (k_vals < 1)]

    # Rankings (descending = highest first)
    lp_rank = np.argsort(-lp_arr)  # indices sorted by LP descending
    dn_rank = np.argsort(-dn_arr)  # indices sorted by ΔNLL descending

    prec_lp = []
    rec_lp = []
    prec_ap = []
    rec_ap = []
    prec_rand = []
    rec_rand = []

    for k in k_vals:
        top_k = max(1, int(round(k * n)))

        # True positives = top k% by ΔNLL
        true_pos_set = set(dn_rank[:top_k])

        # LP predictions = top k% by LP
        lp_pred_set = set(lp_rank[:top_k])
        overlap_lp = len(true_pos_set & lp_pred_set)
        prec_lp.append(overlap_lp / len(lp_pred_set) if lp_pred_set else 0)
        rec_lp.append(overlap_lp / len(true_pos_set) if true_pos_set else 0)

        # Random baseline = expected overlap
        prec_rand.append(k)  # E[precision] = k for random
        rec_rand.append(k)   # E[recall] = k for random

        # Activation power predictions
        if ap_arr is not None:
            ap_rank = np.argsort(-ap_arr)
            ap_pred_set = set(ap_rank[:top_k])
            overlap_ap = len(true_pos_set & ap_pred_set)
            prec_ap.append(overlap_ap / len(ap_pred_set) if ap_pred_set else 0)
            rec_ap.append(overlap_ap / len(true_pos_set) if true_pos_set else 0)

    # (a) Precision@k
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(k_vals * 100, prec_lp, "o-", color="#2c3e50", linewidth=2, markersize=4, label="LP")
    if ap_arr is not None:
        ax.plot(k_vals * 100, prec_ap, "s--", color="#e67e22", linewidth=1.8, markersize=4, label="ActPower")
    ax.plot(k_vals * 100, prec_rand, ":", color="#95a5a6", linewidth=1.5, label="Random")
    ax.set_xlabel("k (%)")
    ax.set_ylabel("Precision@k")
    ax.set_title("LP retrieves true supernodes", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_ylim(0, 1.02)

    # (b) Recall@k
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(k_vals * 100, rec_lp, "o-", color="#2c3e50", linewidth=2, markersize=4, label="LP")
    if ap_arr is not None:
        ax.plot(k_vals * 100, rec_ap, "s--", color="#e67e22", linewidth=1.8, markersize=4, label="ActPower")
    ax.plot(k_vals * 100, rec_rand, ":", color="#95a5a6", linewidth=1.5, label="Random")
    ax.set_xlabel("k (%)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Tail recovery rate", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_ylim(0, 1.02)

    # (c) Summary stats
    ax = axes[2]
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    # Compute AUC-style summary: average precision across k values
    avg_prec_lp = np.mean(prec_lp)
    avg_prec_ap = np.mean(prec_ap) if ap_arr is not None else 0
    avg_prec_rand = np.mean(prec_rand)

    bars = [avg_prec_lp]
    labels = ["LP"]
    colors = ["#2c3e50"]
    if ap_arr is not None:
        bars.append(avg_prec_ap)
        labels.append("ActPower")
        colors.append("#e67e22")
    bars.append(avg_prec_rand)
    labels.append("Random")
    colors.append("#95a5a6")

    x_pos = np.arange(len(bars))
    ax.bar(x_pos, bars, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Precision@k")
    ax.set_title("Summary", fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.25, axis="y")

    # Add value labels on bars
    for i, v in enumerate(bars):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


def plot_lp_vs_ablation_improved(
    *,
    lp: Sequence[float],
    delta_nll: Sequence[float],
    layer_label: str = "",
    rho: float = 0.01,
    spearman_by_layer: Optional[Sequence[float]] = None,
    layer_indices: Optional[Sequence[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Improved LP validation with 2 panels showing clear relationships.
    
      (a) LP percentile vs mean |ΔNLL| (bar chart showing trend)
      (b) Tail retrieval: hit rate for top-k% by LP vs random
    
    Key improvement: Uses bar charts and hit rates instead of noisy scatter.
    """
    lp_arr = np.asarray(list(lp), dtype=np.float64).reshape(-1)
    dn_arr = np.asarray(list(delta_nll), dtype=np.float64).reshape(-1)
    m = min(lp_arr.size, dn_arr.size)
    lp_arr = lp_arr[:m]
    dn_arr = dn_arr[:m]
    
    mask = np.isfinite(lp_arr) & np.isfinite(dn_arr) & (lp_arr > 0)
    lp_filt = lp_arr[mask]
    dn_filt = dn_arr[mask]
    n = lp_filt.size
    abs_dnll = np.abs(dn_filt)
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    
    # ========== Panel A: LP percentile vs mean |ΔNLL| ==========
    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if n < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
    else:
        lp_percentiles = np.percentile(lp_filt, [0, 25, 50, 75, 90, 95, 99, 100])
        labels = ['0-25%', '25-50%', '50-75%', '75-90%', '90-95%', '95-99%', 'Top 1%']
        means, stds = [], []
        for i in range(len(lp_percentiles) - 1):
            if i == len(lp_percentiles) - 2:
                mask_q = (lp_filt >= lp_percentiles[i]) & (lp_filt <= lp_percentiles[i + 1])
            else:
                mask_q = (lp_filt >= lp_percentiles[i]) & (lp_filt < lp_percentiles[i + 1])
            if mask_q.sum() > 0:
                means.append(np.mean(abs_dnll[mask_q]))
                stds.append(np.std(abs_dnll[mask_q]) / np.sqrt(mask_q.sum()))
            else:
                means.append(0)
                stds.append(0)
        
        x = np.arange(len(labels))
        colors = ['#95a5a6'] * 4 + ['#f39c12'] * 2 + ['#c0392b']
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(r'Mean $|\Delta\mathrm{NLL}|$', fontsize=9)
        ax.set_title('LP percentile vs ablation effect', fontsize=10)
        ax.grid(True, alpha=0.25, axis='y')
    
    # ========== Panel B: Tail hit rate ==========
    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    
    if n >= 10:
        k_values = [1, 2, 5, 10, 20]
        lp_rank = np.argsort(-lp_filt)
        dnll_rank = np.argsort(-abs_dnll)
        
        hit_rates, expected_random = [], []
        for k in k_values:
            top_k = max(1, int(round(k / 100 * n)))
            lp_top = set(lp_rank[:top_k])
            dnll_top = set(dnll_rank[:top_k])
            overlap = len(lp_top & dnll_top)
            hit_rates.append(overlap / top_k)
            expected_random.append(k / 100)
        
        xb = np.arange(len(k_values))
        width = 0.35
        ax.bar(xb - width / 2, hit_rates, width, label='LP', color='#2c3e50', alpha=0.85)
        ax.bar(xb + width / 2, expected_random, width, label='Random', color='#95a5a6', alpha=0.6)
        ax.set_xticks(xb)
        ax.set_xticklabels([f'Top {k}%' for k in k_values], fontsize=8)
        ax.set_ylabel('Hit rate', fontsize=9)
        ax.set_title(r'LP retrieves high-$|\Delta\mathrm{NLL}|$', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(0, max(0.65, max(hit_rates) * 1.15))
        ax.grid(True, alpha=0.25, axis='y')
        
        for i, (hr, er) in enumerate(zip(hit_rates, expected_random)):
            if hr > er and er > 0:
                improvement = hr / er
                ax.text(xb[i] - width / 2, hr + 0.02, f'{improvement:.1f}x', ha='center', va='bottom', 
                       fontsize=7, fontweight='bold', color='#27ae60')
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path, dpi=dpi)
    return fig


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

    has_curves = isinstance(curves, dict) and bool(curves)
    # If we don't have cumulative curves saved, fall back to a single-panel figure
    # focusing on effective dimension (the key reported quantity for this diagnostic).
    if not has_curves:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))
        ax.plot(layers, deff_s, "o-", color="#2c3e50", linewidth=2.0, markersize=3.5, label="Supernodes")
        if deff_r is not None and deff_r.size == deff_s.size:
            ax.plot(layers, deff_r, "o--", color="#7f8c8d", linewidth=1.8, markersize=3.0, label="Random")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(r"Effective dimension $d_{\mathrm{eff}}$")
        ax.set_title("High-dimensional write support", fontsize=10.5)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        plt.tight_layout()
        if save_path is not None:
            _save(fig, save_path, dpi=dpi)
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    if has_curves:
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
    ax.set_title("Write support dispersion (examples)", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, deff_s, "o-", color="#2c3e50", linewidth=2.0, markersize=3.5, label="Supernodes")
    if deff_r is not None and deff_r.size == deff_s.size:
        ax.plot(layers, deff_r, "o--", color="#7f8c8d", linewidth=1.8, markersize=3.0, label="Random")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"Effective dimension $d_{\mathrm{eff}}$")
    ax.set_title("High-dimensional write support", fontsize=10.5)
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
    decile_effect_sizes: Optional[Sequence[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Two-panel summary of read-halo dependence across depth with decile analysis."""
    layers = np.asarray(list(layer_indices), dtype=int)
    rho = np.asarray(list(spearman_rho), dtype=np.float64)
    mh = np.asarray(list(read_halo_mean_abs_delta_u), dtype=np.float64)
    mr = np.asarray(list(random_mean_abs_delta_u), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax = axes[0]
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
    ax.plot(layers, rho, "o-", color="#2980b9", linewidth=2.0, markersize=3.5)
    ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.2, alpha=0.8)
    med = np.median(rho) if rho.size > 0 else 0.0
    ax.axhline(med, color="#2c3e50", linestyle=":", linewidth=1.6, label=f"Median ρ = {med:.2f}")
    ax.set_xlabel("Layer index (target)")
    ax.set_ylabel("Spearman ρ(ReadConn, mean|Δu|)")
    ax.set_title("ReadConn correlates with support dependence", fontsize=10.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    ax = axes[1]
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    # If decile effect sizes provided, show as bar chart; otherwise show line plot
    if decile_effect_sizes is not None and len(decile_effect_sizes) > 0:
        deciles = np.asarray(list(decile_effect_sizes), dtype=np.float64)
        x = np.arange(1, len(deciles) + 1)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(deciles)))
        ax.bar(x, deciles, color=colors, edgecolor="#2c3e50", linewidth=0.5)
        ax.set_xlabel("ReadConn decile (1=lowest, 10=highest)")
        ax.set_ylabel(r"Mean $|\Delta u|$ under support ablation")
        ax.set_title("Decile effect sizes", fontsize=10.5)
        # Add ratio annotation
        if len(deciles) >= 2:
            ratio = deciles[-1] / deciles[0] if deciles[0] > 0 else float("inf")
            ax.text(0.95, 0.95, f"Top/Bottom = {ratio:.1f}×",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7"))
    else:
        ax.plot(layers, mh, "o-", color="#f39c12", linewidth=2.0, markersize=3.5, label="Top ReadConn decile")
        ax.plot(layers, mr, "o--", color="#95a5a6", linewidth=1.8, markersize=3.0, label="Bottom decile")
        ax.set_xlabel("Layer index (target)")
        ax.set_ylabel(r"Mean $|\Delta u_j|$")
        ax.set_title("Top vs bottom decile disruption", fontsize=10.5)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    ax.grid(True, alpha=0.25)

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

