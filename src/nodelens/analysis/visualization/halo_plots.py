"""
Halo Redundancy Visualization Module.

Creates plots for analyzing redundancy patterns within and between halo/non-halo groups
across layer depth.

Plots include:
1. Redundancy by layer depth (bar chart comparing 3 groups)
2. Histogram distributions for each group
3. Box plots comparing groups
4. Layer-wise heatmaps
5. Summary interpretation panel
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_halo_redundancy_by_depth(
    layers: List[int],
    halo_halo_means: List[float],
    non_halo_means: List[float],
    cross_group_means: List[float],
    halo_halo_stds: Optional[List[float]] = None,
    non_halo_stds: Optional[List[float]] = None,
    cross_group_stds: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Halo Redundancy by Layer Depth",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot redundancy values for each group across layer depth.

    Args:
        layers: List of layer indices
        halo_halo_means: Mean within-halo redundancy per layer
        non_halo_means: Mean within-non-halo redundancy per layer
        cross_group_means: Mean cross-group redundancy per layer
        *_stds: Standard deviations (optional, for error bars)
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(layers))
    width = 0.25

    # Left plot: Bar chart
    ax = axes[0]

    # Plot bars
    ax.bar(x - width, halo_halo_means, width, label="Within-Halo", color="#e74c3c", alpha=0.8)
    ax.bar(x, non_halo_means, width, label="Within-Non-Halo", color="#3498db", alpha=0.8)
    ax.bar(x + width, cross_group_means, width, label="Cross-Group", color="#2ecc71", alpha=0.8)

    # Add error bars if provided
    if halo_halo_stds is not None:
        ax.errorbar(x - width, halo_halo_means, yerr=halo_halo_stds, fmt="none", color="darkred", capsize=3)
    if non_halo_stds is not None:
        ax.errorbar(x, non_halo_means, yerr=non_halo_stds, fmt="none", color="darkblue", capsize=3)
    if cross_group_stds is not None:
        ax.errorbar(x + width, cross_group_means, yerr=cross_group_stds, fmt="none", color="darkgreen", capsize=3)

    ax.set_xlabel("Layer Depth", fontsize=12)
    ax.set_ylabel("Mean Redundancy (MI in nats)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{layer_idx}" for layer_idx in layers], rotation=45)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Right plot: Line plot with trends
    ax2 = axes[1]

    ax2.plot(layers, halo_halo_means, "o-", color="#e74c3c", label="Within-Halo", linewidth=2, markersize=8)
    ax2.plot(layers, non_halo_means, "s-", color="#3498db", label="Within-Non-Halo", linewidth=2, markersize=8)
    ax2.plot(layers, cross_group_means, "^-", color="#2ecc71", label="Cross-Group", linewidth=2, markersize=8)

    # Add fill between for confidence intervals
    if halo_halo_stds is not None:
        hh_arr = np.array(halo_halo_means)
        hh_std = np.array(halo_halo_stds)
        ax2.fill_between(layers, hh_arr - hh_std, hh_arr + hh_std, color="#e74c3c", alpha=0.2)
    if non_halo_stds is not None:
        nh_arr = np.array(non_halo_means)
        nh_std = np.array(non_halo_stds)
        ax2.fill_between(layers, nh_arr - nh_std, nh_arr + nh_std, color="#3498db", alpha=0.2)
    if cross_group_stds is not None:
        cg_arr = np.array(cross_group_means)
        cg_std = np.array(cross_group_stds)
        ax2.fill_between(layers, cg_arr - cg_std, cg_arr + cg_std, color="#2ecc71", alpha=0.2)

    ax2.set_xlabel("Layer Depth", fontsize=12)
    ax2.set_ylabel("Mean Redundancy (MI in nats)", fontsize=12)
    ax2.set_title("Redundancy Trends by Layer Depth", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved halo redundancy depth plot to {save_path}")

    return fig


def plot_halo_redundancy_comprehensive(
    results_list: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Create comprehensive halo redundancy visualization with 4 panels.

    Args:
        results_list: List of per-layer results with keys:
            - layer_idx
            - halo_halo: {mean, std, values}
            - non_halo_non_halo: {mean, std, values}
            - cross_group: {mean, std, values}
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Collect all values
    all_halo_halo = []
    all_non_halo = []
    all_cross = []

    layers = []
    halo_means = []
    non_halo_means = []
    cross_means = []

    for r in results_list:
        layers.append(r.get("layer_idx", len(layers)))

        hh = r.get("halo_halo", {})
        nh = r.get("non_halo_non_halo", {})
        cg = r.get("cross_group", {})

        halo_means.append(hh.get("mean", 0))
        non_halo_means.append(nh.get("mean", 0))
        cross_means.append(cg.get("mean", 0))

        if "values" in hh:
            all_halo_halo.extend(hh["values"][:2000])
        if "values" in nh:
            all_non_halo.extend(nh["values"][:2000])
        if "values" in cg:
            all_cross.extend(cg["values"][:2000])

    # Panel 1: Histogram comparison (all layers combined)
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(all_halo_halo or [1]), max(all_non_halo or [1]), max(all_cross or [1])) * 1.1, 50)

    if all_halo_halo:
        ax.hist(all_halo_halo, bins=bins, alpha=0.5, density=True, label=f"Within-Halo (μ={np.mean(all_halo_halo):.3f})", color="#e74c3c")
    if all_non_halo:
        ax.hist(all_non_halo, bins=bins, alpha=0.5, density=True, label=f"Within-Non-Halo (μ={np.mean(all_non_halo):.3f})", color="#3498db")
    if all_cross:
        ax.hist(all_cross, bins=bins, alpha=0.5, density=True, label=f"Cross-Group (μ={np.mean(all_cross):.3f})", color="#2ecc71")

    ax.set_xlabel("Redundancy (MI in nats)")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise Redundancy Distribution (All Layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Mean redundancy by layer
    ax = axes[0, 1]
    x = np.arange(len(layers))
    width = 0.25

    ax.bar(x - width, halo_means, width, label="Within-Halo", color="#e74c3c", alpha=0.8)
    ax.bar(x, non_halo_means, width, label="Within-Non-Halo", color="#3498db", alpha=0.8)
    ax.bar(x + width, cross_means, width, label="Cross-Group", color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Redundancy")
    ax.set_title("Mean Redundancy by Layer Depth")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{layer_idx}" for layer_idx in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Box plots
    ax = axes[1, 0]
    data_to_plot = []
    labels = []
    colors = []

    if all_halo_halo:
        data_to_plot.append(all_halo_halo[:2000])
        labels.append("Within-\nHalo")
        colors.append("#e74c3c")
    if all_non_halo:
        data_to_plot.append(all_non_halo[:2000])
        labels.append("Within-\nNon-Halo")
        colors.append("#3498db")
    if all_cross:
        data_to_plot.append(all_cross[:2000])
        labels.append("Cross-\nGroup")
        colors.append("#2ecc71")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Redundancy (MI in nats)")
    ax.set_title("Redundancy Distribution by Group")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Summary and interpretation
    ax = axes[1, 1]
    ax.axis("off")

    # Compute summary statistics
    avg_halo = np.mean(all_halo_halo) if all_halo_halo else 0
    avg_non_halo = np.mean(all_non_halo) if all_non_halo else 0
    avg_cross = np.mean(all_cross) if all_cross else 0

    std_halo = np.std(all_halo_halo) if all_halo_halo else 0
    std_non_halo = np.std(all_non_halo) if all_non_halo else 0
    std_cross = np.std(all_cross) if all_cross else 0

    # Interpretation
    if avg_halo > avg_non_halo * 1.2:
        halo_interpret = "OK Halo neurons MORE redundant -> Current approach VALID"
    elif avg_halo < avg_non_halo * 0.8:
        halo_interpret = "FAIL Non-halo MORE redundant -> Revise halo definition"
    else:
        halo_interpret = "~ Similar redundancy -> May need different criteria"

    if avg_cross < avg_halo * 0.8:
        cross_interpret = "OK Cross-group LOW -> Groups carry different info"
    else:
        cross_interpret = "~ Cross-group similar -> Info not well separated"

    if avg_halo > avg_non_halo * 1.1 and avg_cross < avg_halo * 0.9:
        echo_interpret = "OK Halo is 'echo chamber' -> Safe to prune redundant halo"
    else:
        echo_interpret = "~ Halo structure not clearly separated"

    summary_text = f"""
HALO REDUNDANCY ANALYSIS SUMMARY
================================

Group Statistics:
─────────────────────────────────────────
  Within-Halo:     μ = {avg_halo:.4f}  (σ = {std_halo:.4f})
  Within-Non-Halo: μ = {avg_non_halo:.4f}  (σ = {std_non_halo:.4f})
  Cross-Group:     μ = {avg_cross:.4f}  (σ = {std_cross:.4f})

Halo/Non-Halo Ratio: {avg_halo / max(avg_non_halo, 1e-8):.2f}x

INTERPRETATION:
─────────────────────────────────────────
{halo_interpret}

{cross_interpret}

{echo_interpret}

THEORETICAL BASIS (Gaussian mutual information):
─────────────────────────────────────────
Redundancy: I(Yᵢ; Yⱼ) = -½ log(1 - ρ²)
This is the Gaussian MI between neuron pairs.
High redundancy = neurons carry similar info.
"""

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comprehensive halo redundancy plot to {save_path}")

    return fig


def plot_halo_redundancy_heatmap(
    results_list: List[Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create heatmap showing redundancy values across layers and groups.

    Args:
        results_list: List of per-layer results
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Build data matrix
    layers = [r.get("layer_idx", i) for i, r in enumerate(results_list)]
    groups = ["Within-Halo", "Within-Non-Halo", "Cross-Group"]

    data = np.zeros((3, len(layers)))
    for i, r in enumerate(results_list):
        data[0, i] = r.get("halo_halo", {}).get("mean", 0)
        data[1, i] = r.get("non_halo_non_halo", {}).get("mean", 0)
        data[2, i] = r.get("cross_group", {}).get("mean", 0)

    # Create heatmap
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r")

    # Add colorbar
    ax.figure.colorbar(im, ax=ax, label="Mean Redundancy (MI)")

    # Set ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels([f"L{layer_idx}" for layer_idx in layers])
    ax.set_yticklabels(groups)

    # Add value annotations
    for i in range(len(groups)):
        for j in range(len(layers)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel("Layer Depth")
    ax.set_title("Halo Redundancy Heatmap by Layer and Group")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved halo redundancy heatmap to {save_path}")

    return fig
