"""
Cluster visualization module for neural network analysis.

Provides visualizations for:
1. Metric space scatter plots (RQ vs Red, RQ vs Syn, Red vs Syn)
2. Cluster composition across depth (stacked bars)
3. Cross-layer influence matrices (heatmaps)
4. Cluster stability analysis
5. Cascade test results by cluster type
6. Pruning comparisons (unified with LLM experiments)

This module works for both vision (ResNet, VGG) and LLM (Qwen, Llama) experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Import unified pruning functions
from .pruning_plots import (
    plot_unified_pruning_comparison,
    plot_pruning_accuracy_loss_grid,
    plot_pruning_recovery_chart,
    PRUNING_METHOD_COLORS,
)

# Import metric plotting functions
from .metric_plots import (
    plot_metric_histogram,
    plot_metric_violin,
    plot_multi_metric_histogram,
    plot_metric_correlation_heatmap,
    plot_top_neurons_bar,
    METRIC_COLORS,
)


CLUSTER_COLORS = {
    "critical": "#e74c3c",
    "redundant": "#3498db", 
    "synergistic": "#2ecc71",
    "background": "#95a5a6",
    "unknown": "#bdc3c7",
}


def plot_metric_scatter(
    rq: np.ndarray,
    redundancy: np.ndarray,
    synergy: np.ndarray,
    labels: np.ndarray,
    type_mapping: Dict[int, str],
    layer_name: str = "",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Optional["plt.Figure"]:
    """
    Plot 2D projections of metric space with cluster colors.
    
    Creates 3 subplots: RQ vs Red, RQ vs Syn, Red vs Syn
    """
    if not HAS_MPL:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    log_rq = np.log(np.clip(rq, 1e-10, None))
    
    pairs = [
        (log_rq, redundancy, "log(RQ)", "Redundancy"),
        (log_rq, synergy, "log(RQ)", "Synergy"),
        (redundancy, synergy, "Redundancy", "Synergy"),
    ]
    
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        for cid, ctype in type_mapping.items():
            mask = labels == cid
            color = CLUSTER_COLORS.get(ctype, "#999999")
            ax.scatter(x[mask], y[mask], c=color, label=ctype, alpha=0.6, s=20)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Metric Space Clusters: {layer_name}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster scatter to {save_path}")
    
    return fig


def plot_cluster_evolution(
    layer_results: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional["plt.Figure"]:
    """
    Plot cluster composition across network depth as stacked bars.
    
    Args:
        layer_results: List of dicts with 'layer_name' and 'type_counts'
    """
    if not HAS_MPL:
        return None
    
    layers = [r["layer_name"] for r in layer_results]
    types = ["critical", "redundant", "synergistic", "background"]
    
    # Build data matrix
    data = {t: [] for t in types}
    for r in layer_results:
        tc = r.get("type_counts", {})
        total = sum(tc.values()) or 1
        for t in types:
            data[t].append(tc.get(t, 0) / total * 100)
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(layers))
    bottom = np.zeros(len(layers))
    
    for t in types:
        color = CLUSTER_COLORS.get(t, "#999999")
        ax.bar(x, data[t], bottom=bottom, label=t, color=color, alpha=0.8)
        bottom += np.array(data[t])
    
    ax.set_xlabel("Layer")
    ax.set_ylabel("Percentage of Channels")
    ax.set_title("Cluster Composition Across Depth")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster evolution to {save_path}")
    
    return fig


def plot_influence_matrix(
    flow: Dict[str, Dict[str, float]],
    layer_name: str = "",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional["plt.Figure"]:
    """
    Plot cluster-to-cluster influence matrix as heatmap.
    
    Args:
        flow: Nested dict flow[source_type][target_type] = value
    """
    if not HAS_MPL:
        return None
    
    types = ["critical", "redundant", "synergistic", "background"]
    matrix = np.zeros((len(types), len(types)))
    
    for i, src in enumerate(types):
        for j, tgt in enumerate(types):
            matrix[i, j] = flow.get(src, {}).get(tgt, 0)
    
    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix_norm, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(types)))
    ax.set_yticks(np.arange(len(types)))
    ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_yticklabels([t.capitalize() for t in types])
    ax.set_xlabel("Target Cluster (Layer ℓ+1)")
    ax.set_ylabel("Source Cluster (Layer ℓ)")
    ax.set_title(f"Cross-Cluster Influence: {layer_name}")
    
    # Add annotations
    for i in range(len(types)):
        for j in range(len(types)):
            ax.text(j, i, f"{matrix_norm[i, j]:.2f}",
                   ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax, label="Normalized Influence")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved influence matrix to {save_path}")
    
    return fig


def plot_cascade_test(
    results: Dict[str, Any],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional["plt.Figure"]:
    """
    Plot cascade test results by cluster type.
    
    Args:
        results: Dict mapping cluster_type to CascadeResult
    """
    if not HAS_MPL:
        return None
    
    types = list(results.keys())
    acc_drops = [results[t].accuracy_drop * 100 for t in types]
    colors = [CLUSTER_COLORS.get(t, "#999999") for t in types]
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(types))
    bars = ax.bar(x, acc_drops, color=colors, alpha=0.8)
    
    ax.set_xlabel("Cluster Type")
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("Cascade Damage by Cluster Type")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types])
    
    # Add value labels
    for bar, val in zip(bars, acc_drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax.axhline(y=np.mean(acc_drops), color='gray', linestyle='--', 
               label=f'Mean: {np.mean(acc_drops):.2f}%')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cascade test to {save_path}")
    
    return fig


def plot_halo_properties(
    halo_results: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> Optional["plt.Figure"]:
    """
    Plot halo redundancy and synergy by source cluster type.
    
    Args:
        halo_results: List of dicts with cluster_type, halo_red, halo_syn
    """
    if not HAS_MPL:
        return None
    
    types = [r["cluster_type"] for r in halo_results]
    reds = [r.get("halo_red", 0) for r in halo_results]
    syns = [r.get("halo_syn", 0) for r in halo_results]
    colors = [CLUSTER_COLORS.get(t, "#999999") for t in types]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(types))
    
    axes[0].bar(x, reds, color=colors, alpha=0.8)
    axes[0].set_ylabel("Halo Redundancy")
    axes[0].set_title("Halo Redundancy by Source Cluster")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([t.capitalize() for t in types])
    axes[0].axhline(y=np.mean(reds), color='gray', linestyle='--')
    
    axes[1].bar(x, syns, color=colors, alpha=0.8)
    axes[1].set_ylabel("Halo Synergy")
    axes[1].set_title("Halo Synergy by Source Cluster")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([t.capitalize() for t in types])
    axes[1].axhline(y=np.mean(syns), color='gray', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved halo properties to {save_path}")
    
    return fig


def plot_centroid_evolution(
    layer_centroids: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Optional["plt.Figure"]:
    """
    Plot how cluster centroids evolve across network depth.
    
    Creates 2D trajectory plots showing centroid movement in:
    - log(RQ) vs Redundancy
    - log(RQ) vs Synergy  
    - Redundancy vs Synergy
    
    Args:
        layer_centroids: List of dicts with 'layer_name', 'depth', 'centroids', 'type_mapping'
            where centroids is [n_clusters, 3] array (log_rq, red, syn)
        save_path: Optional path to save figure
    """
    if not HAS_MPL:
        return None
    
    if not layer_centroids:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    types = ["critical", "redundant", "synergistic", "background"]
    
    # Collect centroid trajectories by type
    trajectories = {t: {"log_rq": [], "red": [], "syn": [], "depth": []} for t in types}
    
    for layer_data in layer_centroids:
        centroids = np.array(layer_data["centroids"])  # [n_clusters, 3]
        type_mapping = layer_data.get("type_mapping", {})
        depth = layer_data.get("depth", 0)
        
        for cluster_id, cluster_type in type_mapping.items():
            if cluster_type in trajectories and int(cluster_id) < len(centroids):
                c = centroids[int(cluster_id)]
                trajectories[cluster_type]["log_rq"].append(c[0])
                trajectories[cluster_type]["red"].append(c[1])
                trajectories[cluster_type]["syn"].append(c[2])
                trajectories[cluster_type]["depth"].append(depth)
    
    # Plot pairs
    pairs = [
        ("log_rq", "red", "log(RQ)", "Redundancy"),
        ("log_rq", "syn", "log(RQ)", "Synergy"),
        ("red", "syn", "Redundancy", "Synergy"),
    ]
    
    for ax, (x_key, y_key, x_label, y_label) in zip(axes, pairs):
        for ctype in types:
            traj = trajectories[ctype]
            if not traj["depth"]:
                continue
            
            # Sort by depth
            sorted_idx = np.argsort(traj["depth"])
            x_vals = np.array(traj[x_key])[sorted_idx]
            y_vals = np.array(traj[y_key])[sorted_idx]
            depths = np.array(traj["depth"])[sorted_idx]
            
            color = CLUSTER_COLORS.get(ctype, "#999999")
            
            # Plot trajectory with arrows
            ax.plot(x_vals, y_vals, '-', color=color, alpha=0.7, linewidth=2, label=ctype)
            
            # Add markers with depth coloring
            scatter = ax.scatter(x_vals, y_vals, c=depths, cmap='viridis', 
                                s=80, edgecolors=color, linewidths=2, zorder=5)
            
            # Add start/end markers
            if len(x_vals) > 0:
                ax.scatter(x_vals[0], y_vals[0], marker='o', s=150, 
                          facecolors='white', edgecolors=color, linewidths=3, zorder=6)
                ax.scatter(x_vals[-1], y_vals[-1], marker='s', s=150,
                          facecolors=color, edgecolors='black', linewidths=2, zorder=6)
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Add colorbar for depth
    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.12, aspect=40)
    cbar.set_label('Layer Depth', fontsize=11)
    
    fig.suptitle('Cluster Centroid Evolution Across Network Depth\n(○ = early layers, ■ = late layers)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved centroid evolution to {save_path}")
    
    return fig


def plot_centroid_depth_profiles(
    layer_centroids: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional["plt.Figure"]:
    """
    Plot each metric's centroid value vs depth for each cluster type.
    
    Shows how log(RQ), Redundancy, and Synergy change with depth
    for each functional type.
    """
    if not HAS_MPL:
        return None
    
    if not layer_centroids:
        return None
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    types = ["critical", "redundant", "synergistic", "background"]
    metrics = [("log_rq", "log(RQ) - Alignment"), 
               ("red", "Redundancy"), 
               ("syn", "Synergy")]
    
    # Collect data
    data = {t: {"depth": [], "log_rq": [], "red": [], "syn": []} for t in types}
    
    for layer_data in layer_centroids:
        centroids = np.array(layer_data["centroids"])
        type_mapping = layer_data.get("type_mapping", {})
        depth = layer_data.get("depth", 0)
        
        for cluster_id, cluster_type in type_mapping.items():
            if cluster_type in data and int(cluster_id) < len(centroids):
                c = centroids[int(cluster_id)]
                data[cluster_type]["depth"].append(depth)
                data[cluster_type]["log_rq"].append(c[0])
                data[cluster_type]["red"].append(c[1])
                data[cluster_type]["syn"].append(c[2])
    
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for ctype in types:
            d = data[ctype]
            if not d["depth"]:
                continue
            
            sorted_idx = np.argsort(d["depth"])
            depths = np.array(d["depth"])[sorted_idx]
            values = np.array(d[metric_key])[sorted_idx]
            
            color = CLUSTER_COLORS.get(ctype, "#999999")
            ax.plot(depths, values, 'o-', color=color, label=ctype, 
                   linewidth=2, markersize=6)
        
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)
    
    axes[-1].set_xlabel('Layer Depth', fontsize=11)
    fig.suptitle('Cluster Centroid Metrics vs Network Depth', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved centroid depth profiles to {save_path}")
    
    return fig


def plot_pruning_comparison(
    results: Dict[str, Dict[float, Dict[str, float]]],
    baseline_acc: float,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional["plt.Figure"]:
    """
    Plot pruning accuracy comparison across methods and sparsity levels.
    
    This is a convenience wrapper around plot_unified_pruning_comparison
    for backward compatibility with vision experiments.
    
    Args:
        results: Dict mapping method -> {ratio -> {'accuracy_after_ft': float}}
        baseline_acc: Baseline (unpruned) accuracy
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    # Use the unified pruning comparison function
    return plot_unified_pruning_comparison(
        results=results,
        baseline_value=baseline_acc,
        metric='accuracy',
        higher_is_better=True,
        title='Pruning Method Comparison',
        save_path=save_path,
        figsize=figsize,
    )


def plot_metric_distributions_for_layer(
    metrics: Dict[str, np.ndarray],
    layer_name: str = "",
    save_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Optional["plt.Figure"]:
    """
    Plot histograms of all metrics (RQ, Redundancy, Synergy) for a single layer.
    
    This is a convenience function that combines multiple metric histograms
    into a single figure for a given layer.
    
    Args:
        metrics: Dict with keys like 'rq', 'redundancy', 'synergy' mapping to arrays
        layer_name: Name of the layer for title
        save_dir: Directory to save figure (filename auto-generated)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL or not metrics:
        return None
    
    # Use the multi-metric histogram from metric_plots
    save_path = None
    if save_dir:
        safe_name = layer_name.replace('.', '_').replace('/', '_')
        save_path = Path(save_dir) / f"metric_distributions_{safe_name}.png"
    
    return plot_multi_metric_histogram(
        metrics=metrics,
        layer_name=layer_name,
        bins=30,
        save_path=save_path,
        figsize=figsize,
    )


def plot_layer_metric_summary(
    layer_metrics: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional["plt.Figure"]:
    """
    Plot summary statistics of all metrics across all layers.
    
    Creates a 2x2 grid with:
    - Mean values heatmap
    - Standard deviation heatmap
    - Metric correlations (averaged across layers)
    - Layer-wise metric ranges
    
    Args:
        layer_metrics: Dict mapping layer_name -> {metric_name -> values}
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL or not layer_metrics:
        return None
    
    layer_names = list(layer_metrics.keys())
    if not layer_names:
        return None
    
    # Get all metric names from first layer
    metric_names = list(layer_metrics[layer_names[0]].keys())
    if not metric_names:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Mean values heatmap
    ax = axes[0, 0]
    mean_matrix = np.zeros((len(layer_names), len(metric_names)))
    for i, layer in enumerate(layer_names):
        for j, metric in enumerate(metric_names):
            values = layer_metrics[layer].get(metric, [])
            mean_matrix[i, j] = np.mean(values) if len(values) > 0 else 0
    
    im = ax.imshow(mean_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_title('Mean Values by Layer', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    # 2. Std values heatmap
    ax = axes[0, 1]
    std_matrix = np.zeros((len(layer_names), len(metric_names)))
    for i, layer in enumerate(layer_names):
        for j, metric in enumerate(metric_names):
            values = layer_metrics[layer].get(metric, [])
            std_matrix[i, j] = np.std(values) if len(values) > 0 else 0
    
    im = ax.imshow(std_matrix, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(metric_names)))
    ax.set_yticks(range(len(layer_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_title('Std Dev by Layer', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    # 3. Average correlation across layers
    ax = axes[1, 0]
    corr_sum = np.zeros((len(metric_names), len(metric_names)))
    n_valid = 0
    for layer in layer_names:
        try:
            data = np.column_stack([
                np.asarray(layer_metrics[layer].get(m, [])).flatten() 
                for m in metric_names
            ])
            if data.shape[0] > 10:  # Need enough samples
                corr = np.corrcoef(data.T)
                if not np.any(np.isnan(corr)):
                    corr_sum += corr
                    n_valid += 1
        except:
            continue
    
    if n_valid > 0:
        avg_corr = corr_sum / n_valid
        im = ax.imshow(avg_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(metric_names, fontsize=9)
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                ax.text(j, i, f'{avg_corr[i, j]:.2f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax)
    ax.set_title('Average Metric Correlation', fontsize=12)
    
    # 4. Metric ranges across depth
    ax = axes[1, 1]
    for i, metric in enumerate(metric_names):
        means = []
        stds = []
        for layer in layer_names:
            values = layer_metrics[layer].get(metric, [])
            if len(values) > 0:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(0)
                stds.append(0)
        
        color = METRIC_COLORS.get(metric.lower(), f'C{i}')
        x = np.arange(len(layer_names))
        ax.plot(x, means, 'o-', label=metric, color=color, linewidth=1.5, markersize=4)
        ax.fill_between(x, np.array(means) - np.array(stds), 
                       np.array(means) + np.array(stds), alpha=0.2, color=color)
    
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Value (mean ± std)', fontsize=10)
    ax.set_title('Metrics Across Depth', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Layer Metric Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved layer metric summary to {save_path}")
    
    return fig


# Legacy function name kept for backward compatibility
def plot_vision_pruning_comparison(
    results: Dict[str, Dict[float, Dict[str, float]]],
    baseline_acc: float,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional["plt.Figure"]:
    """Alias for plot_pruning_comparison for backward compatibility."""
    return plot_pruning_comparison(results, baseline_acc, save_path, figsize)
