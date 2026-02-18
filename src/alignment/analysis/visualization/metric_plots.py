"""
Metric visualization module for alignment analysis.

Provides histogram, distribution, and comparison plots for metrics like:
- Rayleigh Quotient (RQ)
- Redundancy (Gaussian MI)
- Synergy (PID-based)
- Taylor Saliency
- Outlier Index

These visualizations are used by both vision (ClusterAnalysisExperiment) and
LLM (LLMAlignmentExperiment) experiments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    Figure = None

try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


# Standard color scheme for metrics
METRIC_COLORS = {
    "rq": "#3498db",           # Blue
    "rayleigh_quotient": "#3498db",
    "redundancy": "#e74c3c",   # Red
    "synergy": "#2ecc71",      # Green
    "taylor": "#9b59b6",       # Purple
    "magnitude": "#f39c12",    # Orange
    "outlier_index": "#1abc9c", # Teal
    "activation": "#34495e",   # Dark gray
}


def plot_metric_histogram(
    values: np.ndarray,
    metric_name: str = "metric",
    layer_name: str = "",
    bins: int = 50,
    log_scale: bool = False,
    highlight_percentile: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot histogram of metric values with optional percentile highlighting.
    
    Args:
        values: 1D array of metric values
        metric_name: Name of the metric (for labels and colors)
        layer_name: Layer name for title
        bins: Number of histogram bins
        log_scale: Whether to use log scale on x-axis
        highlight_percentile: If set, highlight values above this percentile
        save_path: Optional path to save figure
        figsize: Figure size
        title: Custom title (auto-generated if None)
        
    Returns:
        Matplotlib Figure or None if matplotlib not available
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available for plotting")
        return None
    
    values = np.asarray(values).flatten()
    if len(values) == 0:
        logger.warning(f"Empty values array for {metric_name}")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get color for this metric
    color = METRIC_COLORS.get(metric_name.lower(), "#7f8c8d")
    
    # Handle log scale
    plot_values = np.log10(np.clip(values, 1e-10, None)) if log_scale else values
    
    # Plot histogram
    n, bin_edges, patches = ax.hist(
        plot_values, bins=bins, color=color, alpha=0.7, 
        edgecolor='white', linewidth=0.5
    )
    
    # Highlight percentile if requested
    if highlight_percentile is not None:
        threshold = np.percentile(values, highlight_percentile)
        threshold_plot = np.log10(max(threshold, 1e-10)) if log_scale else threshold
        ax.axvline(threshold_plot, color='red', linestyle='--', linewidth=2,
                   label=f'{highlight_percentile}th percentile: {threshold:.4f}')
        
        # Color bars above threshold
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            if left_edge >= threshold_plot:
                patch.set_facecolor('#e74c3c')
                patch.set_alpha(0.9)
        ax.legend()
    
    # Add statistics
    stats_text = f"n={len(values):,}\nμ={np.mean(values):.4f}\nσ={np.std(values):.4f}"
    stats_text += f"\nmed={np.median(values):.4f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    xlabel = f"log10({metric_name})" if log_scale else metric_name.replace('_', ' ').title()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Distribution"
        if layer_name:
            title += f" - {layer_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved histogram to {save_path}")
    
    return fig


def plot_metric_violin(
    layer_metrics: Dict[str, np.ndarray],
    metric_name: str = "metric",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
    max_points_per_layer: int = 1000,
) -> Optional[Figure]:
    """
    Plot violin plot of metric values across layers.
    
    Args:
        layer_metrics: Dict mapping layer_name -> metric values array
        metric_name: Name of the metric
        save_path: Optional path to save figure
        figsize: Figure size
        max_points_per_layer: Subsample if more points than this
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not layer_metrics:
        logger.warning(f"Empty layer_metrics for {metric_name}")
        return None
    
    # Prepare data
    plot_data = []
    layer_names = []
    
    for layer_name, values in layer_metrics.items():
        values = np.asarray(values).flatten()
        if len(values) == 0:
            continue
        
        # Subsample if needed
        if len(values) > max_points_per_layer:
            values = np.random.choice(values, max_points_per_layer, replace=False)
        
        plot_data.append(values)
        layer_names.append(layer_name)
    
    if not plot_data:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    color = METRIC_COLORS.get(metric_name.lower(), "#7f8c8d")
    
    if HAS_SEABORN:
        # Use seaborn for nicer violins
        import pandas as pd
        df_rows = []
        for layer_name, values in zip(layer_names, plot_data):
            for v in values:
                df_rows.append({"Layer": layer_name, "Value": v})
        df = pd.DataFrame(df_rows)
        sns.violinplot(data=df, x="Layer", y="Value", ax=ax, color=color, alpha=0.7)
    else:
        # Fallback to matplotlib violin
        parts = ax.violinplot(plot_data, positions=range(len(layer_names)), showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} Distribution Across Layers", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved violin plot to {save_path}")
    
    return fig


def plot_metric_boxplot(
    layer_metrics: Dict[str, np.ndarray],
    metric_name: str = "metric",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
    show_outliers: bool = True,
) -> Optional[Figure]:
    """
    Plot boxplot of metric values across layers.
    
    Args:
        layer_metrics: Dict mapping layer_name -> metric values array
        metric_name: Name of the metric
        save_path: Optional path to save figure
        figsize: Figure size
        show_outliers: Whether to show outlier points
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not layer_metrics:
        return None
    
    layer_names = list(layer_metrics.keys())
    data = [np.asarray(layer_metrics[l]).flatten() for l in layer_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    color = METRIC_COLORS.get(metric_name.lower(), "#7f8c8d")
    
    bp = ax.boxplot(data, patch_artist=True, showfliers=show_outliers)
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Layer", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved boxplot to {save_path}")
    
    return fig


def plot_multi_metric_histogram(
    metrics: Dict[str, np.ndarray],
    layer_name: str = "",
    bins: int = 30,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> Optional[Figure]:
    """
    Plot histograms of multiple metrics side by side.
    
    Args:
        metrics: Dict mapping metric_name -> values array
        layer_name: Layer name for title
        bins: Number of histogram bins
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not metrics:
        return None
    
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, metric_names):
        values = np.asarray(metrics[metric_name]).flatten()
        if len(values) == 0:
            continue
        
        color = METRIC_COLORS.get(metric_name.lower(), "#7f8c8d")
        
        ax.hist(values, bins=bins, color=color, alpha=0.7, 
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        
        # Add statistics
        stats = f"μ={np.mean(values):.3f}\nσ={np.std(values):.3f}"
        ax.text(0.98, 0.98, stats, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(True, alpha=0.3, axis='y')
    
    title = "Metric Distributions"
    if layer_name:
        title += f" - {layer_name}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-metric histogram to {save_path}")
    
    return fig


def plot_metric_scatter_matrix(
    metrics: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    label_names: Optional[Dict[int, str]] = None,
    layer_name: str = "",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 12),
    max_points: int = 2000,
) -> Optional[Figure]:
    """
    Plot scatter matrix of all metric pairs with optional cluster coloring.
    
    Args:
        metrics: Dict mapping metric_name -> values array
        labels: Optional cluster labels for coloring
        label_names: Optional mapping of label id -> name
        layer_name: Layer name for title
        save_path: Optional path to save figure
        figsize: Figure size
        max_points: Maximum points to plot (subsampled if more)
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not metrics or len(metrics) < 2:
        return None
    
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    
    # Get sample size
    n_samples = len(metrics[metric_names[0]])
    
    # Subsample if needed
    if n_samples > max_points:
        idx = np.random.choice(n_samples, max_points, replace=False)
    else:
        idx = np.arange(n_samples)
    
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)
    
    # Cluster colors
    cluster_colors = {
        "critical": "#e74c3c",
        "redundant": "#3498db",
        "synergistic": "#2ecc71",
        "background": "#95a5a6",
    }
    
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                values = np.asarray(metrics[m1])[idx]
                color = METRIC_COLORS.get(m1.lower(), "#7f8c8d")
                ax.hist(values, bins=30, color=color, alpha=0.7)
                ax.set_ylabel("Count" if j == 0 else "")
            else:
                # Off-diagonal: scatter
                x = np.asarray(metrics[m2])[idx]
                y = np.asarray(metrics[m1])[idx]
                
                if labels is not None:
                    labels_subset = np.asarray(labels)[idx]
                    unique_labels = np.unique(labels_subset)
                    for lbl in unique_labels:
                        mask = labels_subset == lbl
                        lbl_name = label_names.get(lbl, str(lbl)) if label_names else str(lbl)
                        color = cluster_colors.get(lbl_name.lower(), "#7f8c8d")
                        ax.scatter(x[mask], y[mask], c=color, alpha=0.5, s=10, label=lbl_name)
                else:
                    ax.scatter(x, y, alpha=0.3, s=10, c='#34495e')
            
            # Labels only on edges
            if i == n_metrics - 1:
                ax.set_xlabel(m2.replace('_', ' ').title(), fontsize=10)
            if j == 0:
                ax.set_ylabel(m1.replace('_', ' ').title(), fontsize=10)
            
            ax.tick_params(labelsize=8)
    
    # Add legend to top-right plot if labels provided
    if labels is not None and label_names:
        axes[0, -1].legend(loc='upper right', fontsize=8)
    
    title = "Metric Scatter Matrix"
    if layer_name:
        title += f" - {layer_name}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved scatter matrix to {save_path}")
    
    return fig


def plot_metric_correlation_heatmap(
    metrics: Dict[str, np.ndarray],
    layer_name: str = "",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[Figure]:
    """
    Plot correlation heatmap between metrics.
    
    Args:
        metrics: Dict mapping metric_name -> values array
        layer_name: Layer name for title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not metrics or len(metrics) < 2:
        return None
    
    metric_names = list(metrics.keys())
    n_metrics = len(metric_names)
    
    # Compute correlation matrix
    data = np.column_stack([np.asarray(metrics[m]).flatten() for m in metric_names])
    corr = np.corrcoef(data.T)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SEABORN:
        sns.heatmap(corr, xticklabels=metric_names, yticklabels=metric_names,
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax)
    else:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n_metrics))
        ax.set_yticks(range(n_metrics))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(metric_names)
        
        # Add annotations
        for i in range(n_metrics):
            for j in range(n_metrics):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Correlation')
    
    title = "Metric Correlations"
    if layer_name:
        title += f" - {layer_name}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
    
    return fig


def plot_layer_metric_heatmap(
    layer_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
    normalize_metrics: bool = True,
) -> Optional[Figure]:
    """
    Plot heatmap of mean metric values across layers.
    
    Args:
        layer_metrics: Dict mapping layer_name -> {metric_name -> mean_value}
        save_path: Optional path to save figure
        figsize: Figure size
        normalize_metrics: Whether to normalize each metric to [0, 1]
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    if not layer_metrics:
        return None
    
    layer_names = list(layer_metrics.keys())
    metric_names = list(layer_metrics[layer_names[0]].keys())
    
    # Build matrix
    matrix = np.zeros((len(layer_names), len(metric_names)))
    for i, layer in enumerate(layer_names):
        for j, metric in enumerate(metric_names):
            matrix[i, j] = layer_metrics[layer].get(metric, 0)
    
    # Normalize if requested
    if normalize_metrics:
        for j in range(len(metric_names)):
            col = matrix[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                matrix[:, j] = (col - col_min) / (col_max - col_min)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if HAS_SEABORN:
        sns.heatmap(matrix, xticklabels=metric_names, yticklabels=layer_names,
                    annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    else:
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(layer_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_yticklabels(layer_names)
        plt.colorbar(im, ax=ax)
    
    ax.set_title("Mean Metrics Across Layers", fontsize=14, fontweight='bold')
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved layer metric heatmap to {save_path}")
    
    return fig


def plot_top_neurons_bar(
    values: np.ndarray,
    metric_name: str = "metric",
    layer_name: str = "",
    top_k: int = 20,
    show_bottom: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[Figure]:
    """
    Plot bar chart of top-k (and optionally bottom-k) neurons by metric value.
    
    Args:
        values: 1D array of metric values per neuron
        metric_name: Name of the metric
        layer_name: Layer name for title
        top_k: Number of top/bottom neurons to show
        show_bottom: Whether to also show bottom-k neurons
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None
    """
    if not HAS_MPL:
        return None
    
    values = np.asarray(values).flatten()
    if len(values) == 0:
        return None
    
    sorted_idx = np.argsort(values)
    top_idx = sorted_idx[-top_k:][::-1]
    bottom_idx = sorted_idx[:top_k]
    
    n_plots = 2 if show_bottom else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Top neurons
    ax = axes[0]
    ax.bar(range(top_k), values[top_idx], color='#2ecc71', alpha=0.8)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"N{i}" for i in top_idx], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f"Top {top_k} Highest {metric_name.replace('_', ' ').title()}", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Bottom neurons
    if show_bottom:
        ax = axes[1]
        ax.bar(range(top_k), values[bottom_idx], color='#e74c3c', alpha=0.8)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([f"N{i}" for i in bottom_idx], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"Top {top_k} Lowest {metric_name.replace('_', ' ').title()}", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    title = f"{metric_name.replace('_', ' ').title()} - Extreme Neurons"
    if layer_name:
        title += f" ({layer_name})"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved top neurons bar chart to {save_path}")
    
    return fig
