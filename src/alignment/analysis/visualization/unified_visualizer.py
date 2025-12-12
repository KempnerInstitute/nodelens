"""
Unified visualization module for alignment analysis.

This module consolidates visualization functionality from multiple modules,
providing a single interface for all visualization needs.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Try to import seaborn, but make it optional
try:
    import seaborn as sns

    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class UnifiedVisualizer:
    """
    Unified visualization class that combines functionality from:
    - MetricVisualizer
    - AlignmentVisualizer
    - PruningVisualizer
    - ComparisonVisualizer

    This provides a single interface for all visualization needs.
    """

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the unified visualizer.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        # Set style
        try:
            plt.style.use(style)
        except Exception:
            try:
                plt.style.use("seaborn-v0_8-darkgrid")
            except Exception:
                plt.style.use("default")

        self.figsize = figsize
        self.dpi = 300  # Default DPI for saved figures

        # Define color palettes
        if HAS_SEABORN:
            self.colors = sns.color_palette("husl", 10)
        else:
            import matplotlib.cm as cm

            self.colors = [cm.tab10(i) for i in range(10)]

        # Extended colors for strategies
        self.strategy_colors = {
            "magnitude": "#1f77b4",
            "gradient": "#ff7f0e",
            "fisher": "#2ca02c",
            "random": "#d62728",
            "low": "#9467bd",
            "high": "#8c564b",
        }

        # Set global parameters
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    # ========== Time Series Plots ==========

    def plot_metric_evolution(
        self,
        steps: List[int],
        values: Dict[str, List[float]],
        title: str = "Metric Evolution",
        xlabel: str = "Step",
        ylabel: str = "Value",
        legend_title: str = "Series",
        show_confidence: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot the evolution of metrics over time with optional confidence intervals.

        Args:
            steps: List of step numbers
            values: Dictionary mapping series names to value lists
                   Can also contain 'mean' and 'std' keys for confidence intervals
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_title: Legend title
            show_confidence: Whether to show confidence intervals
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, vals) in enumerate(values.items()):
            if name in ["mean", "std"]:
                continue

            color = self.colors[i % len(self.colors)]

            if isinstance(vals, dict) and "mean" in vals:
                # Handle mean/std structure
                means = vals["mean"]
                ax.plot(steps[: len(means)], means, label=name, color=color, linewidth=2)

                if show_confidence and "std" in vals:
                    stds = vals["std"]
                    means = np.array(means)
                    stds = np.array(stds)
                    ax.fill_between(steps[: len(means)], means - stds, means + stds, alpha=0.2, color=color)
            else:
                # Simple list of values
                ax.plot(steps[: len(vals)], vals, label=name, color=color, linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(title=legend_title, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ========== Layer Analysis Plots ==========

    def plot_layer_scores(
        self,
        scores: Dict[str, Union[torch.Tensor, np.ndarray, List[float]]],
        metric_name: str,
        plot_type: str = "violin",
        save_path: Optional[str] = None,
        show_statistics: bool = True,
    ) -> Figure:
        """
        Plot alignment scores across layers.

        Args:
            scores: Dictionary of layer_name -> scores
            metric_name: Name of the metric
            plot_type: Type of plot ('violin', 'box', 'bar')
            save_path: Path to save the plot
            show_statistics: Whether to show mean/std statistics

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        layer_names = list(scores.keys())
        data = []

        for layer_name, layer_scores in scores.items():
            if isinstance(layer_scores, torch.Tensor):
                layer_scores = layer_scores.cpu().numpy()
            elif not isinstance(layer_scores, np.ndarray):
                layer_scores = np.array(layer_scores)
            data.append(layer_scores)

        positions = range(len(layer_names))

        if plot_type == "violin":
            parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor("lightblue")
                pc.set_alpha(0.7)
        elif plot_type == "box":
            ax.boxplot(data, positions=positions, labels=layer_names, showfliers=False)
        elif plot_type == "bar":
            means = [np.mean(d) for d in data]
            stds = [np.std(d) for d in data]
            bars = ax.bar(positions, means, yerr=stds, capsize=5)

            # Color bars by value
            norm = plt.Normalize(min(means), max(means))
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            for bar, val in zip(bars, means):
                bar.set_color(sm.to_rgba(val))

        if plot_type != "bar":
            ax.set_xticks(positions)
            ax.set_xticklabels(layer_names, rotation=45, ha="right")

        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{metric_name} Score")
        ax.set_title(f"{metric_name} Distribution Across Layers")

        if show_statistics and plot_type != "bar":
            stats_text = []
            for i, (name, layer_scores) in enumerate(zip(layer_names, data)):
                mean = np.mean(layer_scores)
                std = np.std(layer_scores)
                stats_text.append(f"{name}: μ={mean:.3f}, σ={std:.3f}")

            textstr = "\n".join(stats_text)
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9, verticalalignment="top", bbox=props)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_importance_histogram(
        self,
        scores: Union[torch.Tensor, np.ndarray, List[float]],
        layer_name: str,
        metric_name: str,
        plots_dir: Union[str, Path],
        top_k: int = 5,
    ) -> Path:
        """
        Plot a histogram of per-neuron importance scores highlighting the top-k neurons.
        """
        if isinstance(scores, torch.Tensor):
            tensor = scores.detach().cpu().to(torch.float32)
            values = tensor.numpy()
        else:
            values = np.asarray(scores, dtype=np.float32)
            tensor = torch.tensor(values)

        if tensor.numel() == 0:
            raise ValueError(f"No importance scores available for {layer_name}/{metric_name}")

        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(values, bins=100, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of Importance Scores — {layer_name}\nMetric: {metric_name}")

        y_max = ax.get_ylim()[1]
        k = min(top_k, tensor.numel())
        if k > 0:
            topk_values, topk_indices = torch.topk(tensor, k=k)
            for i, (idx, val) in enumerate(zip(topk_indices.tolist(), topk_values.tolist())):
                ax.axvline(val, linestyle="--", linewidth=2, label=f"Neuron {idx}: {val:.4f}")
                ax.text(
                    val,
                    y_max * (0.95 - i * 0.05),
                    f"Neuron {idx} ({val:.4f})",
                    fontsize=10,
                    color="red",
                )
            ax.legend(fontsize=9)

        fig.tight_layout()
        safe = layer_name.replace(".", "_").replace("/", "_")
        save_path = plots_dir / f"{safe}_{metric_name}_importance_histogram.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_neuron_outgoing_weights(
        self,
        weights: Union[torch.Tensor, np.ndarray],
        layer_name: str,
        neuron_index: int,
        plots_dir: Union[str, Path],
        top_k: int = 5,
    ) -> Path:
        """
        Plot a histogram of outgoing weights for a neuron, highlighting top-k connections.
        """
        if isinstance(weights, torch.Tensor):
            W = weights.detach().cpu().to(torch.float32)
        else:
            W = torch.tensor(weights, dtype=torch.float32)

        if neuron_index < 0 or neuron_index >= W.shape[1]:
            raise ValueError(f"Neuron index {neuron_index} out of range for layer '{layer_name}'")

        outgoing = W[:, neuron_index].numpy()
        magnitudes = np.abs(outgoing)
        k = min(top_k, outgoing.shape[0])
        top_idxs = np.argpartition(-magnitudes, range(k))[:k]

        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(outgoing, bins=80, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Outgoing Weight Value")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Outgoing Weights Histogram — {layer_name}\nNeuron {neuron_index}")

        y_max = ax.get_ylim()[1]
        for i, idx in enumerate(top_idxs):
            val = outgoing[idx]
            ax.axvline(val, linestyle="--", linewidth=2)
            ax.text(
                val,
                y_max * (0.95 - i * 0.05),
                f"to output {idx}: {val:.4f}",
                fontsize=10,
                color="red",
            )

        fig.tight_layout()
        safe = layer_name.replace(".", "_").replace("/", "_")
        save_path = plots_dir / f"{safe}_neuron_{neuron_index}_outgoing_weights.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    # ========== Single-Distribution & Scatter Plots ==========

    def plot_1d_histogram(
        self,
        values: Union[torch.Tensor, np.ndarray, List[float]],
        title: str = "Histogram",
        xlabel: str = "value",
        ylabel: str = "Count",
        bins: int = 100,
        logx: bool = False,
        color: Optional[str] = None,
        vline: Optional[float] = None,
        vline_label: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot a 1D histogram for a single metric (e.g., activation variance, RQ, MI).

        Args:
            values: 1D tensor/array/list of scalar values.
            title:  Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            bins:   Number of histogram bins.
            logx:   If True, plot log10 of positive values.
            color:  Bar color.
            vline:  Optional vertical line position.
            vline_label: Label for vertical line.
            save_path: Optional path to save the figure.
        """
        if isinstance(values, torch.Tensor):
            arr = values.detach().cpu().to(torch.float32).numpy()
        elif isinstance(values, np.ndarray):
            arr = values.astype(np.float32, copy=False)
        else:
            arr = np.asarray(values, dtype=np.float32)

        # Remove non-finite values
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            logger.warning("plot_1d_histogram: no finite values to plot.")
            fig, _ = plt.subplots(figsize=self.figsize)
            return fig

        if logx:
            arr = arr[arr > 0]
            if arr.size == 0:
                logger.warning("plot_1d_histogram: no positive values for log-scale histogram.")
                fig, _ = plt.subplots(figsize=self.figsize)
                return fig
            arr = np.log10(arr)

        fig, ax = plt.subplots(figsize=self.figsize)
        hist_kwargs = {"bins": bins, "alpha": 0.7, "edgecolor": "black"}
        if color:
            hist_kwargs["color"] = color
        ax.hist(arr, **hist_kwargs)
        
        if vline is not None:
            ax.axvline(vline, color='r', linestyle='--', linewidth=2, label=vline_label)
            if vline_label:
                ax.legend()
        
        ax.set_title(title)
        ax.set_xlabel("log10(value)" if logx else xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved histogram to {save_path}")

        return fig

    def plot_scatter_2d(
        self,
        x: Union[torch.Tensor, np.ndarray, List[float]],
        y: Union[torch.Tensor, np.ndarray, List[float]],
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.5,
        s: float = 10.0,
    ) -> Figure:
        """
        Simple 2D scatter plot helper, useful for relationships like activation vs RQ.

        Args:
            x: X-axis values.
            y: Y-axis values (must be same length as x).
            xlabel: Label for x-axis.
            ylabel: Label for y-axis.
            title: Plot title.
            save_path: Optional path to save the figure.
            alpha: Point transparency.
            s: Point size.
        """
        def _to_array(z):
            if isinstance(z, torch.Tensor):
                return z.detach().cpu().to(torch.float32).numpy()
            if isinstance(z, np.ndarray):
                return z.astype(np.float32, copy=False)
            return np.asarray(z, dtype=np.float32)

        x_arr = _to_array(x)
        y_arr = _to_array(y)

        # Mask to finite pairs
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr = x_arr[mask]
        y_arr = y_arr[mask]

        if x_arr.size == 0 or y_arr.size == 0:
            logger.warning("plot_scatter_2d: no finite (x, y) pairs to plot.")
            fig, _ = plt.subplots(figsize=self.figsize)
            return fig

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(x_arr, y_arr, s=s, alpha=alpha, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved scatter plot to {save_path}")

        return fig

    def plot_supernode_comparison(
        self,
        scores: Dict[str, Union[torch.Tensor, np.ndarray]],
        supernode_mask: Union[torch.Tensor, np.ndarray],
        metrics: List[str],
        layer_name: str = "",
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Figure]:
        """
        Plot comparison of metrics between supernodes and non-supernodes.

        Creates violin plots, histograms, and summary statistics comparing
        supernode neurons to regular neurons for each metric.

        Args:
            scores: Dictionary mapping metric names to score arrays [num_neurons]
            supernode_mask: Boolean mask indicating supernodes [num_neurons]
            metrics: List of metric names to compare
            layer_name: Name of the layer for plot titles
            save_dir: Directory to save plots

        Returns:
            List of generated figures
        """
        def _to_array(z):
            if isinstance(z, torch.Tensor):
                return z.detach().cpu().to(torch.float32).numpy()
            return np.asarray(z, dtype=np.float32)

        mask = _to_array(supernode_mask).astype(bool)
        figures = []

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for metric_name in metrics:
            if metric_name not in scores:
                logger.warning(f"Metric '{metric_name}' not found in scores, skipping.")
                continue

            vals = _to_array(scores[metric_name])
            if vals.shape[0] != mask.shape[0]:
                logger.warning(f"Score shape {vals.shape} doesn't match mask shape {mask.shape}, skipping {metric_name}.")
                continue

            supernode_vals = vals[mask]
            non_supernode_vals = vals[~mask]

            if len(supernode_vals) == 0 or len(non_supernode_vals) == 0:
                logger.warning(f"Empty group for {metric_name}, skipping.")
                continue

            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Violin/box plot comparison
            ax = axes[0]
            data = [supernode_vals, non_supernode_vals]
            labels = [f"Supernodes\n(n={len(supernode_vals)})", f"Non-supernodes\n(n={len(non_supernode_vals)})"]
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True, showfliers=False)
            bp['boxes'][0].set_facecolor('coral')
            bp['boxes'][1].set_facecolor('steelblue')
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name}: Supernodes vs Non-supernodes")
            
            # Add mean markers
            for i, d in enumerate(data):
                ax.scatter([i + 1], [np.mean(d)], color='darkred', marker='D', s=50, zorder=3, label='Mean' if i == 0 else '')
            ax.legend()

            # Histogram comparison
            ax = axes[1]
            bins = np.linspace(min(vals.min(), 0), vals.max(), 50)
            ax.hist(supernode_vals, bins=bins, alpha=0.6, color='coral', 
                   label=f"Supernodes (mean={np.mean(supernode_vals):.4f})", density=True)
            ax.hist(non_supernode_vals, bins=bins, alpha=0.6, color='steelblue',
                   label=f"Non-supernodes (mean={np.mean(non_supernode_vals):.4f})", density=True)
            ax.axvline(np.mean(supernode_vals), color='darkred', linestyle='--', linewidth=2)
            ax.axvline(np.mean(non_supernode_vals), color='darkblue', linestyle='--', linewidth=2)
            ax.set_xlabel(metric_name)
            ax.set_ylabel("Density")
            ax.set_title(f"{metric_name} Distribution Comparison")
            ax.legend()

            layer_suffix = f" - {layer_name}" if layer_name else ""
            fig.suptitle(f"Supernode Analysis{layer_suffix}", fontsize=12)
            plt.tight_layout()

            if save_dir is not None:
                safe_metric = metric_name.replace("/", "_").replace(" ", "_")
                safe_layer = layer_name.replace(".", "_").replace("/", "_") if layer_name else "all"
                save_path = save_dir / f"supernode_comparison_{safe_metric}_{safe_layer}.png"
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved supernode comparison to {save_path}")

            figures.append(fig)

        return figures

    def plot_scatter_with_groups(
        self,
        x: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        group_mask: Union[torch.Tensor, np.ndarray],
        xlabel: str,
        ylabel: str,
        title: str,
        group_labels: Tuple[str, str] = ("Supernodes", "Non-supernodes"),
        save_path: Optional[Union[str, Path]] = None,
        show_regression: bool = True,
    ) -> Figure:
        """
        Scatter plot with two groups highlighted differently.

        Args:
            x: X-axis values
            y: Y-axis values
            group_mask: Boolean mask for group 1 (True) vs group 2 (False)
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            group_labels: Labels for the two groups
            save_path: Path to save figure
            show_regression: Whether to show regression lines

        Returns:
            Matplotlib figure
        """
        def _to_array(z):
            if isinstance(z, torch.Tensor):
                return z.detach().cpu().to(torch.float32).numpy()
            return np.asarray(z, dtype=np.float32)

        x_arr = _to_array(x)
        y_arr = _to_array(y)
        mask = _to_array(group_mask).astype(bool)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot non-supernodes first (background)
        ax.scatter(x_arr[~mask], y_arr[~mask], s=15, alpha=0.4, c='steelblue',
                  label=group_labels[1], edgecolors='none')
        # Plot supernodes on top
        ax.scatter(x_arr[mask], y_arr[mask], s=40, alpha=0.8, c='coral',
                  label=group_labels[0], edgecolors='darkred', linewidths=0.5)

        if show_regression:
            # Regression for all data
            finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if finite_mask.sum() > 2:
                from scipy import stats
                slope, intercept, r_value, _, _ = stats.linregress(x_arr[finite_mask], y_arr[finite_mask])
                x_line = np.linspace(x_arr[finite_mask].min(), x_arr[finite_mask].max(), 100)
                ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5,
                       label=f"r={r_value:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved grouped scatter to {save_path}")

        return fig

    # ========== Heatmaps ==========

    def plot_heatmap(
        self,
        data: Union[pd.DataFrame, np.ndarray, Dict[str, Dict[str, float]]],
        title: str = "Heatmap",
        cmap: str = "coolwarm",
        annotate: bool = True,
        fmt: str = ".3f",
        xlabel: str = None,
        ylabel: str = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create a heatmap visualization.

        Args:
            data: Data to plot (DataFrame, array, or nested dict)
            title: Plot title
            cmap: Colormap to use
            annotate: Whether to annotate cells with values
            fmt: Format string for annotations
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data

        fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.8), max(8, len(df.index) * 0.5)))

        if HAS_SEABORN:
            sns.heatmap(df, ax=ax, cmap=cmap, center=0, annot=annotate, fmt=fmt, cbar_kws={"label": "Value"}, linewidths=0.5)
        else:
            im = ax.imshow(df.values, cmap=cmap, aspect="auto")

            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_xticklabels(df.columns)
            ax.set_yticklabels(df.index)

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Value", rotation=270, labelpad=15)

            if annotate:
                for i in range(len(df.index)):
                    for j in range(len(df.columns)):
                        ax.text(j, i, format(df.iloc[i, j], fmt), ha="center", va="center", color="black")

        ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pairwise_redundancy_matrix(
        self,
        redundancy_matrix: Union[torch.Tensor, np.ndarray],
        layer_name: str,
        title: Optional[str] = None,
        cmap: str = "YlOrRd",
        annotate: bool = False,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot a pairwise redundancy matrix for neurons in a layer.

        Args:
            redundancy_matrix: Square matrix [num_neurons, num_neurons] of redundancy values
            layer_name: Name of the layer
            title: Optional custom title
            cmap: Colormap (YlOrRd works well for redundancy - higher = more redundant = warmer)
            annotate: Whether to annotate cells (only for small matrices)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        if isinstance(redundancy_matrix, torch.Tensor):
            matrix = redundancy_matrix.detach().cpu().numpy()
        else:
            matrix = np.asarray(redundancy_matrix)

        num_neurons = matrix.shape[0]
        
        # Determine figure size based on matrix size
        fig_size = max(8, min(20, num_neurons * 0.3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Only annotate for small matrices
        should_annotate = annotate and num_neurons <= 20

        if HAS_SEABORN:
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                annot=should_annotate,
                fmt=".2f" if should_annotate else None,
                cbar_kws={"label": "Redundancy (bits)"},
                square=True,
                linewidths=0.5 if num_neurons <= 50 else 0,
            )
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect="equal")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Redundancy (bits)", rotation=270, labelpad=15)

            if should_annotate:
                for i in range(num_neurons):
                    for j in range(num_neurons):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            ax.set_title(f"Pairwise Redundancy Matrix - {layer_name}\n({num_neurons} neurons)", fontsize=14, fontweight="bold")

        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")

        # Add statistics annotation
        upper_tri = matrix[np.triu_indices(num_neurons, k=1)]
        if len(upper_tri) > 0:
            stats_text = f"Mean: {np.mean(upper_tri):.3f}\nMax: {np.max(upper_tri):.3f}\nMin: {np.min(upper_tri):.3f}"
            ax.text(
                1.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_metric_correlation_scatter(
        self,
        scores_x: Union[torch.Tensor, np.ndarray, List[float]],
        scores_y: Union[torch.Tensor, np.ndarray, List[float]],
        metric_name_x: str,
        metric_name_y: str,
        layer_name: str,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot scatter plot comparing two metric scores for correlation analysis.

        Args:
            scores_x: Scores for x-axis metric
            scores_y: Scores for y-axis metric  
            metric_name_x: Name of x-axis metric
            metric_name_y: Name of y-axis metric
            layer_name: Name of the layer
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        if isinstance(scores_x, torch.Tensor):
            x = scores_x.detach().cpu().numpy().flatten()
        else:
            x = np.asarray(scores_x).flatten()

        if isinstance(scores_y, torch.Tensor):
            y = scores_y.detach().cpu().numpy().flatten()
        else:
            y = np.asarray(scores_y).flatten()

        # Ensure same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=30, edgecolors="black", linewidths=0.5)

        # Add correlation coefficient
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.05, 0.95,
                f"Pearson r = {correlation:.3f}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            # Add trend line
            if not np.isnan(correlation):
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, label="Linear fit")

        ax.set_xlabel(metric_name_x, fontsize=12)
        ax.set_ylabel(metric_name_y, fontsize=12)
        ax.set_title(f"Metric Correlation: {metric_name_x} vs {metric_name_y}\nLayer: {layer_name}", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ========== SCAR / Supernode Visualizations ==========

    def plot_scar_layer_scores(
        self,
        scar_scores: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray, List[float]]]],
        metric_name: str = "scar_loss_proxy",
        plot_type: str = "violin",
        save_path: Optional[Union[str, Path]] = None,
        show_statistics: bool = True,
    ) -> Figure:
        """
        Convenience wrapper for visualizing SCAR-style supernode metrics (e.g. scar_loss_proxy)
        across layers, using the same interface as plot_layer_scores.

        Args:
            scar_scores: Dict[layer_name -> Dict[metric_name -> scores_tensor]]
            metric_name: Which SCAR metric to visualize
                         (e.g. 'scar_loss_proxy', 'scar_activation_power', 'scar_curvature', 'scar_taylor')
            plot_type:   'violin' | 'box' | 'bar'
            save_path:   Optional path to save the figure
            show_statistics: Whether to include mean/std text box when applicable
        """
        layer_to_scores: Dict[str, Union[torch.Tensor, np.ndarray, List[float]]] = {}
        for layer_name, metrics in scar_scores.items():
            if metric_name in metrics:
                layer_to_scores[layer_name] = metrics[metric_name]

        if not layer_to_scores:
            logger.warning(f"No SCAR scores found for metric '{metric_name}'.")
            # Fallback to an empty figure
            fig, _ = plt.subplots(figsize=self.figsize)
            return fig

        return self.plot_layer_scores(
            scores=layer_to_scores,
            metric_name=metric_name,
            plot_type=plot_type,
            save_path=save_path,
            show_statistics=show_statistics,
        )

    def plot_scar_heatmap(
        self,
        scar_scores: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray, List[float]]]],
        metrics: Optional[List[str]] = None,
        title: str = "SCAR Metrics Heatmap",
        save_path: Optional[Union[str, Path]] = None,
        normalize_per_metric: bool = True,
    ) -> Figure:
        """
        Create a heatmap of mean SCAR metrics across layers.

        Args:
            scar_scores: Dict[layer_name -> Dict[metric_name -> scores_tensor]]
            metrics:     List of metric names to include (default: all present keys)
            title:       Plot title
            save_path:   Optional path to save the figure
            normalize_per_metric: If True, normalize each metric column to [0, 1] for visualization
                                  (important since SCAR metrics have vastly different scales)
        """
        # Determine metrics to include
        all_metric_names = sorted({m for layer_scores in scar_scores.values() for m in layer_scores.keys()})
        if metrics is None:
            metrics = all_metric_names
        else:
            metrics = [m for m in metrics if m in all_metric_names]

        if not metrics:
            logger.warning("plot_scar_heatmap: no SCAR metrics to plot.")
            fig, _ = plt.subplots(figsize=self.figsize)
            return fig

        # Build nested dict layer -> metric -> mean value
        layer_metric_means: Dict[str, Dict[str, float]] = {}
        for layer_name, metric_dict in scar_scores.items():
            layer_metric_means[layer_name] = {}
            for metric_name in metrics:
                vals = metric_dict.get(metric_name, None)
                if vals is None:
                    continue
                if isinstance(vals, torch.Tensor):
                    arr = vals.detach().cpu().numpy()
                elif isinstance(vals, np.ndarray):
                    arr = vals
                else:
                    arr = np.asarray(vals)
                if arr.size == 0:
                    continue
                layer_metric_means[layer_name][metric_name] = float(np.mean(arr))

        if not layer_metric_means:
            logger.warning("plot_scar_heatmap: no non-empty SCAR metrics to plot.")
            fig, _ = plt.subplots(figsize=self.figsize)
            return fig

        # Convert to DataFrame
        df = pd.DataFrame(layer_metric_means).T  # layers as rows, metrics as columns
        
        # Store original values for annotations
        df_original = df.copy()
        
        # Normalize per metric column for visualization (SCAR metrics have vastly different scales)
        if normalize_per_metric and len(df) > 1:
            for col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max - col_min > 1e-12:
                    df[col] = (df[col] - col_min) / (col_max - col_min)
                else:
                    df[col] = 0.5  # All values same, set to middle
        
        fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 2), max(8, len(df.index) * 0.4)))
        
        if HAS_SEABORN:
            # Use normalized values for coloring, but show original values as annotations
            # Don't use center=0 since SCAR values are all positive
            sns.heatmap(
                df, ax=ax, cmap="viridis", 
                annot=df_original.applymap(lambda x: f"{x:.2e}"),  # Scientific notation for small values
                fmt="",  # Empty fmt since we're passing formatted strings
                cbar_kws={"label": "Normalized Value" if normalize_per_metric else "Value"}, 
                linewidths=0.5,
                vmin=0, vmax=1 if normalize_per_metric else None,
            )
        else:
            im = ax.imshow(df.values, cmap="viridis", aspect="auto", vmin=0, vmax=1 if normalize_per_metric else None)
            
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_xticklabels(df.columns)
            ax.set_yticklabels(df.index)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Normalized Value" if normalize_per_metric else "Value", rotation=270, labelpad=15)
            
            # Annotate with original values in scientific notation
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    val = df_original.iloc[i, j]
                    ax.text(j, i, f"{val:.2e}", ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Layer")
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig

    # ========== Pruning Analysis ==========

    def plot_pruning_performance(
        self,
        results: Dict[str, Dict[float, Dict[str, float]]],
        metrics: List[str] = ["accuracy", "loss"],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show_confidence: bool = True,
    ) -> Figure:
        """
        Plot performance metrics for multiple pruning strategies.

        Args:
            results: Nested dict of strategy -> sparsity -> metric -> value
            metrics: List of metrics to plot
            save_path: Path to save the plot
            title: Overall title for the plot
            show_confidence: Whether to show confidence intervals

        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * num_metrics // 2))

        gs = GridSpec(num_metrics, 1, figure=fig, hspace=0.3)

        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx, 0])

            for strategy, strategy_results in results.items():
                sparsities = sorted(strategy_results.keys())

                # Extract values
                means = []
                stds = []

                for sparsity in sparsities:
                    if isinstance(strategy_results[sparsity], dict):
                        if "mean" in strategy_results[sparsity]:
                            means.append(strategy_results[sparsity]["mean"].get(metric, 0))
                            if "std" in strategy_results[sparsity] and show_confidence:
                                stds.append(strategy_results[sparsity]["std"].get(metric, 0))
                        else:
                            means.append(strategy_results[sparsity].get(metric, 0))
                    else:
                        means.append(strategy_results[sparsity])

                # Plot
                color = self.strategy_colors.get(strategy, self.colors[0])
                line = ax.plot(sparsities, means, "o-", label=strategy, linewidth=2.5, markersize=8, color=color)

                if stds and show_confidence:
                    means = np.array(means)
                    stds = np.array(stds)
                    ax.fill_between(sparsities, means - stds, means + stds, alpha=0.2, color=line[0].get_color())

            ax.set_xlabel("Sparsity Level", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f"{metric.capitalize()} vs Sparsity", fontsize=12)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_pruning_before_after(
        self,
        sparsities: List[float],
        before_accuracies: Dict[str, List[float]],
        after_accuracies: Dict[str, List[float]],
        before_std: Optional[Dict[str, List[float]]] = None,
        after_std: Optional[Dict[str, List[float]]] = None,
        algorithm: str = "Pruning",
        save_dir: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        total_params: Optional[int] = None,
    ) -> List[Figure]:
        """
        Create before/after fine-tuning comparison plots for pruning experiments.

        Args:
            sparsities: List of sparsity levels (0.0 to 1.0).
            before_accuracies: Dict mapping selection mode to accuracy list before fine-tuning.
            after_accuracies: Dict mapping selection mode to accuracy list after fine-tuning.
            before_std: Optional dict of standard deviations for before accuracies.
            after_std: Optional dict of standard deviations for after accuracies.
            algorithm: Name of the pruning algorithm.
            save_dir: Directory to save figures.
            dpi: DPI for saved figures.
            total_params: Total number of parameters in the model (for secondary x-axis).

        Returns:
            List of generated figures.
        """
        figures = []
        x_values = [s * 100 for s in sparsities]
        
        def format_params(n: int) -> str:
            """Format parameter count with K/M suffix."""
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.0f}K"
            else:
                return str(n)
        
        def add_param_axis(ax, fig, x_vals, total_params):
            """Add secondary x-axis showing remaining parameters."""
            if total_params is None:
                return
            # Make room at top for secondary axis
            fig.subplots_adjust(top=0.85)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            # Use fewer ticks to avoid clutter
            tick_positions = [0, 20, 40, 60, 80, 100]
            param_labels = [format_params(int(total_params * (1 - t/100))) for t in tick_positions]
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(param_labels, fontsize=9)
            ax2.set_xlabel("Remaining Parameters", fontsize=10)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Multiple selection modes: create separate before/after plots
        if len(before_accuracies) > 1:
            # Before fine-tuning plot
            fig_before, ax_before = plt.subplots(figsize=self.figsize)
            for mode, accuracies in before_accuracies.items():
                if before_std and mode in before_std:
                    ax_before.errorbar(
                        x_values, accuracies, yerr=before_std[mode],
                        fmt="o-", label=f"{mode} mode", linewidth=2.5, markersize=8,
                        capsize=5, capthick=2
                    )
                else:
                    ax_before.plot(x_values, accuracies, "o-", label=f"{mode} mode",
                                   linewidth=2.5, markersize=8)

            ax_before.set_xlabel("Pruning %", fontsize=12)
            ax_before.set_ylabel("Accuracy (%)", fontsize=12)
            ax_before.set_title(f"{algorithm} Pruning - Before Fine-tuning",
                               fontsize=14, fontweight="bold")
            ax_before.grid(True, alpha=0.3)
            ax_before.legend(loc="best")
            ax_before.set_xlim(0, 100)
            ax_before.set_ylim(0, 105)
            add_param_axis(ax_before, fig_before, x_values, total_params)

            if save_dir:
                # Use lowercase underscore naming for consistency with LLM format
                algo_safe = algorithm.lower().replace(" ", "_")
                fig_before.savefig(save_dir / f"pruning_{algo_safe}_accuracy_before.png",
                                   dpi=dpi, bbox_inches="tight")
            figures.append(fig_before)

            # After fine-tuning plot
            fig_after, ax_after = plt.subplots(figsize=self.figsize)
            for mode, accuracies in after_accuracies.items():
                if after_std and mode in after_std:
                    ax_after.errorbar(
                        x_values, accuracies, yerr=after_std[mode],
                        fmt="o-", label=f"{mode} mode", linewidth=2.5, markersize=8,
                        capsize=5, capthick=2
                    )
                else:
                    ax_after.plot(x_values, accuracies, "o-", label=f"{mode} mode",
                                  linewidth=2.5, markersize=8)

            ax_after.set_xlabel("Pruning %", fontsize=12)
            ax_after.set_ylabel("Accuracy (%)", fontsize=12)
            ax_after.set_title(f"{algorithm} Pruning - After Fine-tuning",
                              fontsize=14, fontweight="bold")
            ax_after.grid(True, alpha=0.3)
            ax_after.legend(loc="best")
            ax_after.set_xlim(0, 100)
            ax_after.set_ylim(0, 105)
            add_param_axis(ax_after, fig_after, x_values, total_params)

            if save_dir:
                # Use lowercase underscore naming for consistency with LLM format
                algo_safe = algorithm.lower().replace(" ", "_")
                fig_after.savefig(save_dir / f"pruning_{algo_safe}_accuracy_after.png",
                                  dpi=dpi, bbox_inches="tight")
            figures.append(fig_after)

        else:
            # Single selection mode: combined before/after plot
            selection_mode = list(before_accuracies.keys())[0]
            fig, ax = plt.subplots(figsize=self.figsize)

            # Before fine-tuning
            if before_std and selection_mode in before_std:
                ax.errorbar(
                    x_values, before_accuracies[selection_mode],
                    yerr=before_std[selection_mode],
                    fmt="o-", label="Before Fine-tuning", color="#FF6B6B",
                    linewidth=2.5, markersize=8, capsize=5, capthick=2
                )
            else:
                ax.plot(x_values, before_accuracies[selection_mode], "o-",
                        label="Before Fine-tuning", color="#FF6B6B",
                        linewidth=2.5, markersize=8)

            # After fine-tuning
            if after_std and selection_mode in after_std:
                ax.errorbar(
                    x_values, after_accuracies[selection_mode],
                    yerr=after_std[selection_mode],
                    fmt="o-", label="After Fine-tuning", color="#4ECDC4",
                    linewidth=2.5, markersize=8, capsize=5, capthick=2
                )
            else:
                ax.plot(x_values, after_accuracies[selection_mode], "o-",
                        label="After Fine-tuning", color="#4ECDC4",
                        linewidth=2.5, markersize=8)

            ax.set_xlabel("Pruning %", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title(f"{algorithm} Pruning ({selection_mode} mode)",
                        fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 105)
            add_param_axis(ax, fig, x_values, total_params)

            if save_dir:
                # Use lowercase underscore naming for consistency with LLM format
                algo_safe = algorithm.lower().replace(" ", "_")
                fig.savefig(save_dir / f"pruning_{algo_safe}_accuracy.png",
                            dpi=dpi, bbox_inches="tight")
            figures.append(fig)

        return figures

    def plot_pruning_loss_before_after(
        self,
        sparsities: List[float],
        before_losses: Dict[str, List[float]],
        after_losses: Dict[str, List[float]],
        before_std: Optional[Dict[str, List[float]]] = None,
        after_std: Optional[Dict[str, List[float]]] = None,
        algorithm: str = "Pruning",
        save_dir: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        total_params: Optional[int] = None,
    ) -> List[Figure]:
        """
        Create before/after fine-tuning comparison plots for pruning experiments (loss metric).

        Args:
            sparsities: List of sparsity levels (0.0 to 1.0).
            before_losses: Dict mapping selection mode to loss list before fine-tuning.
            after_losses: Dict mapping selection mode to loss list after fine-tuning.
            before_std: Optional dict of standard deviations for before losses.
            after_std: Optional dict of standard deviations for after losses.
            algorithm: Name of the pruning algorithm.
            save_dir: Directory to save figures.
            dpi: DPI for saved figures.
            total_params: Total number of parameters in the model (for secondary x-axis).

        Returns:
            List of generated figures.
        """
        figures = []
        x_values = [s * 100 for s in sparsities]
        
        def format_params(n: int) -> str:
            """Format parameter count with K/M suffix."""
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.0f}K"
            else:
                return str(n)
        
        def add_param_axis(ax, fig, x_vals, total_params):
            """Add secondary x-axis showing remaining parameters."""
            if total_params is None:
                return
            # Make room at top for secondary axis
            fig.subplots_adjust(top=0.85)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            # Use fewer ticks to avoid clutter
            tick_positions = [0, 20, 40, 60, 80, 100]
            param_labels = [format_params(int(total_params * (1 - t/100))) for t in tick_positions]
            ax2.set_xticks(tick_positions)
            ax2.set_xticklabels(param_labels, fontsize=9)
            ax2.set_xlabel("Remaining Parameters", fontsize=10)

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Multiple selection modes: create separate before/after plots
        if len(before_losses) > 1:
            # Before fine-tuning plot
            fig_before, ax_before = plt.subplots(figsize=self.figsize)
            for mode, losses in before_losses.items():
                if before_std and mode in before_std:
                    ax_before.errorbar(
                        x_values, losses, yerr=before_std[mode],
                        fmt="o-", label=f"{mode} mode", linewidth=2.5, markersize=8,
                        capsize=5, capthick=2
                    )
                else:
                    ax_before.plot(x_values, losses, "o-", label=f"{mode} mode",
                                   linewidth=2.5, markersize=8)

            ax_before.set_xlabel("Pruning %", fontsize=12)
            ax_before.set_ylabel("Loss", fontsize=12)
            ax_before.set_title(f"{algorithm} Pruning - Before Fine-tuning (Loss)",
                               fontsize=14, fontweight="bold")
            ax_before.grid(True, alpha=0.3)
            ax_before.legend(loc="best")
            ax_before.set_xlim(0, 100)
            add_param_axis(ax_before, fig_before, x_values, total_params)

            if save_dir:
                algo_safe = algorithm.lower().replace(" ", "_")
                fig_before.savefig(save_dir / f"pruning_{algo_safe}_loss_before.png",
                                   dpi=dpi, bbox_inches="tight")
            figures.append(fig_before)

            # After fine-tuning plot
            fig_after, ax_after = plt.subplots(figsize=self.figsize)
            for mode, losses in after_losses.items():
                if after_std and mode in after_std:
                    ax_after.errorbar(
                        x_values, losses, yerr=after_std[mode],
                        fmt="o-", label=f"{mode} mode", linewidth=2.5, markersize=8,
                        capsize=5, capthick=2
                    )
                else:
                    ax_after.plot(x_values, losses, "o-", label=f"{mode} mode",
                                  linewidth=2.5, markersize=8)

            ax_after.set_xlabel("Pruning %", fontsize=12)
            ax_after.set_ylabel("Loss", fontsize=12)
            ax_after.set_title(f"{algorithm} Pruning - After Fine-tuning (Loss)",
                              fontsize=14, fontweight="bold")
            ax_after.grid(True, alpha=0.3)
            ax_after.legend(loc="best")
            ax_after.set_xlim(0, 100)
            add_param_axis(ax_after, fig_after, x_values, total_params)

            if save_dir:
                algo_safe = algorithm.lower().replace(" ", "_")
                fig_after.savefig(save_dir / f"pruning_{algo_safe}_loss_after.png",
                                  dpi=dpi, bbox_inches="tight")
            figures.append(fig_after)

        else:
            # Single selection mode: combined before/after plot
            selection_mode = list(before_losses.keys())[0]
            fig, ax = plt.subplots(figsize=self.figsize)

            # Before fine-tuning
            if before_std and selection_mode in before_std:
                ax.errorbar(
                    x_values, before_losses[selection_mode],
                    yerr=before_std[selection_mode],
                    fmt="o-", label="Before Fine-tuning", color="#FF6B6B",
                    linewidth=2.5, markersize=8, capsize=5, capthick=2
                )
            else:
                ax.plot(x_values, before_losses[selection_mode], "o-",
                        label="Before Fine-tuning", color="#FF6B6B",
                        linewidth=2.5, markersize=8)

            # After fine-tuning
            if after_std and selection_mode in after_std:
                ax.errorbar(
                    x_values, after_losses[selection_mode],
                    yerr=after_std[selection_mode],
                    fmt="o-", label="After Fine-tuning", color="#4ECDC4",
                    linewidth=2.5, markersize=8, capsize=5, capthick=2
                )
            else:
                ax.plot(x_values, after_losses[selection_mode], "o-",
                        label="After Fine-tuning", color="#4ECDC4",
                        linewidth=2.5, markersize=8)

            ax.set_xlabel("Pruning %", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"{algorithm} Pruning ({selection_mode} mode) - Loss",
                        fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)
            ax.set_xlim(0, 100)
            add_param_axis(ax, fig, x_values, total_params)

            if save_dir:
                algo_safe = algorithm.lower().replace(" ", "_")
                fig.savefig(save_dir / f"pruning_{algo_safe}_loss.png",
                            dpi=dpi, bbox_inches="tight")
            figures.append(fig)

        return figures

    def plot_pruning_improvement(
        self,
        sparsities: List[float],
        before_accuracies: List[float],
        after_accuracies: List[float],
        algorithm: str = "Pruning",
        selection_mode: str = "",
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
    ) -> Figure:
        """
        Create a bar chart showing accuracy improvement from fine-tuning.

        Args:
            sparsities: List of sparsity levels.
            before_accuracies: Accuracies before fine-tuning.
            after_accuracies: Accuracies after fine-tuning.
            algorithm: Pruning algorithm name.
            selection_mode: Selection mode (low, high, random).
            save_path: Path to save the figure.
            dpi: DPI for saved figure.

        Returns:
            Matplotlib figure.
        """
        improvements = [after - before for before, after in zip(before_accuracies, after_accuracies)]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(
            range(len(improvements)),
            improvements,
            tick_label=[f"{s:.0%}" for s in sparsities],
            color=["#4ECDC4" if imp >= 0 else "#FF6B6B" for imp in improvements],
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{imp:+.1f}%",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("Sparsity Level", fontsize=12)
        ax.set_ylabel("Accuracy Improvement (%)", fontsize=12)
        mode_str = f" ({selection_mode} mode)" if selection_mode else ""
        ax.set_title(f"{algorithm} Pruning{mode_str}: Fine-tuning Improvement",
                    fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    # ========== Comparison Plots ==========

    def plot_radar_chart(
        self, data: Dict[str, Dict[str, float]], title: str = "Multi-Metric Comparison", save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Create a radar chart comparing multiple metrics.

        Args:
            data: Nested dict: {series_name: {metric_name: value}}
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        series = list(data.keys())
        metrics = list(next(iter(data.values())).keys())
        num_vars = len(metrics)

        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

        for i, series_name in enumerate(series):
            values = [data[series_name][metric] for metric in metrics]
            values += values[:1]

            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, "o-", linewidth=2, label=series_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    # ========== Comprehensive Reports ==========

    def create_comprehensive_report(self, results: Dict[str, Any], output_dir: Union[str, Path], experiment_name: str = "experiment"):
        """
        Create a comprehensive visual report with multiple plots.

        Args:
            results: Dictionary containing all results
            output_dir: Directory to save the report
            experiment_name: Name of the experiment
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Generate various plots based on available data

        # 1. Metric evolution
        if "metrics_over_time" in results:
            for metric_name, values in results["metrics_over_time"].items():
                self.plot_metric_evolution(
                    results.get("steps", list(range(len(next(iter(values.values())))))),
                    values,
                    title=f"{metric_name} Evolution",
                    ylabel=metric_name,
                    save_path=plots_dir / f"{metric_name}_evolution.png",
                )

        # 2. Layer scores
        if "layer_scores" in results:
            for metric_name, scores in results["layer_scores"].items():
                self.plot_layer_scores(scores, metric_name, save_path=plots_dir / f"{metric_name}_layers.png")

        # 3. Heatmaps
        if "heatmap_data" in results:
            self.plot_heatmap(results["heatmap_data"], title="Metrics Heatmap", save_path=plots_dir / "metrics_heatmap.png")

        # 4. Pruning results
        if "pruning_results" in results:
            self.plot_pruning_performance(results["pruning_results"], save_path=plots_dir / "pruning_performance.png")

        # 5. Comparisons
        if "comparison_data" in results:
            self.plot_radar_chart(results["comparison_data"], save_path=plots_dir / "comparison_radar.png")

        # Create summary statistics
        self._create_summary_statistics(results, output_dir)

        # Create README
        self._create_readme(experiment_name, results, output_dir)

        logger.info(f"Comprehensive report saved to {output_dir}")

    def _create_summary_statistics(self, results: Dict[str, Any], output_dir: Path):
        """Create summary statistics CSV."""
        summary = []

        if "layer_scores" in results:
            for metric_name, layer_scores in results["layer_scores"].items():
                for layer_name, scores in layer_scores.items():
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    elif not isinstance(scores, np.ndarray):
                        scores = np.array(scores)

                    summary.append(
                        {
                            "Metric": metric_name,
                            "Layer": layer_name,
                            "Mean": np.mean(scores),
                            "Std": np.std(scores),
                            "Min": np.min(scores),
                            "Max": np.max(scores),
                            "Count": len(scores),
                        }
                    )

        if summary:
            df = pd.DataFrame(summary)
            df.to_csv(output_dir / "summary_statistics.csv", index=False)

    def _create_readme(self, experiment_name: str, results: Dict[str, Any], output_dir: Path):
        """Create README file for the report."""
        readme_content = f"""# {experiment_name} Report

Generated visualization report for alignment analysis.

## Report Generation
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Experiment: {experiment_name}

## Contents

### Plots Directory
- `*_evolution.png`: Metric evolution over time
- `*_layers.png`: Layer-wise metric distributions
- `metrics_heatmap.png`: Heatmap of all metrics
- `pruning_performance.png`: Pruning experiment results
- `comparison_radar.png`: Multi-metric comparison

### Data Files
- `summary_statistics.csv`: Summary statistics for all metrics

## Experiment Summary
"""

        # Add summary statistics
        if "layer_scores" in results:
            readme_content += f"- Number of metrics: {len(results['layer_scores'])}\n"
            all_layers = set()
            for scores in results["layer_scores"].values():
                all_layers.update(scores.keys())
            readme_content += f"- Number of layers: {len(all_layers)}\n"

        if "pruning_results" in results:
            readme_content += f"- Pruning strategies: {', '.join(results['pruning_results'].keys())}\n"

        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)


    def plot_distribution(
        self,
        data: Union[Dict[str, List[float]], np.ndarray, torch.Tensor],
        title: str = "Metric Distribution",
        xlabel: str = "Value",
        ylabel: str = "Density",
        bins: int = 50,
        show_kde: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot distribution of metric values with optional KDE overlay.

        Args:
            data: Either a dict mapping series names to value lists, or a single array/tensor.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            bins: Number of histogram bins.
            show_kde: Whether to overlay a KDE curve.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Normalize input to dict format
        if isinstance(data, (np.ndarray, torch.Tensor, list)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            elif isinstance(data, list):
                data = np.asarray(data)
            data = {"values": data.flatten()}

        for i, (name, values) in enumerate(data.items()):
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            elif isinstance(values, list):
                values = np.asarray(values)
            values = values.flatten()

            # Remove non-finite values
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            color = self.colors[i % len(self.colors)]
            ax.hist(values, bins=bins, alpha=0.6, label=name, color=color, density=True)

            # Add KDE
            if show_kde and values.size > 1:
                try:
                    from scipy.stats import gaussian_kde

                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    ax.plot(x_range, kde(x_range), color=color, linewidth=2)
                except Exception:
                    pass  # Skip KDE if it fails

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if len(data) > 1:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_sparsity_performance(
        self,
        sparsities: List[float],
        perplexities: List[float],
        strategy_name: str = "Alignment",
        baseline_ppl: Optional[float] = None,
        title: str = "Sparsity vs Perplexity",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot perplexity vs sparsity curve for pruning evaluation.

        Args:
            sparsities: List of sparsity levels (0.0 to 1.0).
            perplexities: Corresponding perplexity values.
            strategy_name: Name of the pruning strategy.
            baseline_ppl: Optional baseline perplexity (unpruned model).
            title: Plot title.
            save_path: Optional path to save the figure.

        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(sparsities, perplexities, "o-", linewidth=2.5, markersize=8, label=strategy_name)

        if baseline_ppl is not None:
            ax.axhline(y=baseline_ppl, color="gray", linestyle="--", linewidth=1.5, label="Baseline")

        ax.set_xlabel("Sparsity", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_llm_pruning_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        baseline_ppl: Optional[float] = None,
        baseline_values: Optional[Dict[str, float]] = None,
        metric: str = "perplexity",
        title: str = "LLM Pruning Comparison",
        save_path: Optional[Union[str, Path]] = None,
        total_params: Optional[int] = None,
    ) -> Figure:
        """
        Plot pruning comparison for LLMs with configurable evaluation metric.
        All strategies (metric_low, metric_high, metric_random) are plotted together.
        
        Args:
            results: Dict mapping strategy names to their results.
                Each strategy should have:
                - 'sparsities': List of sparsity levels
                - 'perplexities' or other metric: Values at each sparsity
            baseline_ppl: Optional baseline perplexity (for backward compatibility)
            baseline_values: Optional dict of baseline values for each metric
            metric: Which metric to plot ('perplexity', 'accuracy_hellaswag', etc.)
            title: Plot title
            save_path: Path to save the figure
            total_params: Total number of parameters for secondary x-axis
            
        Returns:
            Matplotlib figure
        """
        # Metric configuration: name, ylabel, lower_is_better
        metric_config = {
            "perplexity": ("Perplexity", True),
            "loss": ("Cross-Entropy Loss", True),
            "bits_per_byte": ("Bits per Byte", True),
            "normalized_perplexity": ("Normalized Score", False),
            "accuracy_hellaswag": ("HellaSwag Accuracy (%)", False),
            "accuracy_arc_easy": ("ARC-Easy Accuracy (%)", False),
            "accuracy_arc_challenge": ("ARC-Challenge Accuracy (%)", False),  # NVIDIA Minitron
            "accuracy_piqa": ("PIQA Accuracy (%)", False),
            "accuracy_boolq": ("BoolQ Accuracy (%)", False),
            "accuracy_winogrande": ("WinoGrande Accuracy (%)", False),
            "accuracy_truthfulqa": ("TruthfulQA Accuracy (%)", False),  # NVIDIA Minitron
            "accuracy_mmlu": ("MMLU Accuracy (%)", False),
            "accuracy_gsm8k": ("GSM8k Math Accuracy (%)", False),  # NVIDIA Minitron
            "accuracy_mbpp": ("MBPP Code Accuracy (%)", False),  # NVIDIA Minitron
            "accuracy_humaneval": ("HumanEval Code Accuracy (%)", False),  # NVIDIA Minitron
        }
        
        ylabel, lower_is_better = metric_config.get(metric, (metric.replace("_", " ").title(), True))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Helper function to format parameter counts
        def format_params(n: int) -> str:
            if n >= 1_000_000_000:
                return f"{n/1_000_000_000:.1f}B"
            elif n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)
        
        # Define markers for different modes (shape distinguishes mode)
        mode_markers = {
            "low": {"marker": "o", "linestyle": "-"},      # Circle, solid
            "high": {"marker": "s", "linestyle": "-"},     # Square, solid
            "random": {"marker": "^", "linestyle": "--"},  # Triangle, dashed
        }
        
        # Define colors for different algorithms (color distinguishes algorithm)
        algo_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]
        
        # Group by algorithm (metric)
        algorithms = {}
        for strategy_name, data in results.items():
            sparsities = data.get("sparsities", [])
            # Try to get the requested metric, fall back to perplexities
            metric_key = metric if metric in data else "perplexities"
            values = data.get(metric_key, data.get("perplexities", []))
            
            if not sparsities or not values:
                continue
            
            # Parse algorithm and mode from strategy name (e.g., "rayleigh_quotient_low")
            parts = strategy_name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ["low", "high", "random"]:
                algorithm = parts[0]
                mode = parts[1]
            else:
                algorithm = strategy_name
                mode = "low"
            
            if algorithm not in algorithms:
                algorithms[algorithm] = {}
            algorithms[algorithm][mode] = {
                "sparsities": sparsities,
                "values": values,
            }
        
        # Plot each algorithm with its modes
        for algo_idx, (algorithm, modes) in enumerate(algorithms.items()):
            algo_display = algorithm.replace("_", " ").title()
            algo_color = algo_colors[algo_idx % len(algo_colors)]
            
            for mode, data in modes.items():
                mode_style = mode_markers.get(mode, {"marker": "o", "linestyle": "-"})
                label = f"{algo_display} ({mode})"
                
                ax.plot(
                    data["sparsities"],
                    data["values"],
                    marker=mode_style["marker"],
                    linestyle=mode_style["linestyle"],
                    color=algo_color,  # Color by algorithm
                    linewidth=2,
                    markersize=8,
                    label=label,
                    alpha=0.8,
                )
        
        # Plot baseline
        baseline = baseline_ppl if metric == "perplexity" else (baseline_values or {}).get(metric)
        if baseline is not None:
            ax.axhline(y=baseline, color="black", linestyle=":", linewidth=2, label=f"Baseline ({baseline:.2f})")
        
        ax.set_xlabel("Sparsity (% Pruned)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left" if lower_is_better else "lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for y-axis if range is large (only for perplexity-like metrics)
        if lower_is_better and metric in ["perplexity", "bits_per_byte", "loss"]:
            all_vals = [v for data in results.values() for v in data.get(metric, data.get("perplexities", []))]
            if all_vals and max(all_vals) / (min(all_vals) + 1e-6) > 10:
                ax.set_yscale("log")
        
        # Add secondary x-axis for parameter count
        if total_params is not None and total_params > 0:
            # Get the sparsity values from the x-axis
            xlim = ax.get_xlim()
            
            # Create secondary axis at the top
            ax2 = ax.twiny()
            ax2.set_xlim(xlim)
            
            # Calculate tick positions (use same as primary axis or custom)
            sparsity_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Filter to those in xlim range
            sparsity_ticks = [s for s in sparsity_ticks if xlim[0] <= s <= xlim[1]]
            
            ax2.set_xticks(sparsity_ticks)
            param_labels = [format_params(int(total_params * (1 - s))) for s in sparsity_ticks]
            ax2.set_xticklabels(param_labels, fontsize=9)
            ax2.set_xlabel("Remaining Parameters", fontsize=11)
            
            # Adjust layout to make room for top axis
            fig.subplots_adjust(top=0.85)
        else:
            plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved LLM pruning comparison ({metric}) to {save_path}")
        
        return fig

    def plot_pruning_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "accuracy",
        baseline_value: Optional[float] = None,
        title: str = "Pruning Strategy Comparison",
        save_path: Optional[Union[str, Path]] = None,
        total_params: Optional[int] = None,
    ) -> Figure:
        """
        Create a publication-quality pruning comparison plot with error bars.
        
        Args:
            results: Dict mapping strategy names to their results.
                Each strategy should have:
                - 'sparsities' or 'pruning_amounts': List of sparsity levels
                - 'accuracies_mean' or 'accuracies': Mean accuracy values
                - 'accuracies_std' (optional): Standard deviation for error bars
            metric: Metric to plot ('accuracy' or 'loss')
            baseline_value: Optional baseline value to show as horizontal line
            title: Plot title
            save_path: Path to save the figure
            total_params: Total number of parameters for secondary x-axis
            
        Returns:
            Matplotlib figure
        """
        def format_params(n: int) -> str:
            """Format parameter count with K/M suffix."""
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.0f}K"
            else:
                return str(n)
        # Color palette
        colors = {
            'magnitude_low': '#1f77b4',      # Blue
            'magnitude_high': '#aec7e8',     # Light blue
            'rayleigh_quotient_low': '#2ca02c',   # Green
            'rayleigh_quotient_high': '#98df8a',  # Light green
            'activation_l2_norm_low': '#ff7f0e', # Orange
            'activation_l2_norm_high': '#ffbb78', # Light orange
            'mutual_information_gaussian_low': '#9467bd',  # Purple
            'mutual_information_gaussian_high': '#c5b0d5', # Light purple
            'pairwise_redundancy_gaussian_low': '#d62728', # Red
            'pairwise_redundancy_gaussian_high': '#ff9896', # Light red
            'random_low': '#7f7f7f',         # Gray
            'random_high': '#c7c7c7',        # Light gray
            'random_random': '#bcbd22',      # Olive
        }
        
        markers = {
            'magnitude': 'o',
            'rayleigh_quotient': 's',
            'activation_l2_norm': '^',
            'mutual_information_gaussian': 'D',
            'pairwise_redundancy_gaussian': 'v',
            'random': 'x',
        }
        
        linestyles = {
            'low': '-',
            'high': '--',
            'random': ':',
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for strategy_key, strategy_data in results.items():
            # Parse strategy name and mode
            parts = strategy_key.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ['low', 'high', 'random']:
                strategy_name = parts[0]
                mode = parts[1]
            else:
                strategy_name = strategy_key
                mode = 'low'
            
            # Get data - prefer pruning_amounts (target) over sparsities (actual) for x-axis
            # This ensures plots respect the configured sparsity_levels
            sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
            
            if metric == 'accuracy':
                means = strategy_data.get('accuracies_mean', strategy_data.get('accuracies_after_finetune', 
                        strategy_data.get('accuracies_before_finetune', [])))
                stds = strategy_data.get('accuracies_std', None)
            else:
                means = strategy_data.get('losses_mean', strategy_data.get('losses_after_finetune',
                        strategy_data.get('losses_before_finetune', [])))
                stds = strategy_data.get('losses_std', None)
            
            if not sparsities or not means:
                continue
            
            # Convert to percentages for x-axis
            x_values = [s * 100 for s in sparsities]
            
            # Get styling
            color = colors.get(strategy_key, colors.get(f"{strategy_name}_{mode}", '#333333'))
            marker = markers.get(strategy_name, 'o')
            linestyle = linestyles.get(mode, '-')
            
            # Create label
            label = f"{strategy_name.replace('_', ' ').title()} ({mode})"
            
            # Plot with or without error bars
            if stds is not None and len(stds) == len(means):
                ax.errorbar(
                    x_values, means, yerr=stds,
                    fmt=f'{marker}{linestyle}',
                    label=label,
                    color=color,
                    linewidth=2.5,
                    markersize=8,
                    capsize=4,
                    capthick=2,
                    elinewidth=1.5,
                    alpha=0.9
                )
            else:
                ax.plot(
                    x_values, means,
                    f'{marker}{linestyle}',
                    label=label,
                    color=color,
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.9
                )
        
        # Add baseline
        if baseline_value is not None:
            ax.axhline(y=baseline_value, color='black', linestyle='-.', 
                      linewidth=2, label='Baseline (unpruned)', alpha=0.7)
        
        # Styling
        ax.set_xlabel('Sparsity (%)', fontsize=14, fontweight='bold')
        ylabel = 'Accuracy (%)' if metric == 'accuracy' else 'Loss'
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9, 
                 ncol=2 if len(results) > 6 else 1)
        
        # Axis limits
        ax.set_xlim(0, 100)
        if metric == 'accuracy':
            ax.set_ylim(0, max(105, ax.get_ylim()[1]))
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add secondary x-axis with parameter count
        if total_params is not None:
            fig.subplots_adjust(top=0.85)  # Make room for secondary axis
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            # Calculate remaining params at key sparsity levels
            param_ticks = [0, 20, 40, 60, 80, 100]
            param_labels = [format_params(int(total_params * (1 - t/100))) for t in param_ticks]
            ax2.set_xticks(param_ticks)
            ax2.set_xticklabels(param_labels, fontsize=10)
            ax2.set_xlabel("Remaining Parameters", fontsize=12, fontweight='bold')
        else:
            plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig

    def plot_pruning_summary_grid(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Create a 2x2 grid of pruning analysis plots.
        
        Args:
            results: Dict mapping strategy names to their results
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure with 4 subplots
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # Collect all data
        all_sparsities = set()
        strategies = list(results.keys())
        
        for strategy_data in results.values():
            sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
            all_sparsities.update(sparsities)
        
        all_sparsities = sorted(all_sparsities)
        
        # Plot 1: Accuracy before fine-tuning
        ax1 = fig.add_subplot(gs[0, 0])
        for strategy_key, strategy_data in results.items():
            sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
            accs = strategy_data.get('accuracies_before_finetune', [])
            if sparsities and accs:
                ax1.plot([s*100 for s in sparsities], accs, 'o-', 
                        label=strategy_key.replace('_', ' ').title(), 
                        linewidth=2, markersize=6)
        ax1.set_xlabel('Sparsity (%)', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('Before Fine-tuning', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='best')
        
        # Plot 2: Accuracy after fine-tuning
        ax2 = fig.add_subplot(gs[0, 1])
        for strategy_key, strategy_data in results.items():
            sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
            accs = strategy_data.get('accuracies_after_finetune', [])
            if sparsities and accs:
                ax2.plot([s*100 for s in sparsities], accs, 'o-',
                        label=strategy_key.replace('_', ' ').title(),
                        linewidth=2, markersize=6)
        ax2.set_xlabel('Sparsity (%)', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('After Fine-tuning', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, loc='best')
        
        # Plot 3: Improvement from fine-tuning
        ax3 = fig.add_subplot(gs[1, 0])
        for strategy_key, strategy_data in results.items():
            sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
            before = strategy_data.get('accuracies_before_finetune', [])
            after = strategy_data.get('accuracies_after_finetune', [])
            if sparsities and before and after and len(before) == len(after):
                improvement = [a - b for a, b in zip(after, before)]
                ax3.bar([s*100 + strategies.index(strategy_key)*2 for s in sparsities], 
                       improvement, width=2, 
                       label=strategy_key.replace('_', ' ').title(), alpha=0.8)
        ax3.set_xlabel('Sparsity (%)', fontsize=11)
        ax3.set_ylabel('Accuracy Improvement (%)', fontsize=11)
        ax3.set_title('Fine-tuning Improvement', fontsize=13, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=8, loc='best')
        
        # Plot 4: Strategy comparison at specific sparsity
        ax4 = fig.add_subplot(gs[1, 1])
        if all_sparsities:
            # Pick middle sparsity
            target_sparsity = all_sparsities[len(all_sparsities)//2]
            strategy_names = []
            before_vals = []
            after_vals = []
            
            for strategy_key, strategy_data in results.items():
                sparsities = strategy_data.get('pruning_amounts', strategy_data.get('sparsities', []))
                if target_sparsity in sparsities:
                    idx = sparsities.index(target_sparsity)
                    before = strategy_data.get('accuracies_before_finetune', [])
                    after = strategy_data.get('accuracies_after_finetune', [])
                    if idx < len(before) and idx < len(after):
                        strategy_names.append(strategy_key.replace('_', '\n').title())
                        before_vals.append(before[idx])
                        after_vals.append(after[idx])
            
            if strategy_names:
                x = np.arange(len(strategy_names))
                width = 0.35
                ax4.bar(x - width/2, before_vals, width, label='Before', alpha=0.8)
                ax4.bar(x + width/2, after_vals, width, label='After', alpha=0.8)
                ax4.set_xticks(x)
                ax4.set_xticklabels(strategy_names, fontsize=8)
                ax4.set_ylabel('Accuracy (%)', fontsize=11)
                ax4.set_title(f'Strategy Comparison at {target_sparsity*100:.0f}% Sparsity', 
                             fontsize=13, fontweight='bold')
                ax4.legend(fontsize=10)
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Pruning Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig

    # ========== Supernode Analysis Plots ==========

    def plot_supernode_activation_distribution(
        self,
        activation_values: Union[torch.Tensor, np.ndarray],
        threshold_value: float,
        threshold_percentile: float,
        layer_name: str = "",
        metric_name: str = "scar_activation_power",
        save_path: Optional[Union[str, Path]] = None,
        log_scale: bool = True,
    ) -> Figure:
        """
        Plot distribution of supernode scores with threshold.

        Args:
            activation_values: Array of score values for all neurons
            threshold_value: The threshold value separating supernodes
            threshold_percentile: Percentile of threshold (e.g., 0.01 for top 1%)
            layer_name: Layer name for title
            metric_name: Name of the metric used for supernode identification
            save_path: Path to save figure
            log_scale: Whether to use log scale on y-axis

        Returns:
            Matplotlib figure
        """
        vals = self._to_numpy(activation_values)
        
        # Format metric name for display
        metric_display = metric_name.replace("_", " ").replace("scar ", "").title()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(vals, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(threshold_value, color='coral', linestyle='--', linewidth=2,
                   label=f"Supernode threshold (top {threshold_percentile*100:.1f}%)")
        ax.set_xlabel(metric_display)
        ax.set_ylabel("Count")
        title = f"Supernode Score Distribution ({metric_display})"
        if layer_name:
            title += f"\n{layer_name}"
        ax.set_title(title)
        ax.legend()
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved supernode score distribution to {save_path}")

        return fig

    def plot_outgoing_weights_distribution(
        self,
        weights: Union[torch.Tensor, np.ndarray],
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot histogram of outgoing weights from supernodes.

        Args:
            weights: Flattened array of weight values
            layer_name: Layer name for title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        vals = self._to_numpy(weights).flatten()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(vals, bins=100, alpha=0.7, color='steelblue', edgecolor='none')
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Count")
        title = "Outgoing Weights from Supernodes"
        if layer_name:
            title += f" - {layer_name}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved outgoing weights distribution to {save_path}")

        return fig

    def plot_supernode_influence(
        self,
        influence_values: Union[torch.Tensor, np.ndarray],
        threshold_value: float,
        threshold_percentile: float,
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot distribution of supernode influence on downstream neurons.

        Args:
            influence_values: Total weight from supernodes for each output neuron
            threshold_value: Threshold for "follower" neurons
            threshold_percentile: Percentile of threshold
            layer_name: Layer name for title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        vals = self._to_numpy(influence_values)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(vals, bins=100, alpha=0.7, color='steelblue', edgecolor='none',
                label="All output neurons")
        ax.axvline(threshold_value, color='coral', linestyle='--', linewidth=2,
                   label=f"Follower threshold (top {threshold_percentile*100:.1f}%)")
        ax.set_xlabel("Total Weight from Supernodes")
        ax.set_ylabel("Count")
        title = "Supernode Influence on Output Neurons"
        if layer_name:
            title += f" - {layer_name}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved supernode influence distribution to {save_path}")

        return fig

    def plot_correlation_matrix(
        self,
        corr_matrix: Union[torch.Tensor, np.ndarray],
        title: str = "Correlation Matrix",
        xlabel: str = "Neuron Index",
        ylabel: str = "Neuron Index",
        save_path: Optional[Union[str, Path]] = None,
        vmin: float = -1,
        vmax: float = 1,
    ) -> Figure:
        """
        Plot a correlation matrix as a heatmap.

        Args:
            corr_matrix: Square correlation matrix
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap

        Returns:
            Matplotlib figure
        """
        matrix = self._to_numpy(corr_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Correlation")
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation matrix to {save_path}")

        return fig

    def plot_redundancy_comparison(
        self,
        high_redundancy: Union[torch.Tensor, np.ndarray],
        low_redundancy: Union[torch.Tensor, np.ndarray],
        high_mean: float,
        low_mean: float,
        layer_name: str = "",
        follower_fraction: float = 0.1,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> List[Figure]:
        """
        Create comparison plots for redundancy between high and low supernode-connected neurons.

        Args:
            high_redundancy: Pairwise correlations for high-connected neurons
            low_redundancy: Pairwise correlations for low-connected neurons
            high_mean: Mean redundancy for high-connected group
            low_mean: Mean redundancy for low-connected group
            layer_name: Layer name for titles
            follower_fraction: Fraction used to define high/low groups
            save_dir: Directory to save figures

        Returns:
            List of matplotlib figures
        """
        high_vals = self._to_numpy(high_redundancy)
        low_vals = self._to_numpy(low_redundancy)

        figures = []
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        layer_suffix = layer_name.replace('.', '_') if layer_name else "layer"

        # Plot 1: Side-by-side histograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(high_vals, bins=50, alpha=0.7, color='coral', edgecolor='darkred')
        axes[0].axvline(high_mean, color='darkred', linestyle='--', linewidth=2,
                        label=f"Mean: {high_mean:.4f}")
        axes[0].set_xlabel("Absolute Pairwise Correlation")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"High Supernode-Connected Neurons\n(top {follower_fraction*100:.0f}%)")
        axes[0].legend()
        axes[0].set_xlim(0, 1)

        axes[1].hist(low_vals, bins=50, alpha=0.7, color='steelblue', edgecolor='darkblue')
        axes[1].axvline(low_mean, color='darkblue', linestyle='--', linewidth=2,
                        label=f"Mean: {low_mean:.4f}")
        axes[1].set_xlabel("Absolute Pairwise Correlation")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Low Supernode-Connected Neurons\n(bottom {follower_fraction*100:.0f}%)")
        axes[1].legend()
        axes[1].set_xlim(0, 1)

        plt.suptitle(f"Redundancy Comparison: High vs Low Supernode Connection - {layer_name}", fontsize=12)
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / f"redundancy_comparison_sidebyside_{layer_suffix}.png",
                        dpi=self.dpi, bbox_inches='tight')
        figures.append(fig)

        # Plot 2: Overlaid histograms
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.hist(high_vals, bins=50, alpha=0.6, color='coral',
                label=f"High-connected (mean={high_mean:.4f})")
        ax.hist(low_vals, bins=50, alpha=0.6, color='steelblue',
                label=f"Low-connected (mean={low_mean:.4f})")
        ax.axvline(high_mean, color='darkred', linestyle='--', linewidth=2)
        ax.axvline(low_mean, color='darkblue', linestyle='--', linewidth=2)
        ax.set_xlabel("Absolute Pairwise Correlation (Redundancy)")
        ax.set_ylabel("Count")
        ax.set_title(f"Redundancy: High vs Low Supernode-Connected Neurons\n{layer_name}")
        ax.legend()
        ax.set_xlim(0, 1)
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / f"redundancy_comparison_overlay_{layer_suffix}.png",
                        dpi=self.dpi, bbox_inches='tight')
        figures.append(fig)

        # Plot 3: Box plot comparison (without outlier dots for cleaner visualization)
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([high_vals, low_vals], labels=['High-connected', 'Low-connected'],
                        patch_artist=True, notch=True, showfliers=False)
        bp['boxes'][0].set_facecolor('coral')
        bp['boxes'][1].set_facecolor('steelblue')
        ax.set_ylabel("Absolute Pairwise Correlation (Redundancy)")

        # Compute effect size
        pooled_std = np.sqrt((np.std(high_vals)**2 + np.std(low_vals)**2) / 2)
        effect_size = (high_mean - low_mean) / (pooled_std + 1e-8)
        ax.set_title(f"Redundancy Distribution Comparison\n{layer_name}\n(Effect size: {effect_size:.3f})")
        plt.tight_layout()

        if save_dir:
            fig.savefig(save_dir / f"redundancy_comparison_boxplot_{layer_suffix}.png",
                        dpi=self.dpi, bbox_inches='tight')
        figures.append(fig)

        return figures

    def plot_metric_scatter_by_group(
        self,
        x_values: Union[torch.Tensor, np.ndarray],
        y_values: Union[torch.Tensor, np.ndarray],
        group_labels: Union[torch.Tensor, np.ndarray, List[str]],
        xlabel: str,
        ylabel: str,
        title: str,
        save_path: Optional[Union[str, Path]] = None,
        colors: Optional[Dict[str, str]] = None,
    ) -> Figure:
        """
        Scatter plot with points colored by group.

        Args:
            x_values: X-axis values
            y_values: Y-axis values
            group_labels: Group label for each point
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure
            colors: Optional mapping of group names to colors

        Returns:
            Matplotlib figure
        """
        x = self._to_numpy(x_values)
        y = self._to_numpy(y_values)

        if isinstance(group_labels, (torch.Tensor, np.ndarray)):
            labels = self._to_numpy(group_labels)
        else:
            labels = np.array(group_labels)

        fig, ax = plt.subplots(figsize=self.figsize)

        unique_groups = np.unique(labels)
        default_colors = {'High': 'coral', 'Low': 'steelblue', 'high': 'coral', 'low': 'steelblue'}
        colors = colors or default_colors

        for group in unique_groups:
            mask = labels == group
            color = colors.get(str(group), None)
            ax.scatter(x[mask], y[mask], alpha=0.6, label=str(group), c=color, s=20)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved grouped scatter to {save_path}")

        return fig

    def plot_rq_vs_mi(
        self,
        rq_scores: Union[torch.Tensor, np.ndarray],
        mi_scores: Union[torch.Tensor, np.ndarray],
        redundancy_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Scatter plot of Rayleigh Quotient vs Mutual Information.

        Args:
            rq_scores: RQ values per neuron
            mi_scores: MI values per neuron
            redundancy_scores: Optional redundancy values for coloring
            layer_name: Layer name for title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        rq = self._to_numpy(rq_scores)
        mi = self._to_numpy(mi_scores)

        fig, ax = plt.subplots(figsize=self.figsize)

        if redundancy_scores is not None:
            redundancy = self._to_numpy(redundancy_scores)
            # Ensure same length
            min_len = min(len(rq), len(mi), len(redundancy))
            scatter = ax.scatter(rq[:min_len], mi[:min_len], c=redundancy[:min_len],
                                 cmap='viridis', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, label="Redundancy")
        else:
            ax.scatter(rq, mi, alpha=0.6, s=30, c='steelblue')

        ax.set_xlabel("Rayleigh Quotient")
        ax.set_ylabel("Mutual Information")
        title = "RQ vs MI"
        if layer_name:
            title += f" - {layer_name}"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved RQ vs MI plot to {save_path}")

        return fig

    # =========================================================================
    # Supernode Robustness Analysis Plots
    # =========================================================================

    def plot_metric_similarity_heatmap(
        self,
        similarity_matrix: np.ndarray,
        metric_names: List[str],
        title: str = "Metric Similarity",
        save_path: Optional[Union[str, Path]] = None,
        cmap: str = "YlOrRd",
        vmin: Optional[float] = 0,
        vmax: Optional[float] = 1,
    ) -> Figure:
        """
        Plot a heatmap showing similarity/correlation between different metrics.
        
        Args:
            similarity_matrix: Square matrix of similarity values
            metric_names: Names of metrics (for axis labels)
            title: Plot title
            save_path: Path to save figure
            cmap: Colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Similarity" if "Jaccard" in title else "Correlation")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(metric_names)))
        
        # Shorten metric names for display
        short_names = [m.replace('scar_', '').replace('gaussian_', '').replace('_', '\n') 
                       for m in metric_names]
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        
        # Add value annotations
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                val = similarity_matrix[i, j]
                text_color = 'white' if val > 0.5 * (vmax if vmax else 1) else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved metric similarity heatmap to {save_path}")
        
        return fig

    def plot_supernode_stability_distribution(
        self,
        stability_scores: np.ndarray,
        num_supernodes: int,
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot the distribution of supernode stability scores from bootstrap analysis.
        
        Args:
            stability_scores: Array of stability scores (0-1) for each neuron
            num_supernodes: Number of supernodes being selected
            layer_name: Layer name for title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Histogram of stability scores
        ax = axes[0]
        ax.hist(stability_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='none')
        ax.axvline(0.8, color='coral', linestyle='--', linewidth=2, 
                   label='High stability threshold (80%)')
        ax.axvline(0.5, color='orange', linestyle=':', linewidth=2,
                   label='Moderate threshold (50%)')
        
        highly_stable = np.sum(stability_scores > 0.8)
        ax.set_xlabel("Stability Score (fraction of bootstrap samples)")
        ax.set_ylabel("Number of Neurons")
        ax.set_title(f"Bootstrap Stability Distribution\n{highly_stable} highly stable neurons (>80%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Right: Sorted stability scores (stability curve)
        ax = axes[1]
        sorted_stability = np.sort(stability_scores)[::-1]
        x = np.arange(len(sorted_stability))
        
        ax.fill_between(x, sorted_stability, alpha=0.3, color='steelblue')
        ax.plot(x, sorted_stability, color='steelblue', linewidth=1.5)
        ax.axvline(num_supernodes, color='coral', linestyle='--', linewidth=2,
                   label=f'Supernode threshold (top {num_supernodes})')
        ax.axhline(0.8, color='orange', linestyle=':', linewidth=1.5,
                   label='80% stability line')
        
        ax.set_xlabel("Neuron Rank (sorted by stability)")
        ax.set_ylabel("Stability Score")
        ax.set_title("Stability Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(len(sorted_stability), num_supernodes * 10))
        
        layer_suffix = f" - {layer_name}" if layer_name else ""
        fig.suptitle(f"Supernode Stability Analysis{layer_suffix}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved stability distribution to {save_path}")
        
        return fig

    def plot_supernode_consistency_bars(
        self,
        metric_supernode_indices: Dict[str, set],
        total_neurons: int,
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot bar chart showing supernode consistency across metrics.
        
        Shows how many neurons appear as supernodes in 1, 2, 3, ... N metrics.
        
        Args:
            metric_supernode_indices: Dict mapping metric names to sets of supernode indices
            total_neurons: Total number of neurons in the layer
            layer_name: Layer name for title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Count how many metrics identify each neuron as supernode
        neuron_metric_count = np.zeros(total_neurons)
        for indices in metric_supernode_indices.values():
            for idx in indices:
                if idx < total_neurons:
                    neuron_metric_count[idx] += 1
        
        n_metrics = len(metric_supernode_indices)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Bar chart of neurons by number of metrics
        ax = axes[0]
        counts_per_level = [np.sum(neuron_metric_count == i) for i in range(n_metrics + 1)]
        x_labels = [f"{i}" for i in range(n_metrics + 1)]
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_metrics + 1))
        
        bars = ax.bar(x_labels, counts_per_level, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Number of Metrics Identifying as Supernode")
        ax.set_ylabel("Number of Neurons")
        ax.set_title("Supernode Identification Consistency")
        
        # Add value labels on bars
        for bar, count in zip(bars, counts_per_level):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{int(count)}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Right: Stacked bar for each metric
        ax = axes[1]
        metrics = list(metric_supernode_indices.keys())
        x_pos = np.arange(len(metrics))
        
        # Calculate unique vs shared for each metric
        unique_counts = []
        shared_counts = []
        for metric in metrics:
            indices = metric_supernode_indices[metric]
            unique = sum(1 for idx in indices if neuron_metric_count[idx] == 1)
            shared = len(indices) - unique
            unique_counts.append(unique)
            shared_counts.append(shared)
        
        short_names = [m.replace('scar_', '').replace('gaussian_', '').replace('_analytic', '')
                       for m in metrics]
        
        ax.bar(x_pos, unique_counts, label='Unique to this metric', color='lightcoral', edgecolor='darkred')
        ax.bar(x_pos, shared_counts, bottom=unique_counts, label='Shared with others', 
               color='lightgreen', edgecolor='darkgreen')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("Number of Supernodes")
        ax.set_title("Unique vs Shared Supernodes per Metric")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        layer_suffix = f" - {layer_name}" if layer_name else ""
        fig.suptitle(f"Cross-Metric Supernode Consistency{layer_suffix}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved consistency bars to {save_path}")
        
        return fig

    def plot_metric_score_scatter_matrix(
        self,
        metric_scores: Dict[str, np.ndarray],
        supernode_indices: set,
        layer_name: str = "",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot scatter matrix comparing scores from different metrics.
        
        Args:
            metric_scores: Dict mapping metric names to score arrays
            supernode_indices: Set of supernode indices to highlight
            layer_name: Layer name for title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        metrics = list(metric_scores.keys())
        n_metrics = len(metrics)
        
        if n_metrics < 2:
            logger.warning("Need at least 2 metrics for scatter matrix")
            return plt.figure()
        
        fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(3 * n_metrics, 3 * n_metrics))
        
        # Create supernode mask
        n_neurons = len(next(iter(metric_scores.values())))
        supernode_mask = np.zeros(n_neurons, dtype=bool)
        for idx in supernode_indices:
            if idx < n_neurons:
                supernode_mask[idx] = True
        
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                ax = axes[i, j] if n_metrics > 1 else axes
                
                scores1 = metric_scores[m1]
                scores2 = metric_scores[m2]
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(scores1[~supernode_mask], bins=30, alpha=0.5, color='steelblue',
                           label='Regular', density=True)
                    ax.hist(scores1[supernode_mask], bins=30, alpha=0.7, color='coral',
                           label='Supernodes', density=True)
                    ax.set_xlabel(m1.replace('scar_', '').replace('_', ' '))
                    if j == 0:
                        ax.legend(fontsize=7)
                else:
                    # Off-diagonal: scatter
                    ax.scatter(scores2[~supernode_mask], scores1[~supernode_mask], 
                              s=5, alpha=0.3, c='steelblue', label='Regular')
                    ax.scatter(scores2[supernode_mask], scores1[supernode_mask],
                              s=20, alpha=0.8, c='coral', edgecolors='darkred', 
                              linewidths=0.5, label='Supernodes')
                    
                    # Add correlation
                    from scipy import stats
                    corr, _ = stats.spearmanr(scores1, scores2)
                    ax.text(0.05, 0.95, f'ρ={corr:.2f}', transform=ax.transAxes,
                           fontsize=8, va='top', fontweight='bold')
                    
                    if i == n_metrics - 1:
                        ax.set_xlabel(m2.replace('scar_', '').replace('_', ' '), fontsize=9)
                    if j == 0:
                        ax.set_ylabel(m1.replace('scar_', '').replace('_', ' '), fontsize=9)
        
        layer_suffix = f" - {layer_name}" if layer_name else ""
        fig.suptitle(f"Metric Score Correlations{layer_suffix}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved scatter matrix to {save_path}")
        
        return fig

    def _to_numpy(self, data: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().to(torch.float32).numpy()
        if isinstance(data, np.ndarray):
            return data.astype(np.float32, copy=False)
        return np.asarray(data, dtype=np.float32)


# Convenience functions for quick plotting


def plot_quick_summary(scores: Dict[str, Any], title: str = "Summary", save_path: Optional[str] = None):
    """Quick plotting function for immediate visualization."""
    visualizer = UnifiedVisualizer()

    if isinstance(next(iter(scores.values())), (list, np.ndarray, torch.Tensor)):
        # Layer scores
        fig = visualizer.plot_layer_scores(scores, title, save_path=save_path)
    else:
        # Time series
        steps = list(range(len(next(iter(scores.values())))))
        fig = visualizer.plot_metric_evolution(steps, scores, title=title, save_path=save_path)

    if not save_path:
        plt.show()

    return fig


def generate_experiment_visualizations(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    config: Optional[Any] = None,
    dpi: int = 300,
) -> List[Path]:
    """
    Generate all standard visualizations for an experiment.
    
    This function can be called from both vision and LLM experiments to produce
    consistent visualizations. It reads the results dictionary and generates
    appropriate plots based on what data is available.
    
    UNIFIED FORMAT: Uses subfolders matching LLM experiment structure:
        - plots/pruning/       - Pruning comparison plots
        - plots/histograms/    - Score distribution histograms
        - plots/scatter/       - Metric scatter plots
        - plots/redundancy/    - Redundancy heatmaps
        - plots/training/      - Training curves
    
    Args:
        results: Experiment results dictionary containing:
            - train_results: Training history (losses, accuracies, alignment)
            - test_results: Final evaluation and alignment scores
            - dropout_results: Progressive dropout analysis
            - pruning_results: Pruning experiment results
            - eigenfeature_results: Eigenfeature analysis
        output_dir: Directory to save plots
        config: Optional experiment config for additional settings
        dpi: DPI for saved figures
        
    Returns:
        List of paths to generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders for organized output (matching LLM experiment format)
    pruning_dir = output_dir / "pruning"
    histogram_dir = output_dir / "histograms"
    scatter_dir = output_dir / "scatter"
    redundancy_dir = output_dir / "redundancy"
    training_dir = output_dir / "training"
    
    for d in [pruning_dir, histogram_dir, scatter_dir, redundancy_dir, training_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    visualizer = UnifiedVisualizer()
    visualizer.dpi = dpi
    generated_plots = []
    
    # Training curves (saved to training/ subfolder)
    train_results = results.get("train_results", {})
    if train_results:
        train_losses = train_results.get("train_losses", [])
        val_losses = train_results.get("val_losses", [])
        train_accs = train_results.get("train_accs", [])
        val_accs = train_results.get("val_accs", [])
        epochs = list(range(1, len(train_losses) + 1))
        
        if epochs and train_losses:
            # Loss plot
            loss_series = {"Train Loss": train_losses}
            if val_losses and len(val_losses) == len(epochs):
                loss_series["Val Loss"] = val_losses
            
            fig = visualizer.plot_metric_evolution(
                epochs, loss_series,
                title="Training Loss",
                xlabel="Epoch",
                ylabel="Loss",
                save_path=training_dir / "training_loss.png",
            )
            plt.close(fig)
            generated_plots.append(training_dir / "training_loss.png")
            
            # Accuracy plot
            if train_accs:
                acc_series = {"Train Acc": train_accs}
                if val_accs and len(val_accs) == len(epochs):
                    acc_series["Val Acc"] = val_accs
                
                fig = visualizer.plot_metric_evolution(
                    epochs, acc_series,
                    title="Training Accuracy",
                    xlabel="Epoch",
                    ylabel="Accuracy (%)",
                    save_path=training_dir / "training_accuracy.png",
                )
                plt.close(fig)
                generated_plots.append(training_dir / "training_accuracy.png")
    
    # Alignment evolution (saved to training/ subfolder)
    alignment_history = train_results.get("alignment", {})
    if alignment_history:
        for method, history in alignment_history.items():
            if not history:
                continue
            
            # Summarize alignment across layers
            summarized = []
            for snapshot in history:
                if isinstance(snapshot, dict):
                    aggregated = []
                    for layer_scores in snapshot.values():
                        if isinstance(layer_scores, (list, np.ndarray)):
                            aggregated.extend(layer_scores)
                    if aggregated:
                        summarized.append(float(np.mean(aggregated)))
                elif isinstance(snapshot, (int, float)):
                    summarized.append(float(snapshot))
            
            if summarized:
                steps = list(range(1, len(summarized) + 1))
                fig = visualizer.plot_metric_evolution(
                    steps, {method: summarized},
                    title=f"{method.replace('_', ' ').title()} Evolution",
                    xlabel="Measurement",
                    ylabel="Average Score",
                    save_path=training_dir / f"alignment_{method}.png",
                )
                plt.close(fig)
                generated_plots.append(training_dir / f"alignment_{method}.png")
    
    # Dropout analysis (saved to training/ subfolder)
    dropout_results = results.get("dropout_results", {})
    if dropout_results:
        dropout_rates = dropout_results.get("dropout_rates", [])
        if dropout_rates:
            # Extract accuracy curves
            accuracy_curves = {}
            if "accuracies" in dropout_results:
                accuracy_curves = dropout_results["accuracies"]
            else:
                for strategy in ["low", "high", "random"]:
                    key = f"accuracies_{strategy}"
                    if key in dropout_results:
                        accuracy_curves[strategy] = dropout_results[key]
            
            if accuracy_curves:
                dropout_pct = [rate * 100 for rate in dropout_rates]
                fig = visualizer.plot_metric_evolution(
                    dropout_pct, accuracy_curves,
                    title="Dropout Analysis",
                    xlabel="Dropout (%)",
                    ylabel="Accuracy (%)",
                    save_path=training_dir / "dropout_accuracy.png",
                )
                plt.close(fig)
                generated_plots.append(training_dir / "dropout_accuracy.png")
    
    # Pruning results (saved to pruning/ subfolder, matching LLM format)
    pruning_results = results.get("pruning_results", {})
    
    # Calculate total_params from model info if available
    total_params = None
    if "model_info" in results:
        total_params = results["model_info"].get("total_params")
    elif "test_results" in results and "model_info" in results["test_results"]:
        total_params = results["test_results"]["model_info"].get("total_params")
    # Try to get from config if available
    if total_params is None and config is not None:
        if hasattr(config, "model") and hasattr(config.model, "total_params"):
            total_params = config.model.total_params
        # Try to get model from config and calculate params
        elif hasattr(config, "model") and hasattr(config.model, "parameters"):
            try:
                total_params = sum(p.numel() for p in config.model.parameters() if p.requires_grad)
            except:
                pass
    # If still None, try to estimate from pruning results (if we have sparsity info)
    # This is a fallback - we can't calculate exactly but can use a reasonable estimate
    if total_params is None and pruning_results and "strategies" in pruning_results:
        # Try to infer from first strategy's data if available
        first_strategy = next(iter(pruning_results["strategies"].values()), None)
        if first_strategy and "total_params" in first_strategy:
            total_params = first_strategy["total_params"]
    
    if pruning_results and "strategies" in pruning_results:
        # Group by algorithm for comparison plots
        algorithm_results = {}
        
        for strategy_key, strategy_data in pruning_results["strategies"].items():
            if not strategy_data.get("pruning_amounts") and not strategy_data.get("sparsities"):
                continue
            
            # Parse strategy name and mode (use lowercase underscores like LLM)
            if "_" in strategy_key:
                parts = strategy_key.rsplit("_", 1)
                if parts[1] in ["low", "high", "random"]:
                    algorithm = parts[0].lower().replace(" ", "_")
                    mode = parts[1]
                else:
                    algorithm = strategy_key.lower().replace(" ", "_")
                    mode = "low"
            else:
                algorithm = strategy_key.lower().replace(" ", "_")
                mode = "low"
            
            if algorithm not in algorithm_results:
                algorithm_results[algorithm] = {
                    "sparsities": strategy_data.get("pruning_amounts", strategy_data.get("sparsities", [])),
                    "before": {},
                    "after": {},
                    "before_std": {},
                    "after_std": {},
                    "before_losses": {},
                    "after_losses": {},
                    "before_losses_std": {},
                    "after_losses_std": {},
                }
            
            algorithm_results[algorithm]["before"][mode] = strategy_data.get("accuracies_before_finetune", [])
            algorithm_results[algorithm]["after"][mode] = strategy_data.get("accuracies_after_finetune", [])
            
            if "accuracies_before_finetune_std" in strategy_data:
                algorithm_results[algorithm]["before_std"][mode] = strategy_data["accuracies_before_finetune_std"]
            if "accuracies_after_finetune_std" in strategy_data:
                algorithm_results[algorithm]["after_std"][mode] = strategy_data["accuracies_after_finetune_std"]
            
            # Collect loss data
            if "losses_before_finetune" in strategy_data:
                algorithm_results[algorithm]["before_losses"][mode] = strategy_data["losses_before_finetune"]
            if "losses_after_finetune" in strategy_data:
                algorithm_results[algorithm]["after_losses"][mode] = strategy_data["losses_after_finetune"]
            if "losses_before_finetune_std" in strategy_data:
                algorithm_results[algorithm]["before_losses_std"][mode] = strategy_data["losses_before_finetune_std"]
            if "losses_after_finetune_std" in strategy_data:
                algorithm_results[algorithm]["after_losses_std"][mode] = strategy_data["losses_after_finetune_std"]
        
        # Generate plots for each algorithm (use pruning_dir subfolder)
        for algorithm, data in algorithm_results.items():
            if not data["sparsities"]:
                continue
            
            # Before/after accuracy comparison (save to pruning subfolder)
            figs = visualizer.plot_pruning_before_after(
                sparsities=data["sparsities"],
                before_accuracies=data["before"],
                after_accuracies=data["after"],
                before_std=data.get("before_std") or None,
                after_std=data.get("after_std") or None,
                algorithm=algorithm.replace("_", " ").title(),
                save_dir=pruning_dir,
                dpi=dpi,
                total_params=total_params,
            )
            for fig in figs:
                plt.close(fig)
            
            # Before/after loss comparison (save to pruning subfolder)
            if data.get("before_losses") and data.get("after_losses"):
                loss_figs = visualizer.plot_pruning_loss_before_after(
                    sparsities=data["sparsities"],
                    before_losses=data["before_losses"],
                    after_losses=data["after_losses"],
                    before_std=data.get("before_losses_std") or None,
                    after_std=data.get("after_losses_std") or None,
                    algorithm=algorithm.replace("_", " ").title(),
                    save_dir=pruning_dir,
                    dpi=dpi,
                    total_params=total_params,
                )
                for fig in loss_figs:
                    plt.close(fig)
            
            # Track generated files (using consistent lowercase names)
            for mode in data["before"]:
                generated_plots.append(pruning_dir / f"pruning_{algorithm}_accuracy_before.png")
                generated_plots.append(pruning_dir / f"pruning_{algorithm}_accuracy_after.png")
                if data.get("before_losses") and mode in data["before_losses"]:
                    generated_plots.append(pruning_dir / f"pruning_{algorithm}_loss_before.png")
                    generated_plots.append(pruning_dir / f"pruning_{algorithm}_loss_after.png")
        
        # Comparison plot across all strategies (save to pruning subfolder)
        if algorithm_results:
            try:
                comparison_data = {}
                for strategy_key, strategy_data in pruning_results["strategies"].items():
                    # Normalize key to lowercase underscores
                    normalized_key = strategy_key.lower().replace(" ", "_")
                    comparison_data[normalized_key] = {
                        "sparsities": strategy_data.get("pruning_amounts", strategy_data.get("sparsities", [])),
                        "accuracies_after_finetune": strategy_data.get("accuracies_after_finetune", []),
                        "accuracies_std": strategy_data.get("accuracies_after_finetune_std"),
                    }
                
                fig = visualizer.plot_pruning_comparison(
                    results=comparison_data,
                    metric="accuracy",
                    title="Pruning Strategy Comparison",
                    save_path=pruning_dir / "pruning_comparison.png",
                    total_params=total_params,
                )
                plt.close(fig)
                generated_plots.append(pruning_dir / "pruning_comparison.png")
            except Exception as e:
                logger.warning(f"Could not generate comparison plot: {e}")
    
    # Eigenfeature analysis (saved to training/ subfolder)
    eigenfeature_results = results.get("eigenfeature_results", {})
    if eigenfeature_results:
        eigen_data = {}
        for layer_name, info in eigenfeature_results.items():
            eigenvalues = info.get("top_eigenvalues", [])
            if eigenvalues:
                eigen_data[layer_name] = {f"eig{i+1}": val for i, val in enumerate(eigenvalues)}
        
        if eigen_data:
            fig = visualizer.plot_heatmap(
                data=eigen_data,
                title="Top Eigenvalues per Layer",
                xlabel="Eigenvalue Index",
                ylabel="Layer",
                save_path=training_dir / "eigenvalues_heatmap.png",
            )
            plt.close(fig)
            generated_plots.append(training_dir / "eigenvalues_heatmap.png")
    
    # ========== Histograms from Alignment Scores (saved to histograms/ subfolder) ==========
    # Check for alignment scores in test_results
    test_results = results.get("test_results", {})
    alignment_scores = test_results.get("alignment", {})
    
    if alignment_scores:
        # Generate histograms for each metric (use histograms subfolder)
        for metric_name, layer_scores in alignment_scores.items():
            if isinstance(layer_scores, dict):
                # Aggregate scores across layers for histogram
                all_scores = []
                for layer_name, scores in layer_scores.items():
                    if isinstance(scores, (list, np.ndarray)):
                        all_scores.extend(scores)
                    elif isinstance(scores, torch.Tensor):
                        all_scores.extend(scores.cpu().numpy().tolist())
                
                if all_scores:
                    # Use lowercase underscore naming like LLM
                    metric_name_safe = metric_name.lower().replace(" ", "_")
                    fig = visualizer.plot_1d_histogram(
                        values=all_scores,
                        title=f"{metric_name.replace('_', ' ').title()} Distribution",
                        xlabel="Score",
                        ylabel="Count",
                        bins=50,
                        save_path=histogram_dir / f"histogram_{metric_name_safe}.png",
                    )
                    plt.close(fig)
                    generated_plots.append(histogram_dir / f"histogram_{metric_name_safe}.png")
                    
                    # Also generate per-layer histograms
                    for layer_name, scores in layer_scores.items():
                        if isinstance(scores, (list, np.ndarray)) and len(scores) > 10:
                            safe_layer = layer_name.replace(".", "_").replace("/", "_").lower()
                            fig = visualizer.plot_1d_histogram(
                                values=scores,
                                title=f"{metric_name.replace('_', ' ').title()} - {layer_name}",
                                xlabel="Score",
                                ylabel="Count",
                                bins=30,
                                save_path=histogram_dir / f"histogram_{metric_name_safe}_{safe_layer}.png",
                            )
                            plt.close(fig)
                            generated_plots.append(histogram_dir / f"histogram_{metric_name_safe}_{safe_layer}.png")
        
        # Generate scatter plots for metric pairs (saved to scatter/ subfolder)
        metric_names = list(alignment_scores.keys())
        if len(metric_names) >= 2:
            # Generate scatter plots for pairs of metrics
            for i in range(len(metric_names)):
                for j in range(i + 1, len(metric_names)):
                    metric1, metric2 = metric_names[i], metric_names[j]
                    scores1 = alignment_scores[metric1]
                    scores2 = alignment_scores[metric2]
                    
                    if isinstance(scores1, dict) and isinstance(scores2, dict):
                        # Aggregate per layer
                        for layer_name in scores1:
                            if layer_name in scores2:
                                s1 = scores1[layer_name]
                                s2 = scores2[layer_name]
                                
                                if isinstance(s1, (list, np.ndarray)) and isinstance(s2, (list, np.ndarray)):
                                    if len(s1) == len(s2) and len(s1) > 10:
                                        # Use lowercase underscore naming like LLM
                                        safe_layer = layer_name.replace(".", "_").replace("/", "_").lower()
                                        m1_safe = metric1.lower().replace(" ", "_")
                                        m2_safe = metric2.lower().replace(" ", "_")
                                        fig = visualizer.plot_scatter_2d(
                                            x=s1,
                                            y=s2,
                                            xlabel=metric1.replace("_", " ").title(),
                                            ylabel=metric2.replace("_", " ").title(),
                                            title=f"{metric1} vs {metric2} - {layer_name}",
                                            save_path=scatter_dir / f"scatter_{m1_safe}_vs_{m2_safe}_{safe_layer}.png",
                                        )
                                        plt.close(fig)
                                        generated_plots.append(scatter_dir / f"scatter_{m1_safe}_vs_{m2_safe}_{safe_layer}.png")
    
    # ========== Redundancy Heatmaps (saved to redundancy/ subfolder) ==========
    # Check for pairwise redundancy matrices in test_results
    redundancy_matrices = test_results.get("redundancy_matrices", {})
    if redundancy_matrices:
        for layer_name, matrix in redundancy_matrices.items():
            if matrix is not None and (isinstance(matrix, (np.ndarray, torch.Tensor))):
                # Use lowercase underscore naming like LLM
                safe_layer = layer_name.replace(".", "_").replace("/", "_").lower()
                try:
                    fig = visualizer.plot_pairwise_redundancy_matrix(
                        redundancy_matrix=matrix,
                        layer_name=layer_name,
                        save_path=redundancy_dir / f"redundancy_heatmap_{safe_layer}.png",
                    )
                    plt.close(fig)
                    generated_plots.append(redundancy_dir / f"redundancy_heatmap_{safe_layer}.png")
                    logger.info(f"Generated redundancy heatmap for {layer_name}")
                except Exception as e:
                    logger.warning(f"Could not generate redundancy heatmap for {layer_name}: {e}")
    
    logger.info(f"Generated {len(generated_plots)} visualization(s) in {output_dir}")
    return generated_plots
