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
            ax.boxplot(data, positions=positions, labels=layer_names)
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
        bins: int = 100,
        logx: bool = False,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """
        Plot a 1D histogram for a single metric (e.g., activation variance, RQ, MI).

        Args:
            values: 1D tensor/array/list of scalar values.
            title:  Plot title.
            xlabel: X-axis label.
            bins:   Number of histogram bins.
            logx:   If True, plot log10 of positive values.
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
        ax.hist(arr, bins=bins, alpha=0.7, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("log10(value)" if logx else xlabel)
        ax.set_ylabel("Count")
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
    ) -> Figure:
        """
        Create a heatmap of mean SCAR metrics across layers.

        Args:
            scar_scores: Dict[layer_name -> Dict[metric_name -> scores_tensor]]
            metrics:     List of metric names to include (default: all present keys)
            title:       Plot title
            save_path:   Optional path to save the figure
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

        # Reuse generic heatmap helper (it will convert nested dict to DataFrame)
        return self.plot_heatmap(
            data=layer_metric_means,
            title=title,
            cmap="coolwarm",
            annotate=True,
            fmt=".3f",
            xlabel="Metric",
            ylabel="Layer",
            save_path=save_path,
        )

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

        Returns:
            List of generated figures.
        """
        figures = []
        x_values = [s * 100 for s in sparsities]

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
            fig_before.tight_layout()

            if save_dir:
                fig_before.savefig(save_dir / f"pruning_{algorithm}_accuracy_before.png",
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
            fig_after.tight_layout()

            if save_dir:
                fig_after.savefig(save_dir / f"pruning_{algorithm}_accuracy_after.png",
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
            fig.tight_layout()

            if save_dir:
                fig.savefig(save_dir / f"pruning_{algorithm}_accuracy.png",
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
