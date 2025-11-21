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
