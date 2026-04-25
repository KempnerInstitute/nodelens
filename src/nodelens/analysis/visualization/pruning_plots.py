"""
Advanced visualization tools for pruning experiments.

This module provides comprehensive plotting capabilities for pruning analysis,
including multi-strategy comparisons, accuracy/loss curves, and statistical analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Try to import seaborn, but make it optional
try:
    import seaborn as sns

    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class PruningVisualizer:
    """
    Advanced visualization for pruning experiments with multi-strategy support.
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize pruning visualizer.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        self.figsize = figsize

        # Define color palette for strategies
        self.strategy_colors = {
            "magnitude": "#1f77b4",
            "gradient": "#ff7f0e",
            "fisher": "#2ca02c",
            "random": "#d62728",
            "low": "#9467bd",
            "high": "#8c564b",
            "magnitude_low": "#1f77b4",
            "magnitude_high": "#17becf",
            "gradient_low": "#ff7f0e",
            "gradient_high": "#ffbb78",
        }

    def plot_pruning_performance(
        self,
        results: Dict[str, Dict[float, Dict[str, float]]],
        metrics: List[str] = ["accuracy", "loss"],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show_confidence: bool = True,
    ) -> plt.Figure:
        """
        Plot performance metrics for multiple pruning strategies.

        Args:
            results: Nested dict of strategy -> sparsity -> metric -> value
                     Can also include 'mean' and 'std' for confidence intervals
            metrics: List of metrics to plot
            save_path: Path to save the plot
            title: Overall title for the plot
            show_confidence: Whether to show confidence intervals if available

        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * num_metrics // 2))

        gs = GridSpec(num_metrics, 2, figure=fig, hspace=0.3, wspace=0.3)

        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx, :])

            for strategy, strategy_results in results.items():
                sparsities = sorted(strategy_results.keys())

                # Extract values
                if isinstance(list(strategy_results.values())[0], dict):
                    # Handle nested structure with mean/std
                    means = []
                    stds = []

                    for sparsity in sparsities:
                        if "mean" in strategy_results[sparsity]:
                            means.append(strategy_results[sparsity]["mean"][metric])
                            if "std" in strategy_results[sparsity] and show_confidence:
                                stds.append(strategy_results[sparsity]["std"][metric])
                        else:
                            means.append(strategy_results[sparsity].get(metric, 0))

                    # Plot with confidence intervals
                    color = self.strategy_colors.get(strategy, None)
                    line = ax.plot(sparsities, means, "o-", label=strategy, linewidth=2.5, markersize=8, color=color)

                    if stds and show_confidence:
                        means = np.array(means)
                        stds = np.array(stds)
                        ax.fill_between(sparsities, means - stds, means + stds, alpha=0.2, color=line[0].get_color())
                else:
                    # Simple structure
                    values = [strategy_results[s] for s in sparsities]
                    color = self.strategy_colors.get(strategy, None)
                    ax.plot(sparsities, values, "o-", label=strategy, linewidth=2.5, markersize=8, color=color)

            # Formatting
            ax.set_xlabel("Sparsity Level", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f"{metric.capitalize()} vs Sparsity", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=10)

            # Format x-axis as percentages
            ax.set_xticklabels([f"{s*100:.0f}%" for s in ax.get_xticks()])

            # Add annotations for key points
            if metric == "accuracy":
                ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Random baseline (10%)")

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved pruning performance plot to {save_path}")

        return fig

    def plot_pruning_comparison_grid(self, results: Dict[str, Dict[float, Dict[str, Any]]], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive grid comparing all aspects of pruning strategies.

        Args:
            results: Pruning results for multiple strategies
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metric_comparison(ax1, results, "accuracy", "Accuracy Comparison")

        # 2. Loss comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_metric_comparison(ax2, results, "loss", "Loss Comparison")

        # 3. Efficiency plot (accuracy drop vs sparsity)
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_efficiency(ax3, results)

        # 4. Strategy ranking
        ax4 = fig.add_subplot(gs[0, 2])
        self._plot_strategy_ranking(ax4, results)

        # 5. Sparsity distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_sparsity_distribution(ax5, results)

        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary_stats(ax6, results)

        fig.suptitle("Comprehensive Pruning Strategy Comparison", fontsize=16)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison grid to {save_path}")

        return fig

    def plot_multi_seed_results(
        self, results: Dict[str, List[Dict[float, Dict[str, float]]]], metric: str = "accuracy", save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot results from multiple seeds with statistical analysis.

        Args:
            results: Dict of strategy -> list of results per seed
            metric: Metric to plot
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Process data
        processed_results = {}
        for strategy, seed_results in results.items():
            sparsities = sorted(seed_results[0].keys())

            # Aggregate across seeds
            metric_values = {s: [] for s in sparsities}
            for seed_result in seed_results:
                for sparsity in sparsities:
                    if metric in seed_result[sparsity]:
                        metric_values[sparsity].append(seed_result[sparsity][metric])

            # Compute statistics
            means = [np.mean(metric_values[s]) for s in sparsities]
            stds = [np.std(metric_values[s]) for s in sparsities]

            processed_results[strategy] = {"sparsities": sparsities, "means": means, "stds": stds, "all_values": metric_values}

        # Plot 1: Mean with confidence intervals
        for strategy, data in processed_results.items():
            color = self.strategy_colors.get(strategy, None)
            line = ax1.plot(data["sparsities"], data["means"], "o-", label=f"{strategy} (μ±σ)", linewidth=2.5, markersize=8, color=color)

            # Add confidence intervals
            means = np.array(data["means"])
            stds = np.array(data["stds"])
            ax1.fill_between(data["sparsities"], means - stds, means + stds, alpha=0.2, color=line[0].get_color())

        ax1.set_xlabel("Sparsity Level")
        ax1.set_ylabel(metric.capitalize())
        ax1.set_title(f"{metric.capitalize()} Across Seeds (Mean ± Std)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Box plots at key sparsity levels
        key_sparsities = [0.5, 0.7, 0.9]
        box_data = []
        labels = []

        for sparsity in key_sparsities:
            for strategy, data in processed_results.items():
                if sparsity in data["all_values"]:
                    box_data.append(data["all_values"][sparsity])
                    labels.append(f"{strategy}\n{sparsity*100:.0f}%")

        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)

        # Color boxes by strategy
        for patch, label in zip(bp["boxes"], labels):
            strategy = label.split("\n")[0]
            patch.set_facecolor(self.strategy_colors.get(strategy, "gray"))
            patch.set_alpha(0.7)

        ax2.set_ylabel(metric.capitalize())
        ax2.set_title(f"{metric.capitalize()} Distribution at Key Sparsity Levels")
        ax2.grid(True, alpha=0.3, axis="y")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        fig.suptitle(f"Multi-Seed {metric.capitalize()} Analysis", fontsize=14)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved multi-seed plot to {save_path}")

        return fig

    def plot_layer_wise_pruning(
        self, layer_sparsities: Dict[str, Dict[str, float]], model_accuracy: Dict[str, float], save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize layer-wise pruning patterns.

        Args:
            layer_sparsities: Dict of strategy -> layer -> sparsity
            model_accuracy: Dict of strategy -> accuracy
            save_path: Path to save the plot

        Returns:
            Matplotlib figure
        """
        strategies = list(layer_sparsities.keys())
        layers = list(next(iter(layer_sparsities.values())).keys())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Prepare data for heatmap
        data = []
        for strategy in strategies:
            row = [layer_sparsities[strategy].get(layer, 0) for layer in layers]
            data.append(row)

        # Plot heatmap
        im = ax1.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

        # Set ticks and labels
        ax1.set_xticks(range(len(layers)))
        ax1.set_yticks(range(len(strategies)))
        ax1.set_xticklabels(layers, rotation=45, ha="right")
        ax1.set_yticklabels(strategies)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Sparsity Level", rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(layers)):
                ax1.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)

        ax1.set_title("Layer-wise Sparsity Patterns by Strategy")

        # Plot accuracy comparison
        ax2.bar(range(len(strategies)), [model_accuracy.get(s, 0) for s in strategies])
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels(strategies)
        ax2.set_ylabel("Final Accuracy")
        ax2.set_title("Model Accuracy After Pruning")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved layer-wise pruning plot to {save_path}")

        return fig

    # ------------------------------------------------------------------
    # New SCAR-specific visualizations
    # ------------------------------------------------------------------

    def plot_metric_distributions_from_alignment(
        self,
        alignment_history: Dict[str, List[Dict[str, List[float]]]],
        metric_mapping: Optional[Dict[str, str]] = None,
        save_dir: Optional[Union[str, Path]] = None,
        max_points: int = 1000,
    ) -> None:
        """
        Generate layer-wise violin/box plots for single-node metrics such as OI/T/R.

        Args:
            alignment_history: train_results["alignment"] from results JSON.
            metric_mapping: dict of metric_key -> pretty label.
            save_dir: directory to save figures.
            max_points: subsample per layer for plotting speed.
        """
        if not alignment_history:
            logger.warning("plot_metric_distributions_from_alignment: empty alignment history.")
            return

        if metric_mapping is None:
            metric_mapping = {
                "activation_outlier_index": "Outlier Index (OI)",
                "scar_taylor": "Taylor Saliency (T)",
                "scar_curvature": "Curvature (R)",
                "rayleigh_quotient": "Rayleigh Quotient",
            }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for metric_key, metric_label in metric_mapping.items():
            history = alignment_history.get(metric_key)
            if not history:
                continue

            snapshot = history[0] if isinstance(history, list) else history
            plot_rows = []
            for layer_name, scores in snapshot.items():
                try:
                    layer_idx = int(layer_name.split("layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    layer_idx = layer_name

                scores_arr = np.array(scores)
                if len(scores_arr) > max_points:
                    scores_arr = np.random.choice(scores_arr, max_points, replace=False)

                for score in scores_arr:
                    plot_rows.append({"Layer": layer_idx, "Score": score})

            if not plot_rows:
                continue

            df = pd.DataFrame(plot_rows)
            plt.figure(figsize=(12, 5))
            if HAS_SEABORN:
                sns.violinplot(data=df, x="Layer", y="Score", inner="quartile", linewidth=0.6, color="#7fc8f8")
            else:
                df.boxplot(column="Score", by="Layer")

            plt.title(f"Layer-wise Distribution of {metric_label}")
            plt.ylabel(metric_label)
            plt.xlabel("Layer Index")
            plt.tight_layout()

            if save_dir:
                filename = save_dir / f"distribution_{metric_key}.png"
                plt.savefig(filename, dpi=300)
                logger.info(f"Saved {filename}")
            plt.close()

    def plot_redundancy_synergy_scatter(
        self,
        redundancy_snapshot: Dict[str, List[float]],
        synergy_snapshot: Dict[str, List[float]],
        outlier_snapshot: Optional[Dict[str, List[float]]] = None,
        layers_to_plot: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Scatter plot of redundancy vs synergy for selected layers.
        """
        if not redundancy_snapshot or not synergy_snapshot:
            logger.warning("plot_redundancy_synergy_scatter: missing redundancy/synergy data.")
            return

        layers = sorted(redundancy_snapshot.keys())
        if not layers:
            return

        if layers_to_plot is None:
            indices = [len(layers) // 4, len(layers) // 2, min(len(layers) - 1, 3 * len(layers) // 4)]
            layers_to_plot = [layers[i] for i in indices if 0 <= i < len(layers)]

        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5 * len(layers_to_plot), 4), sharey=True)
        if len(layers_to_plot) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers_to_plot):
            r_vals = np.array(redundancy_snapshot.get(layer, []))
            s_vals = np.array(synergy_snapshot.get(layer, []))
            if r_vals.size == 0 or s_vals.size == 0:
                continue

            # Determine supernodes via outlier index (top 5%)
            if outlier_snapshot and layer in outlier_snapshot:
                oi_vals = np.array(outlier_snapshot[layer])
                threshold = np.percentile(oi_vals, 95)
                is_supernode = oi_vals > threshold
            else:
                is_supernode = np.zeros_like(r_vals, dtype=bool)

            ax.scatter(r_vals[~is_supernode], s_vals[~is_supernode], s=20, alpha=0.5, label="Followers", c="#5dade2", edgecolors="none")
            ax.scatter(r_vals[is_supernode], s_vals[is_supernode], s=35, alpha=0.9, label="Supernodes", c="#c0392b", edgecolors="white")

            ax.set_title(layer)
            ax.set_xlabel("Gaussian Redundancy")
            if ax == axes[0]:
                ax.set_ylabel("PID Synergy (logits)")
            ax.grid(True, alpha=0.3)

        axes[0].legend(loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved redundancy/synergy plot to {save_path}")
        plt.close()

    def plot_sparsity_perplexity_curves(
        self,
        df: pd.DataFrame,
        x_col: str = "Sparsity",
        y_col: str = "Perplexity",
        hue: str = "Method",
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Sparsity vs. Perplexity",
        y_label: Optional[str] = None,
    ) -> None:
        """
        Plot sparsity-metric curves from a tidy dataframe.

        This is written generically so it can be used for both perplexity
        (language models) and accuracy or loss (vision models) by changing
        ``y_col`` and ``y_label``.
        """
        if df.empty:
            logger.warning("plot_sparsity_perplexity_curves: empty dataframe.")
            return

        plt.figure(figsize=(8, 5))
        if HAS_SEABORN:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue, marker="o", linewidth=2.5, palette="tab10")
        else:
            for method, subdf in df.groupby(hue):
                plt.plot(subdf[x_col], subdf[y_col], "o-", label=method, linewidth=2.5)

        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_label or y_col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend(title=hue)

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved sparsity-perplexity plot to {save_path}")
        plt.close()

    def plot_alignment_vs_synergy_scatter(
        self,
        rq_snapshot: Dict[str, List[float]],
        synergy_snapshot: Dict[str, List[float]],
        layers_to_plot: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Scatter plot of alignment (Rayleigh Quotient) vs. Gaussian PID synergy.

        Intended for vision experiments where we want to see how strongly
        aligned channels (high RQ) contribute to label-conditional synergistic
        information.
        """
        if not rq_snapshot or not synergy_snapshot:
            logger.warning("plot_alignment_vs_synergy_scatter: missing RQ/synergy data.")
            return

        layers = sorted(rq_snapshot.keys())
        if not layers:
            return

        if layers_to_plot is None:
            # Pick roughly early / middle / late layers if available
            indices = [len(layers) // 4, len(layers) // 2, min(len(layers) - 1, 3 * len(layers) // 4)]
            layers_to_plot = [layers[i] for i in indices if 0 <= i < len(layers)]

        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5 * len(layers_to_plot), 4), sharey=True)
        if len(layers_to_plot) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers_to_plot):
            rq_vals = np.array(rq_snapshot.get(layer, []))
            s_vals = np.array(synergy_snapshot.get(layer, []))
            if rq_vals.size == 0 or s_vals.size == 0:
                continue

            ax.scatter(rq_vals, s_vals, s=20, alpha=0.5, c="#34495e", edgecolors="none")
            ax.set_title(layer)
            ax.set_xlabel("Rayleigh Quotient")
            if ax == axes[0]:
                ax.set_ylabel("PID Synergy (logits)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved alignment vs synergy plot to {save_path}")
        plt.close()

    def plot_ablation_summary(
        self,
        df: pd.DataFrame,
        hue: str = "Variant",
        value_col: str = "Perplexity",
        sparsity_col: str = "Sparsity",
        min_sparsity: float = 0.4,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Ablation Study at High Sparsity",
    ) -> None:
        """
        Bar plot summarizing ablation variants at high sparsity.
        """
        if df.empty:
            logger.warning("plot_ablation_summary: empty dataframe.")
            return

        df_filtered = df[df[sparsity_col] >= min_sparsity]
        if df_filtered.empty:
            logger.warning("plot_ablation_summary: no entries above min_sparsity.")
            return

        plt.figure(figsize=(8, 4))
        if HAS_SEABORN:
            sns.barplot(data=df_filtered, x=sparsity_col, y=value_col, hue=hue, palette="Dark2")
        else:
            for idx, (variant, subdf) in enumerate(df_filtered.groupby(hue)):
                plt.bar(
                    subdf[sparsity_col] + idx * 0.01,
                    subdf[value_col],
                    width=0.01,
                    label=variant,
                )

        plt.title(title)
        plt.xlabel("Sparsity")
        plt.ylabel(value_col)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.legend(title=hue)

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved ablation summary plot to {save_path}")
        plt.close()

    def _plot_metric_comparison(self, ax, results, metric, title):
        """Helper to plot metric comparison."""
        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())
            values = [strategy_results[s].get(metric, 0) for s in sparsities]

            color = self.strategy_colors.get(strategy, None)
            ax.plot(sparsities, values, "o-", label=strategy, linewidth=2.5, markersize=8, color=color)

        ax.set_xlabel("Sparsity Level")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_efficiency(self, ax, results):
        """Plot efficiency (accuracy retention vs sparsity)."""
        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())

            # Get initial accuracy (at 0 sparsity or lowest sparsity)
            initial_acc = strategy_results.get(0, strategy_results[sparsities[0]]).get("accuracy", 100)

            # Calculate accuracy retention
            retentions = []
            for s in sparsities:
                current_acc = strategy_results[s].get("accuracy", 0)
                retention = (current_acc / initial_acc) * 100 if initial_acc > 0 else 0
                retentions.append(retention)

            color = self.strategy_colors.get(strategy, None)
            ax.plot(sparsities, retentions, "o-", label=strategy, linewidth=2.5, markersize=8, color=color)

        ax.set_xlabel("Sparsity Level")
        ax.set_ylabel("Accuracy Retention (%)")
        ax.set_title("Pruning Efficiency: Accuracy Retention vs Sparsity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="90% retention")

    def _plot_strategy_ranking(self, ax, results):
        """Create strategy ranking based on area under accuracy curve."""
        rankings = {}

        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())
            accuracies = [strategy_results[s].get("accuracy", 0) for s in sparsities]

            # Calculate area under curve (simple trapezoidal)
            if len(sparsities) > 1:
                auc = np.trapz(accuracies, sparsities)
                rankings[strategy] = auc

        # Sort and plot
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        strategies, scores = zip(*sorted_strategies)

        colors = [self.strategy_colors.get(s, "gray") for s in strategies]
        bars = ax.barh(range(len(strategies)), scores, color=colors, alpha=0.7)

        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_xlabel("AUC Score")
        ax.set_title("Strategy Ranking (by AUC)")
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{score:.2f}", ha="left", va="center", fontsize=9)

    def _plot_sparsity_distribution(self, ax, results):
        """Plot distribution of final sparsities achieved."""
        # This would be more relevant with layer-wise data
        # For now, show target vs achieved sparsity
        ax.text(0.5, 0.5, "Sparsity Distribution\n(Requires layer-wise data)", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_summary_stats(self, ax, results):
        """Create summary statistics table."""
        summary_data = []

        for strategy in results:
            # Get performance at key sparsity levels
            perf_50 = results[strategy].get(0.5, {}).get("accuracy", "N/A")
            perf_70 = results[strategy].get(0.7, {}).get("accuracy", "N/A")
            perf_90 = results[strategy].get(0.9, {}).get("accuracy", "N/A")

            summary_data.append([strategy, f"{perf_50:.1f}%", f"{perf_70:.1f}%", f"{perf_90:.1f}%"])

        # Create table
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=summary_data, colLabels=["Strategy", "50% Sparse", "70% Sparse", "90% Sparse"], cellLoc="center", loc="center")

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor("#4CAF50")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax.set_title("Accuracy at Key Sparsity Levels", pad=20)


# ==============================================================================
# Unified Pruning Comparison Functions (for both Vision and LLM experiments)
# ==============================================================================

# Standard colors for pruning methods
PRUNING_METHOD_COLORS = {
    # Standard methods
    "random": "#95a5a6",
    "magnitude": "#e74c3c",
    "taylor": "#3498db",
    "gradient": "#ff7f0e",
    "fisher": "#2ca02c",
    # Cluster/composite methods
    "composite": "#9b59b6",
    "cluster_aware": "#2ecc71",
    "rq_only": "#f39c12",
    # LLM-specific methods
    "wanda": "#1abc9c",
    "sparsegpt": "#e91e63",
    "scar": "#00bcd4",
    # Low/high variants
    "magnitude_low": "#e74c3c",
    "magnitude_high": "#c0392b",
    "gradient_low": "#ff7f0e",
    "gradient_high": "#d35400",
    "scar_low": "#00bcd4",
    "scar_high": "#0097a7",
    # Network slimming
    "network_slimming": "#8e44ad",
    "geometric_median": "#16a085",
    "hrank": "#7f8c8d",
    "chip": "#16a085",
    # Activation-based metrics
    "activation_l2_norm": "#e74c3c",  # Same as magnitude (they're aliases)
    # Weight-only magnitude (channel-group)
    "weight_magnitude": "#e74c3c",
    "weight_magnitude_low": "#e74c3c",
    "weight_magnitude_high": "#c0392b",
    "activation_mean": "#c0392b",
    "activation_variance": "#a93226",
    # Generalized importance (no outlier assumption)
    "generalized_importance": "#27ae60",  # Green - distinct from supernode methods
    "neighborhood_redundancy": "#2ecc71",
    # Supernode/connectivity methods
    "supernode_protection_score": "#3498db",  # Blue
    "supernode_connectivity_score": "#2980b9",
    "directed_redundancy": "#1abc9c",
    "cross_layer_importance": "#9b59b6",  # Purple
    "within_layer_importance": "#8e44ad",
    # SCAR metrics
    "scar_loss_proxy": "#e67e22",  # Orange
    "scar_activation_power": "#d35400",
    "scar_taylor": "#f39c12",
    "scar_curvature": "#f1c40f",
    # Rayleigh quotient
    "rayleigh_quotient": "#3498db",
    "gaussian_mi_analytic": "#2ecc71",
}

PRUNING_METHOD_MARKERS = {
    "random": "o",
    "magnitude": "s",
    "weight_magnitude": "s",
    "taylor": "^",
    "gradient": "d",
    "composite": "p",
    "cluster_aware": "*",
    "wanda": "v",
    "sparsegpt": "<",
    "scar": ">",
}


def plot_unified_pruning_comparison(
    results: Dict[str, Dict[float, Dict[str, Any]]],
    baseline_value: Optional[float] = None,
    metric: str = "accuracy",
    higher_is_better: bool = True,
    title: str = "Pruning Method Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 7),
    show_before_ft: bool = False,
    x_as_percentage: bool = True,
) -> "plt.Figure":
    """
    Unified pruning comparison plot that works for both vision and LLM experiments.

    Args:
        results: Dict mapping method_name -> {sparsity -> {metric: value, ...}}
                 Supports both 'accuracy' (vision) and 'perplexity' (LLM) metrics.
                 Can include 'accuracy_before_ft', 'accuracy_after_ft', 'perplexity', etc.
        baseline_value: Baseline (unpruned) metric value
        metric: Which metric to plot ('accuracy', 'perplexity', 'loss', etc.)
        higher_is_better: Whether higher metric values are better (True for accuracy, False for perplexity)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        show_before_ft: Whether to show before-fine-tuning values (dashed lines)
        x_as_percentage: Whether to display x-axis as percentage

    Returns:
        Matplotlib Figure

    Example:
        >>> results = {
        ...     'magnitude': {0.3: {'accuracy_after_ft': 0.85}, 0.5: {'accuracy_after_ft': 0.78}},
        ...     'cluster_aware': {0.3: {'accuracy_after_ft': 0.88}, 0.5: {'accuracy_after_ft': 0.82}},
        ... }
        >>> plot_unified_pruning_comparison(results, baseline_value=0.92, metric='accuracy')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine the actual metric key to use
    # Support multiple naming conventions
    metric_keys = {
        "accuracy": ["accuracy_after_ft", "accuracy", "acc"],
        "perplexity": ["perplexity", "ppl"],
        "loss": ["loss", "test_loss"],
    }

    for method, method_results in results.items():
        if not method_results:
            continue

        sparsities = sorted([s for s in method_results.keys() if isinstance(s, (int, float))])
        values = []
        values_before = []

        for sparsity in sparsities:
            data = method_results[sparsity]
            if isinstance(data, dict):
                # Try different key names
                value = None
                for key in metric_keys.get(metric, [metric]):
                    if key in data:
                        value = data[key]
                        break

                if value is None and "error" not in data:
                    # Try any key containing the metric name
                    for k, v in data.items():
                        if metric in k.lower() and isinstance(v, (int, float)):
                            value = v
                            break

                values.append(value)

                # Check for before-fine-tuning value
                if show_before_ft:
                    before_key = f"{metric}_before_ft" if metric != "accuracy" else "accuracy_before_ft"
                    values_before.append(data.get(before_key))
            else:
                values.append(data if isinstance(data, (int, float)) else None)
                values_before.append(None)

        # Filter out None values
        valid_data = [(s, v) for s, v in zip(sparsities, values) if v is not None]
        if not valid_data:
            continue

        valid_sparsities, valid_values = zip(*valid_data)

        # Convert to percentage for display
        if metric == "accuracy" and max(valid_values) <= 1.0:
            valid_values = [v * 100 for v in valid_values]

        x_values = [s * 100 for s in valid_sparsities] if x_as_percentage else list(valid_sparsities)

        # Get style
        color = PRUNING_METHOD_COLORS.get(method.lower(), None)
        marker = PRUNING_METHOD_MARKERS.get(method.lower().split("_")[0], "o")
        label = method.replace("_", " ").title()

        # Plot main line (after fine-tuning)
        ax.plot(x_values, valid_values, marker=marker, color=color, label=label, linewidth=2.5, markersize=8)

        # Plot before fine-tuning (dashed) if requested
        if show_before_ft:
            valid_before = [(s * 100 if x_as_percentage else s, v) for s, v in zip(sparsities, values_before) if v is not None]
            if valid_before:
                x_before, y_before = zip(*valid_before)
                if metric == "accuracy" and max(y_before) <= 1.0:
                    y_before = [v * 100 for v in y_before]
                ax.plot(x_before, y_before, linestyle="--", color=color, alpha=0.5, linewidth=1.5)

    # Add baseline
    if baseline_value is not None:
        baseline_display = baseline_value * 100 if metric == "accuracy" and baseline_value <= 1.0 else baseline_value
        ax.axhline(
            y=baseline_display,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f'Unpruned ({baseline_display:.1f}{"%" if metric == "accuracy" else ""})',
        )

    # Labels and formatting
    xlabel = "Sparsity (%)" if x_as_percentage else "Sparsity"
    ax.set_xlabel(xlabel, fontsize=12)

    ylabel = metric.replace("_", " ").title()
    if metric == "accuracy":
        ylabel += " (%)"
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left" if higher_is_better else "upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved unified pruning comparison to {save_path}")

    return fig


def plot_pruning_accuracy_loss_grid(
    results: Dict[str, Dict[float, Dict[str, float]]],
    baseline_acc: Optional[float] = None,
    baseline_loss: Optional[float] = None,
    title: str = "Pruning Analysis",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> "plt.Figure":
    """
    Plot pruning results showing both accuracy and loss in a grid.

    Args:
        results: Dict mapping method -> {sparsity -> {'accuracy': v, 'loss': v}}
        baseline_acc: Baseline accuracy
        baseline_loss: Baseline loss
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for method, method_results in results.items():
        if not method_results:
            continue

        sparsities = sorted([s for s in method_results.keys() if isinstance(s, (int, float))])
        accuracies = []
        losses = []

        for s in sparsities:
            data = method_results[s]
            if isinstance(data, dict):
                acc = data.get("accuracy_after_ft") or data.get("accuracy")
                loss = data.get("loss") or data.get("test_loss")
                accuracies.append(acc * 100 if acc and acc <= 1.0 else acc)
                losses.append(loss)
            else:
                accuracies.append(None)
                losses.append(None)

        color = PRUNING_METHOD_COLORS.get(method.lower(), None)
        marker = PRUNING_METHOD_MARKERS.get(method.lower().split("_")[0], "o")
        label = method.replace("_", " ").title()

        # Plot accuracy
        valid_acc = [(s * 100, a) for s, a in zip(sparsities, accuracies) if a is not None]
        if valid_acc:
            x, y = zip(*valid_acc)
            ax1.plot(x, y, marker=marker, color=color, label=label, linewidth=2, markersize=7)

        # Plot loss
        valid_loss = [(s * 100, loss_val) for s, loss_val in zip(sparsities, losses) if loss_val is not None]
        if valid_loss:
            x, y = zip(*valid_loss)
            ax2.plot(x, y, marker=marker, color=color, label=label, linewidth=2, markersize=7)

    # Baselines
    if baseline_acc is not None:
        baseline_acc_display = baseline_acc * 100 if baseline_acc <= 1.0 else baseline_acc
        ax1.axhline(y=baseline_acc_display, color="gray", linestyle="--", linewidth=1.5, label="Unpruned")
    if baseline_loss is not None:
        ax2.axhline(y=baseline_loss, color="gray", linestyle="--", linewidth=1.5, label="Unpruned")

    # Format axes
    ax1.set_xlabel("Sparsity (%)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Sparsity", fontsize=13)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Sparsity (%)", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Loss vs Sparsity", fontsize=13)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved pruning accuracy/loss grid to {save_path}")

    return fig


def plot_pruning_recovery_chart(
    results: Dict[str, Dict[float, Dict[str, float]]],
    baseline_value: float,
    metric: str = "accuracy",
    title: str = "Accuracy Recovery After Pruning",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """
    Plot the percentage of original performance retained after pruning.

    Args:
        results: Dict mapping method -> {sparsity -> {metric: value}}
        baseline_value: Baseline (unpruned) value
        metric: Which metric to use
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for method, method_results in results.items():
        if not method_results:
            continue

        sparsities = sorted([s for s in method_results.keys() if isinstance(s, (int, float))])
        recoveries = []

        for s in sparsities:
            data = method_results[s]
            if isinstance(data, dict):
                value = data.get(f"{metric}_after_ft") or data.get(metric)
                if value is not None and baseline_value > 0:
                    recovery = (value / baseline_value) * 100
                    recoveries.append(recovery)
                else:
                    recoveries.append(None)
            else:
                recoveries.append(None)

        valid_data = [(s * 100, r) for s, r in zip(sparsities, recoveries) if r is not None]
        if not valid_data:
            continue

        x, y = zip(*valid_data)

        color = PRUNING_METHOD_COLORS.get(method.lower(), None)
        marker = PRUNING_METHOD_MARKERS.get(method.lower().split("_")[0], "o")
        label = method.replace("_", " ").title()

        ax.plot(x, y, marker=marker, color=color, label=label, linewidth=2.5, markersize=8)

    # Reference lines
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1.5, label="100% (Unpruned)")
    ax.axhline(y=90, color="green", linestyle=":", linewidth=1, alpha=0.7, label="90% Threshold")

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel(f"{metric.title()} Recovery (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([50, 105])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved pruning recovery chart to {save_path}")

    return fig


# ==============================================================================
# NEW: Bar Charts for Clear Method Comparison
# ==============================================================================


def plot_pruning_bar_comparison(
    results: Dict[str, Dict[float, Dict[str, Any]]],
    baseline_value: Optional[float] = None,
    target_sparsity: float = 0.5,
    metric: str = "accuracy",
    show_before_ft: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 7),
    sort_by_performance: bool = True,
) -> "plt.Figure":
    """
    Create a clear bar chart comparing pruning methods at a specific sparsity level.

    Args:
        results: Dict mapping method -> {sparsity -> {metric: value}}
        baseline_value: Baseline (unpruned) value
        target_sparsity: Sparsity level to compare at (e.g., 0.5 for 50%)
        metric: Which metric to plot
        show_before_ft: Whether to also show before-fine-tuning values
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        sort_by_performance: Whether to sort bars by performance

    Returns:
        Matplotlib Figure
    """
    methods = []
    after_ft_values = []
    before_ft_values = []

    for method, method_results in results.items():
        sparsities = [s for s in method_results.keys() if isinstance(s, (int, float))]
        if not sparsities:
            continue

        closest = min(sparsities, key=lambda x: abs(x - target_sparsity))
        if abs(closest - target_sparsity) > 0.15:
            continue

        data = method_results[closest]
        if isinstance(data, dict) and "error" not in data:
            after_val = data.get("accuracy_after_ft") or data.get("accuracy")
            before_val = data.get("accuracy_before_ft")

            if after_val is not None:
                methods.append(method)
                if metric == "accuracy" and after_val <= 1.0:
                    after_val *= 100
                    if before_val is not None:
                        before_val *= 100
                after_ft_values.append(after_val)
                before_ft_values.append(before_val)

    if not methods:
        logger.warning(f"No valid data found for sparsity {target_sparsity}")
        return plt.figure()

    if sort_by_performance:
        sorted_data = sorted(zip(methods, after_ft_values, before_ft_values), key=lambda x: x[1], reverse=True)
        methods, after_ft_values, before_ft_values = zip(*sorted_data)
        methods, after_ft_values, before_ft_values = list(methods), list(after_ft_values), list(before_ft_values)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(methods))
    width = 0.35 if show_before_ft and any(v is not None for v in before_ft_values) else 0.6

    # Colors based on method category
    colors = []
    for m in methods:
        if "random" in m.lower():
            colors.append("#95a5a6")
        elif "magnitude" in m.lower() and "plus" not in m.lower() and "minus" not in m.lower():
            colors.append("#e74c3c")
        elif "composite" in m.lower() or "cluster" in m.lower():
            colors.append("#2ecc71")
        elif "rq" in m.lower():
            colors.append("#3498db")
        elif "redundancy" in m.lower():
            colors.append("#9b59b6")
        elif "synergy" in m.lower():
            colors.append("#f39c12")
        else:
            colors.append("#1abc9c")

    if show_before_ft and any(v is not None for v in before_ft_values):
        ax.bar(
            x - width / 2,
            [v if v is not None else 0 for v in before_ft_values],
            width,
            label="Before Fine-tune",
            alpha=0.5,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        bars_after = ax.bar(x + width / 2, after_ft_values, width, label="After Fine-tune", color=colors, edgecolor="black", linewidth=1)
    else:
        bars_after = ax.bar(x, after_ft_values, width, color=colors, edgecolor="black", linewidth=1)

    if baseline_value is not None:
        baseline_display = baseline_value * 100 if baseline_value <= 1.0 else baseline_value
        ax.axhline(y=baseline_display, color="red", linestyle="--", linewidth=2, label=f"Unpruned: {baseline_display:.1f}%")

    for i, val in enumerate(after_ft_values):
        bar = bars_after[i]
        height = bar.get_height()
        ax.annotate(
            f"{val:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Pruning Method", fontsize=12)
    ax.set_ylabel(f"{metric.title()} (%)", fontsize=12)
    title = title or f"Pruning Method Comparison at {int(target_sparsity*100)}% Sparsity"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], rotation=45, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    y_min = max(0, min(after_ft_values) - 10)
    ax.set_ylim([y_min, 105])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved pruning bar comparison to {save_path}")

    return fig


def plot_pruning_heatmap(
    results: Dict[str, Dict[float, Dict[str, Any]]],
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
    annotate: bool = True,
) -> "plt.Figure":
    """
    Create a heatmap showing accuracy for all method-sparsity combinations.
    """
    all_sparsities = set()
    for method_results in results.values():
        all_sparsities.update(s for s in method_results.keys() if isinstance(s, (int, float)))
    sparsities = sorted(all_sparsities)

    methods = list(results.keys())
    data = np.full((len(methods), len(sparsities)), np.nan)

    for i, method in enumerate(methods):
        method_results = results[method]
        for j, s in enumerate(sparsities):
            if s in method_results:
                d = method_results[s]
                if isinstance(d, dict) and "error" not in d:
                    val = d.get("accuracy_after_ft") or d.get("accuracy")
                    if val is not None:
                        data[i, j] = val * 100 if val <= 1.0 else val

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{metric.title()} (%)", fontsize=12)

    ax.set_xticks(np.arange(len(sparsities)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([f"{int(s*100)}%" for s in sparsities], fontsize=10)
    ax.set_yticklabels([m.replace("_", " ") for m in methods], fontsize=10)

    if annotate:
        for i in range(len(methods)):
            for j in range(len(sparsities)):
                if not np.isnan(data[i, j]):
                    text_color = "white" if data[i, j] < 50 else "black"
                    ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Sparsity Level", fontsize=12)
    ax.set_ylabel("Pruning Method", fontsize=12)
    title = title or "Pruning Performance Heatmap"
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved pruning heatmap to {save_path}")

    return fig


def plot_pruning_ranking(
    results: Dict[str, Dict[float, Dict[str, Any]]],
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> "plt.Figure":
    """
    Create a ranking plot showing methods ordered by average performance.
    """
    method_scores = {}

    for method, method_results in results.items():
        values = []
        for s, data in method_results.items():
            if isinstance(s, (int, float)) and isinstance(data, dict):
                val = data.get("accuracy_after_ft") or data.get("accuracy")
                if val is not None:
                    values.append(val * 100 if val <= 1.0 else val)

        if values:
            method_scores[method] = {
                "mean": np.mean(values),
                "std": np.std(values),
            }

    if not method_scores:
        return plt.figure()

    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]["mean"], reverse=True)
    methods = [m for m, _ in sorted_methods]
    means = [d["mean"] for _, d in sorted_methods]
    stds = [d["std"] for _, d in sorted_methods]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(methods))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(methods)))[::-1]

    bars = ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)

    for bar, mean, std in zip(bars, means, stds):
        width = bar.get_width()
        ax.text(width + std + 1, bar.get_y() + bar.get_height() / 2, f"{mean:.1f}±{std:.1f}%", va="center", fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace("_", " ") for m in methods], fontsize=11)
    ax.set_xlabel(f"Average {metric.title()} (%)", fontsize=12)
    title = title or "Pruning Method Ranking"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    for i in range(len(sorted_methods)):
        ax.text(-2, i, f"#{i+1}", va="center", ha="right", fontsize=11, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ranking plot to {save_path}")

    return fig
