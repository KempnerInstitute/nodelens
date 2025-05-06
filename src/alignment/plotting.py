"""
Plotting utilities for visualization of experiment results.

This module provides functions for visualizing experimental results from
alignment studies, including progressive dropout analysis with different
pruning strategies and other alignment metrics.
"""

import logging
import math
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import json

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Set default figure size
plt.rcParams["figure.figsize"] = (10, 6)
# Use SVG backend for better quality
plt.rcParams["figure.dpi"] = 120


def plot_pruning_experiments(
    results: Dict[str, Dict[str, List[Tuple[float, float]]]],
    metric_names: Dict[str, str],
    output_dir: str,
    filename_prefix: str = "dropout",
    separate_plots: bool = False,
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    fig_size: Tuple[int, int] = (10, 6),
) -> List[Figure]:
    """
    Plot accuracy vs. prune fraction for different metrics and strategies.
    
    Args:
        results: Dict[metric_name, Dict[strategy, [(prune_fraction, accuracy)]]]
        metric_names: Names to display for each metric
        output_dir: Directory to save plots
        filename_prefix: Prefix for filename
        separate_plots: If True, create separate plots for each metric
        title_suffix: Additional text to append to plot title
        ylim: Optional y-axis limits
        fig_size: Figure size (width, height) in inches
        
    Returns:
        List of created figures
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if separate_plots:
        return _plot_separate_figures(
            results, metric_names, output_dir, filename_prefix, 
            title_suffix, ylim, fig_size
        )
    else:
        return _plot_combined_figure(
            results, metric_names, output_dir, filename_prefix, 
            title_suffix, ylim, fig_size
        )


def _plot_combined_figure(
    results: Dict[str, Dict[str, List[Tuple[float, float]]]],
    metric_names: Dict[str, str],
    output_dir: str,
    filename_prefix: str,
    title_suffix: str,
    ylim: Optional[Tuple[float, float]],
    fig_size: Tuple[int, int],
) -> List[Figure]:
    """
    Create a combined plot with all metrics and strategies.
    """
    # Define colors and markers for different strategies
    colors = {
        "random": "gray",
        "high_rq": "green",
        "low_rq": "red",
        "high_mi": "blue",
        "low_mi": "purple",
    }
    
    markers = {
        "random": "o",
        "high_rq": "^",
        "low_rq": "v",
        "high_mi": "s",
        "low_mi": "D",
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot each metric and strategy
    for metric_name, strategies in results.items():
        for strategy, data_points in strategies.items():
            # Extract x and y values
            x = [point[0] for point in data_points]
            y = [point[1] for point in data_points]
            
            # Get display name
            display_name = f"{metric_names.get(metric_name, metric_name)} {strategy}"
            
            # Get color and marker
            color = colors.get(strategy, "black")
            marker = markers.get(strategy, ".")
            
            # Plot
            ax.plot(x, y, marker=marker, label=display_name, color=color)
    
    # Set title and labels
    title = f"Accuracy vs. Prune Fraction{title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Prune Fraction")
    ax.set_ylabel("Accuracy (%)")
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Save figure
    filename = f"{filename_prefix}_combined.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    
    logger.info(f"Saved combined plot to {filepath}")
    
    return [fig]


def _plot_separate_figures(
    results: Dict[str, Dict[str, List[Tuple[float, float]]]],
    metric_names: Dict[str, str],
    output_dir: str,
    filename_prefix: str,
    title_suffix: str,
    ylim: Optional[Tuple[float, float]],
    fig_size: Tuple[int, int],
) -> List[Figure]:
    """
    Create separate plots for each metric.
    """
    # Define colors and markers for different strategies
    colors = {
        "random": "gray",
        "high_rq": "green",
        "low_rq": "red",
        "high_mi": "blue",
        "low_mi": "purple",
    }
    
    markers = {
        "random": "o",
        "high_rq": "^",
        "low_rq": "v",
        "high_mi": "s",
        "low_mi": "D",
    }
    
    figures = []
    
    # Create a separate plot for each metric
    for metric_name, strategies in results.items():
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot each strategy
        for strategy, data_points in strategies.items():
            # Extract x and y values
            x = [point[0] for point in data_points]
            y = [point[1] for point in data_points]
            
            # Get display name
            display_name = strategy
            
            # Get color and marker
            color = colors.get(strategy, "black")
            marker = markers.get(strategy, ".")
            
            # Plot
            ax.plot(x, y, marker=marker, label=display_name, color=color)
        
        # Set title and labels
        metric_display = metric_names.get(metric_name, metric_name)
        title = f"{metric_display}: Accuracy vs. Prune Fraction{title_suffix}"
        ax.set_title(title)
        ax.set_xlabel("Prune Fraction")
        ax.set_ylabel("Accuracy (%)")
        
        # Set y-axis limits if specified
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Save figure
        filename = f"{filename_prefix}_{metric_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches="tight")
        
        logger.info(f"Saved {metric_name} plot to {filepath}")
        
        figures.append(fig)
    
    return figures


def plot_per_layer_independent(
    results: Dict[str, Dict[int, Tuple[float, float]]],
    layer_names: Optional[Dict[int, str]] = None,
    output_dir: str = ".",
    filename_prefix: str = "per_layer",
    title_suffix: str = "",
    ylim: Optional[Tuple[float, float]] = None,
    fig_size: Tuple[int, int] = (12, 8),
) -> List[Figure]:
    """
    Plot accuracy when pruning each layer independently.
    
    Args:
        results: Dict[metric_name, Dict[layer_idx, (prune_fraction, accuracy)]]
        layer_names: Optional mapping of layer indices to display names
        output_dir: Directory to save plots
        filename_prefix: Prefix for filename
        title_suffix: Additional text to append to plot title
        ylim: Optional y-axis limits
        fig_size: Figure size (width, height) in inches
        
    Returns:
        List of created figures
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define colors for different metrics
    colors = {
        "random": "gray",
        "rq": "green",
        "mi": "blue",
        "high_rq": "green",
        "low_rq": "red",
        "high_mi": "blue", 
        "low_mi": "purple",
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # For each metric, create a grouped bar chart
    metrics = list(results.keys())
    num_metrics = len(metrics)
    
    # Get all unique layer indices
    all_layers = set()
    for metric_data in results.values():
        all_layers.update(metric_data.keys())
    all_layers = sorted(all_layers)
    
    # Bar width and positions
    bar_width = 0.8 / num_metrics
    
    # Plot each metric
    for i, metric_name in enumerate(metrics):
        layer_results = results[metric_name]
        
        # Positions of the bars
        positions = np.arange(len(all_layers)) + (i - num_metrics/2 + 0.5) * bar_width
        
        # Heights of the bars (accuracies)
        heights = [layer_results.get(layer_idx, (0, 0))[1] for layer_idx in all_layers]
        
        # Plot bars
        color = colors.get(metric_name, "black")
        ax.bar(positions, heights, bar_width, label=metric_name, color=color, alpha=0.7)
    
    # Set x-axis ticks and labels
    if layer_names is None:
        layer_names = {idx: f"Layer {idx}" for idx in all_layers}
        
    ax.set_xticks(np.arange(len(all_layers)))
    ax.set_xticklabels([layer_names.get(idx, f"Layer {idx}") for idx in all_layers], rotation=45, ha="right")
    
    # Set title and labels
    title = f"Per-Layer Pruning: Accuracy by Layer{title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (%)")
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # Add legend
    ax.legend()
    
    # Save figure
    filename = f"{filename_prefix}_per_layer.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    
    logger.info(f"Saved per-layer plot to {filepath}")
    
    return [fig]


def plot_dropout_results(
    results: Dict,
    plot_dir: str,
    pruning_mode: str = "global",
    dropout_mode: str = "scaled",
    title_prefix: str = "Dropout"
) -> Dict[str, str]:
    """
    Plot dropout experiment results using a style similar to the preref implementation.
    
    Args:
        results: Results dictionary from progressive_dropout
        plot_dir: Directory to save plots to
        pruning_mode: Pruning mode used in the experiment
        dropout_mode: Dropout mode used in the experiment
        title_prefix: Prefix for plot titles
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Check for error in results
    if "error" in results:
        logger.error(f"Cannot plot results due to error: {results['error']}")
        return {}
    
    # Extract dropout fractions
    dropout_fractions = results.get("dropout_fractions", [])
    if not isinstance(dropout_fractions, (list, np.ndarray)) or len(dropout_fractions) == 0:
        logger.error("No dropout fractions found in results")
        return {}
    
    # Set up figure parameters
    strategies = ["high_rq", "low_rq", "random"]
    colors = {"high_rq": "blue", "low_rq": "red", "random": "green"}
    linestyles = {"high_rq": "-", "low_rq": "-", "random": "-"}
    markers = {"high_rq": "o", "low_rq": "s", "random": "^"}
    strategy_labels = {"high_rq": "High RQ", "low_rq": "Low RQ", "random": "Random"}
    
    # Create dictionary to store file paths
    file_paths = {}
    
    # ----------------
    # Plot Accuracy (single figure with all strategies)
    # ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        accs = results.get("accuracies", {}).get(strategy)
        if not isinstance(accs, (list, np.ndarray)) or len(accs) == 0:
            logger.warning(f"No accuracy data for strategy {strategy}")
            continue
        
        ax.plot(
            dropout_fractions,
            accs,
            marker=markers[strategy],
            linestyle=linestyles[strategy],
            color=colors[strategy],
            linewidth=2,
            markersize=6,
            label=strategy_labels[strategy]
        )
    
    ax.set_xlabel('Dropout Fraction', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'{title_prefix} - {pruning_mode.replace("_", " ").title()} Pruning ({dropout_mode})', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 100])  # Fixed y-axis for accuracy
    
    accuracy_file = os.path.join(plot_dir, f"{title_prefix.lower().replace(' ', '_')}_{pruning_mode}_accuracy.png")
    plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    file_paths["accuracy"] = accuracy_file
    logger.info(f"Saved accuracy plot to {accuracy_file}")
    
    # ----------------
    # Plot Loss (single figure with all strategies)
    # ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        losses = results.get("losses", {}).get(strategy)
        if not isinstance(losses, (list, np.ndarray)) or len(losses) == 0:
            logger.warning(f"No loss data for strategy {strategy}")
            continue
        
        ax.plot(
            dropout_fractions,
            losses,
            marker=markers[strategy],
            linestyle=linestyles[strategy],
            color=colors[strategy],
            linewidth=2,
            markersize=6,
            label=strategy_labels[strategy]
        )
    
    ax.set_xlabel('Dropout Fraction', fontsize=14)
    ax.set_ylabel('Loss (%)', fontsize=14)
    ax.set_title(f'{title_prefix} Loss - {pruning_mode.replace("_", " ").title()} Pruning ({dropout_mode})', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    loss_file = os.path.join(plot_dir, f"{title_prefix.lower().replace(' ', '_')}_{pruning_mode}_loss.png")
    plt.savefig(loss_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    file_paths["loss"] = loss_file
    logger.info(f"Saved loss plot to {loss_file}")
    
    # ----------------
    # Plot Individual Strategy Comparisons
    # ----------------
    for strategy in strategies:
        accs = results.get("accuracies", {}).get(strategy)
        if not isinstance(accs, (list, np.ndarray)) or len(accs) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            dropout_fractions,
            accs,
            marker=markers[strategy],
            linestyle=linestyles[strategy],
            color=colors[strategy],
            linewidth=2,
            markersize=6,
            label=f"{strategy_labels[strategy]} Accuracy"
        )
        
        # Add loss to the same plot if available
        losses = results.get("losses", {}).get(strategy)
        if isinstance(losses, (list, np.ndarray)) and len(losses) > 0:
            ax.plot(
                dropout_fractions,
                losses,
                marker=markers[strategy],
                linestyle='--',
                color=colors[strategy],
                linewidth=2,
                markersize=6,
                label=f"{strategy_labels[strategy]} Loss"
            )
        
        ax.set_xlabel('Dropout Fraction', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.set_title(f'{title_prefix} - {strategy_labels[strategy]} ({dropout_mode})', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        
        strategy_file = os.path.join(plot_dir, f"{title_prefix.lower().replace(' ', '_')}_{pruning_mode}_{strategy}.png")
        plt.savefig(strategy_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        file_paths[f"{strategy}_comparison"] = strategy_file
        logger.info(f"Saved {strategy} comparison plot to {strategy_file}")
    
    # ----------------
    # Plot Alignment Values (if available)
    # ----------------
    alignment_values = results.get("alignment_values", {})
    if alignment_values:
        # Create one plot per layer
        for layer_idx in range(len(alignment_values.get("high_rq", [{}])[0] or [])):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for strategy in strategies:
                # Extract alignment values for this layer across dropout fractions
                if strategy in alignment_values:
                    layer_alignments = []
                    for frac_idx, alignment in enumerate(alignment_values[strategy]):
                        if alignment and layer_idx < len(alignment):
                            layer_alignments.append(alignment[layer_idx])
                        else:
                            layer_alignments.append(None)
                    
                    # Filter out None values
                    valid_fractions = []
                    valid_alignments = []
                    for frac_idx, alignment in enumerate(layer_alignments):
                        if alignment is not None:
                            valid_fractions.append(dropout_fractions[frac_idx])
                            valid_alignments.append(alignment)
                    
                    if valid_fractions and valid_alignments:
                        ax.plot(
                            valid_fractions,
                            valid_alignments,
                            marker=markers[strategy],
                            linestyle=linestyles[strategy],
                            color=colors[strategy],
                            linewidth=2,
                            markersize=6,
                            label=strategy_labels[strategy]
                        )
            
            ax.set_xlabel('Dropout Fraction', fontsize=14)
            ax.set_ylabel('Alignment Value', fontsize=14)
            ax.set_title(f'Layer {layer_idx+1} Alignment - {pruning_mode.replace("_", " ").title()}', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=12)
            
            alignment_file = os.path.join(plot_dir, f"{title_prefix.lower().replace(' ', '_')}_{pruning_mode}_layer{layer_idx+1}_alignment.png")
            plt.savefig(alignment_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            file_paths[f"layer{layer_idx+1}_alignment"] = alignment_file
            logger.info(f"Saved layer {layer_idx+1} alignment plot to {alignment_file}")
    
    # Save raw results as JSON for further analysis
    try:
        # Helper function to safely convert tensors and numpy arrays to lists
        def safe_convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_convert(item) for item in obj]
            else:
                return obj
        
        # Create a simplified version of results for JSON serialization
        json_results = {
            "dropout_fractions": safe_convert(dropout_fractions),
            "accuracies": safe_convert(results.get("accuracies", {})),
            "losses": safe_convert(results.get("losses", {})),
        }
        
        # Exclude alignment_values as they can be complex and not easily serializable
        
        json_path = os.path.join(plot_dir, f"{title_prefix.lower().replace(' ', '_')}_{pruning_mode}_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        file_paths["json_results"] = json_path
        logger.info(f"Saved raw results to {json_path}")
    except Exception as e:
        logger.error(f"Error saving JSON results: {str(e)}")
    
    return file_paths


def plot_experiment_summary(
    results: Dict,
    plot_dir: str
) -> Dict[str, str]:
    """
    Generate a comprehensive summary of experiment results.
    
    Args:
        results: Results dictionary from the experiment
        plot_dir: Directory to save plots to
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    # Create the output directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Prepare figure
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Panel 1: Configuration summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # Extract config info
    config = results.get("config", {})
    config_text = []
    
    if hasattr(config, "model") and hasattr(config.model, "model_name"):
        config_text.append(f"Model: {config.model.model_name}")
    
    if hasattr(config, "dataset") and hasattr(config.dataset, "dataset_name"):
        config_text.append(f"Dataset: {config.dataset.dataset_name}")
    
    if hasattr(config, "alignment") and hasattr(config.alignment, "metric"):
        config_text.append(f"Alignment Metric: {config.alignment.metric}")
    
    if hasattr(config, "alignment"):
        if hasattr(config.alignment, "dropout_min") and hasattr(config.alignment, "dropout_max"):
            config_text.append(f"Dropout Range: {config.alignment.dropout_min} to {config.alignment.dropout_max}")
        if hasattr(config.alignment, "dropout_steps"):
            config_text.append(f"Dropout Steps: {config.alignment.dropout_steps}")
    
    if hasattr(config, "extra"):
        if hasattr(config.extra, "dropout_mode"):
            config_text.append(f"Dropout Mode: {config.extra.dropout_mode}")
        if hasattr(config.extra, "dropout_pruning_mode"):
            config_text.append(f"Dropout Pruning Mode: {config.extra.dropout_pruning_mode}")
    
    ax1.text(0.05, 0.95, "\n".join(config_text), fontsize=11, 
             verticalalignment='top', horizontalalignment='left')
    ax1.set_title("Experiment Configuration", fontsize=14)
    
    # Panel 2: Progressive Dropout results if available
    ax2 = fig.add_subplot(gs[0, 1])
    prog_results = results.get("progressive_dropout", {})
    
    if "accuracies" in prog_results and "dropout_fractions" in prog_results:
        fractions = prog_results["dropout_fractions"]
        for strategy, color in [("high_rq", "blue"), ("low_rq", "red"), ("random", "green")]:
            if strategy in prog_results["accuracies"]:
                accs = prog_results["accuracies"][strategy]
                ax2.plot(
                    fractions, 
                    accs, 
                    marker='o', 
                    linestyle='-', 
                    color=color, 
                    label=strategy.replace('_', ' ').title()
                )
        
        ax2.set_xlabel("Dropout Fraction", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Progressive Dropout Results", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No Progressive Dropout Results", 
                fontsize=12, ha='center', va='center')
        ax2.axis('off')
    
    # Panel 3: Eigenvector Dropout results if available
    ax3 = fig.add_subplot(gs[1, 0])
    eig_results = results.get("eigenvector_dropout", {})
    
    if "accuracies" in eig_results and "dropout_fractions" in eig_results:
        fractions = eig_results["dropout_fractions"]
        if "eigenvector" in eig_results["accuracies"]:
            accs = eig_results["accuracies"]["eigenvector"]
            ax3.plot(
                fractions, 
                accs, 
                marker='o', 
                linestyle='-', 
                color='purple', 
                label="Eigenvector"
            )
            
            # Also add the high_rq from progressive dropout for comparison if available
            if "accuracies" in prog_results and "high_rq" in prog_results["accuracies"]:
                prog_fracs = prog_results["dropout_fractions"]
                prog_accs = prog_results["accuracies"]["high_rq"]
                # Only plot if the fractions match
                if len(prog_fracs) == len(fractions) and all(a == b for a, b in zip(prog_fracs, fractions)):
                    ax3.plot(
                        fractions,
                        prog_accs,
                        marker='s',
                        linestyle='--',
                        color='blue',
                        label="High RQ"
                    )
        
        ax3.set_xlabel("Dropout Fraction", fontsize=12)
        ax3.set_ylabel("Accuracy (%)", fontsize=12)
        ax3.set_title("Eigenvector Dropout Results", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No Eigenvector Dropout Results", 
                fontsize=12, ha='center', va='center')
        ax3.axis('off')
    
    # Panel 4: Alignment comparison or other metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Check both progressive and eigenvector results for alignment values
    alignment_data = None
    if "alignment_values" in prog_results and "high_rq" in prog_results["alignment_values"]:
        alignment_data = prog_results["alignment_values"]["high_rq"][0]
    elif "alignment_values" in eig_results and "eigenvector" in eig_results["alignment_values"]:
        alignment_data = eig_results["alignment_values"]["eigenvector"][0]
    
    if alignment_data and len(alignment_data) > 0:
        # Extract alignment values as floats
        alignment_values = []
        for val in alignment_data:
            if isinstance(val, torch.Tensor):
                alignment_values.append(val.item())
            else:
                alignment_values.append(float(val))
        
        # Create bar chart of alignment by layer
        x = np.arange(len(alignment_values))
        bars = ax4.bar(x, alignment_values, width=0.6, alpha=0.7)
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01,
                f'{alignment_values[i]:.3f}',
                ha='center', 
                va='bottom', 
                rotation=0, 
                fontsize=9
            )
        
        ax4.set_xlabel("Layer", fontsize=12)
        ax4.set_ylabel("Alignment Value", fontsize=12)
        ax4.set_title("Alignment by Layer", fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"Layer {i+1}" for i in range(len(alignment_values))])
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, "No Alignment Data", 
                fontsize=12, ha='center', va='center')
        ax4.axis('off')
    
    # Add an overall title
    title = "Experiment Summary"
    if hasattr(config, "model") and hasattr(config.model, "model_name"):
        if hasattr(config, "dataset") and hasattr(config.dataset, "dataset_name"):
            title += f": {config.model.model_name} on {config.dataset.dataset_name}"
        else:
            title += f": {config.model.model_name}"
    
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save the figure
    summary_file = os.path.join(plot_dir, "experiment_summary.png")
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved experiment summary to {summary_file}")
    
    return {"summary": summary_file} 