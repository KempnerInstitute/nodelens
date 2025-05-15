import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
import logging
import torch
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None # Ensure wandb is None if not available, for type hinting or checks

from alignment.config import Config, ExperimentConfig, WandbConfig # Config still used by Experiment class
from alignment.datasets import DataSet, load_dataset

# Set up logger
logger = logging.getLogger(__name__)

# Set preref style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def plot_dropout_results(
    results,
    save_dir=None,
    title_prefix="Dropout Results",
    pruning_mode="global_joint",
    dropout_mode="scaled",
    show=False
):
    """
    Create plots for progressive dropout results, showing accuracy vs dropout fraction.
    
    This enhanced version ensures error bars are shown for all strategies and properly handles empty data.
    
    Args:
        results: Results dictionary with accuracies, stds, losses and dropout_fractions
        save_dir: Directory to save plots to
        title_prefix: Prefix for plot titles
        pruning_mode: Pruning mode used in the experiment
        dropout_mode: Dropout mode used in the experiment
        show: Whether to display plots (vs just saving them)
        
    Returns:
        List of saved plot files
    """
    # Create directory if not exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create a more readable title from the pruning mode
    if pruning_mode == "global_joint":
        pruning_title = "Global Joint Pruning"
    elif pruning_mode == "layer_wise":
        pruning_title = "Layer-wise Pruning"
    elif pruning_mode == "layer_isolated":
        pruning_title = "Layer Isolated Pruning" 
    elif pruning_mode == "cascading_layer":
        pruning_title = "Cascading Layer Pruning"
    else:
        pruning_title = pruning_mode.replace("_", " ").title()
    
    # Check if results has the expected structure
    if not isinstance(results, dict) or "accuracies" not in results or "dropout_fractions" not in results:
        logger.error(f"Invalid results format: {type(results)}")
        return []
        
    # Get data from results
    dropout_fractions = results["dropout_fractions"]
    strategies = list(results["accuracies"].keys())
    
    # Check if we actually have data to plot
    has_data = False
    for strategy in strategies:
        if strategy in results["accuracies"] and len(results["accuracies"][strategy]) > 0:
            has_data = True
            break
    
    if not has_data:
        logger.error("No valid data found to plot!")
        return []
        
    # Clear any previous plots
    plt.clf()
    
    # Set strategy colors and markers
    colors = {"high_rq": "#d62728", "low_rq": "#2ca02c", "random": "#1f77b4"}
    markers = {"high_rq": "o", "low_rq": "s", "random": "^"}
    labels = {
        "high_rq": "Prune Highest Magnitude", 
        "low_rq": "Prune Lowest Magnitude", 
        "random": "Random Pruning"
    }
    
    # Create accuracy plot
    saved_files = []
    
    # Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the figure to avoid any previous plots
    
    plot_count = 0  # Track if we successfully plotted anything
    
    for strategy in strategies:
        if strategy in results["accuracies"] and len(results["accuracies"][strategy]) > 0:
            # Get accuracy and standard deviation
            accs = results["accuracies"][strategy]
            
            # Ensure we have standard deviations
            if "stds" in results and strategy in results["stds"] and len(results["stds"][strategy]) > 0:
                stds = results["stds"][strategy]
            else:
                stds = np.zeros_like(accs)  # Use zeros if no stds available
                
            # Make sure dropout_fractions and accs have the same length
            x_values = dropout_fractions
            if len(x_values) > len(accs):
                x_values = x_values[:len(accs)]
            elif len(x_values) < len(accs):
                accs = accs[:len(x_values)]
                stds = stds[:len(x_values)]
                
            # Plot with error bars
            try:
                plt.errorbar(
                    x_values,
                    accs,
                    yerr=stds,  # Include error bars
                    label=labels.get(strategy, strategy),
                    marker=markers.get(strategy, 'o'),
                    color=colors.get(strategy, 'black'),
                    capsize=4,  # Add caps to error bars
                    markersize=8,
                    linewidth=2
                )
                plot_count += 1
            except Exception as e:
                logger.error(f"Error plotting {strategy}: {str(e)}")
    
    # Only add labels and save if we plotted something
    if plot_count > 0:
        plt.title(f"{title_prefix}: Accuracy vs Dropout Fraction\n({pruning_title}, {dropout_mode} mode)", fontsize=14)
        plt.xlabel("Dropout Fraction", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        if save_dir:
            file_path = os.path.join(save_dir, f"accuracy_vs_dropout_{pruning_mode}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            saved_files.append(file_path)
            logger.info(f"Saved accuracy plot to {file_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    else:
        logger.error("No data could be plotted for accuracy!")
        plt.close()
    
    # Loss Plot (if available)
    if "losses" in results and any(len(results["losses"].get(s, [])) > 0 for s in strategies):
        plt.figure(figsize=(10, 6))
        plt.clf()  # Clear the figure
        
        plot_count = 0
        
        for strategy in strategies:
            if strategy in results["losses"] and len(results["losses"][strategy]) > 0:
                losses = results["losses"][strategy]
                
                # Ensure we have standard deviations (for losses we might not have them)
                if "loss_stds" in results and strategy in results["loss_stds"] and len(results["loss_stds"][strategy]) > 0:
                    loss_stds = results["loss_stds"][strategy]
                else:
                    loss_stds = None
                
                # Make sure dropout_fractions and losses have the same length
                x_values = dropout_fractions
                if len(x_values) > len(losses):
                    x_values = x_values[:len(losses)]
                elif len(x_values) < len(losses):
                    losses = losses[:len(x_values)]
                    if loss_stds is not None:
                        loss_stds = loss_stds[:len(x_values)]
                
                try:
                    if loss_stds:
                        plt.errorbar(
                            x_values,
                            losses,
                            yerr=loss_stds,
                            label=labels.get(strategy, strategy),
                            marker=markers.get(strategy, 'o'),
                            color=colors.get(strategy, 'black'),
                            capsize=4,
                            markersize=8,
                            linewidth=2
                        )
                    else:
                        plt.plot(
                            x_values,
                            losses,
                            label=labels.get(strategy, strategy),
                            marker=markers.get(strategy, 'o'),
                            color=colors.get(strategy, 'black'),
                            markersize=8,
                            linewidth=2
                        )
                    plot_count += 1
                except Exception as e:
                    logger.error(f"Error plotting loss for {strategy}: {str(e)}")
        
        # Only add labels and save if we plotted something
        if plot_count > 0:
            plt.title(f"{title_prefix}: Loss vs Dropout Fraction\n({pruning_title}, {dropout_mode} mode)", fontsize=14)
            plt.xlabel("Dropout Fraction", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            if save_dir:
                file_path = os.path.join(save_dir, f"loss_vs_dropout_{pruning_mode}.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                saved_files.append(file_path)
                logger.info(f"Saved loss plot to {file_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
        else:
            logger.error("No data could be plotted for loss!")
            plt.close()
    
    return saved_files

# Alias for backward compatibility
custom_plot_dropout = plot_dropout_results

def plot_experiment_summary(
    results: Dict,
    figure_path: Optional[str] = None,
    experiment_name: str = "Experiment Summary"
) -> Optional[str]:
    """
    Generate summary plots for alignment experiments.
    
    Args:
        results: Results dictionary containing both progressive and eigenvector dropout results
        figure_path: Path to save the figure
        experiment_name: Name for the experiment
        
    Returns:
        Path to the saved figure, or None if not saved
    """
    # Extract progressive dropout results if available
    prog_results = None
    if "progressive_dropout" in results:
        prog_results = results["progressive_dropout"]
        
    # Extract eigenvector dropout results if available
    eig_results = None
    if "eigenvector_dropout" in results:
        eig_results = results["eigenvector_dropout"]
    
    # Extract training history if available
    training_history = None
    if prog_results and "training_history" in prog_results:
        training_history = prog_results["training_history"]
    elif "training_history" in results:
        training_history = results["training_history"]
    
    # Create figure with 2x2 grid of subplots
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # Add super title
    fig.suptitle(experiment_name, fontsize=18, y=0.98)
    
    # Panel 1: Training history
    ax1 = fig.add_subplot(gs[0, :])
    
    if training_history and "train_acc" in training_history and len(training_history["train_acc"]) > 0:
        epochs = list(range(1, len(training_history["train_acc"]) + 1))
        
        # Plot training accuracy
        line1 = ax1.plot(epochs, training_history["train_acc"], 'o-', color="#1f77b4", linewidth=2, markersize=8, label="Train Accuracy")
        
        # Plot test accuracy
        if "test_acc" in training_history and len(training_history["test_acc"]) > 0:
            line2 = ax1.plot(epochs, training_history["test_acc"], 's-', color="#d62728", linewidth=2, markersize=8, label="Test Accuracy")
        
        # Set up left y-axis
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Accuracy (%)", fontsize=12)
        ax1.set_ylim([0, 100])
        ax1.set_title("Training History", fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Create a second y-axis for loss
        if "train_loss" in training_history and len(training_history["train_loss"]) > 0:
            ax1_right = ax1.twinx()
            
            # Plot training loss
            line3 = ax1_right.plot(epochs, training_history["train_loss"], '^--', color="#2ca02c", linewidth=1.5, markersize=6, label="Train Loss")
            
            # Plot test loss
            if "test_loss" in training_history and len(training_history["test_loss"]) > 0:
                line4 = ax1_right.plot(epochs, training_history["test_loss"], 'D--', color="#9467bd", linewidth=1.5, markersize=6, label="Test Loss")
            
            # Set up right y-axis
            ax1_right.set_ylabel("Loss", fontsize=12)
            
            # Combine legends from both axes
            lines = []
            labels = []
            
            if 'line1' in locals(): 
                lines.extend(line1)
                labels.append("Train Accuracy")
            if 'line2' in locals(): 
                lines.extend(line2)
                labels.append("Test Accuracy")
            if 'line3' in locals(): 
                lines.extend(line3)
                labels.append("Train Loss")
            if 'line4' in locals(): 
                lines.extend(line4)
                labels.append("Test Loss")
            
            ax1.legend(lines, labels, fontsize=10, loc="center right")
    else:
        ax1.text(0.5, 0.5, "No Training History Available", 
                fontsize=12, ha='center', va='center')
        ax1.axis('off')
    
    # Panel 2: Progressive Dropout Results
    ax2 = fig.add_subplot(gs[0, 1])
    
    if prog_results and "dropout_fractions" in prog_results and "accuracies" in prog_results:
        fractions = prog_results["dropout_fractions"]
        strategies = ["high_rq", "low_rq", "random"]
        colors = {"high_rq": "#1f77b4", "low_rq": "#d62728", "random": "#2ca02c"}
        
        for strategy in strategies:
            if strategy in prog_results["accuracies"]:
                accs = prog_results["accuracies"][strategy]
                ax2.plot(fractions, accs, 
                         marker="o" if strategy == "high_rq" else ("s" if strategy == "low_rq" else "^"),
                         color=colors.get(strategy, "black"),
                         linewidth=2, markersize=6,
                         label=strategy.replace("_", " ").title())
        
        ax2.set_xlabel("Dropout Fraction", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Progressive Dropout Results", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No Progressive Dropout Results", 
                fontsize=12, ha='center', va='center')
        ax2.axis('off')
    
    # Panel 3: Eigenvector Dropout or Low-RQ Results
    ax3 = fig.add_subplot(gs[1, 0])
    
    if eig_results and "dropout_fractions" in eig_results and "accuracies" in eig_results and "eigenvector" in eig_results["accuracies"]:
        fractions = eig_results["dropout_fractions"]
        accs = eig_results["accuracies"]["eigenvector"]
        
        ax3.plot(fractions, accs, 'o-', color="#9467bd", linewidth=2, markersize=6, label="Eigenvector")
        ax3.set_xlabel("Dropout Fraction", fontsize=12)
        ax3.set_ylabel("Accuracy (%)", fontsize=12)
        ax3.set_title("Eigenvector Dropout Results", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])
        ax3.legend(fontsize=10)
    elif prog_results and "dropout_fractions" in prog_results and "accuracies" in prog_results and "low_rq" in prog_results["accuracies"]:
        fractions = prog_results["dropout_fractions"]
        accs = prog_results["accuracies"]["low_rq"]
        
        ax3.plot(fractions, accs, 's-', color="#d62728", linewidth=2, markersize=6, label="Low RQ")
        ax3.set_xlabel("Dropout Fraction", fontsize=12)
        ax3.set_ylabel("Accuracy (%)", fontsize=12)
        ax3.set_title("Low RQ Dropout Results", fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 100])
        ax3.legend(fontsize=10)
    else:
        ax3.text(0.5, 0.5, "No Eigenvector or Low RQ Dropout Results", 
                fontsize=12, ha='center', va='center')
        ax3.axis('off')
    
    # Panel 4: High-RQ vs Random Comparison or Alignment Values
    ax4 = fig.add_subplot(gs[1, 1])
    
    if prog_results and "dropout_fractions" in prog_results and "accuracies" in prog_results and "high_rq" in prog_results["accuracies"] and "random" in prog_results["accuracies"]:
        fractions = prog_results["dropout_fractions"]
        high_rq_accs = prog_results["accuracies"]["high_rq"]
        random_accs = prog_results["accuracies"]["random"]
        
        # Calculate the difference between high RQ and random
        # Use the min length if they differ
        min_len = min(len(high_rq_accs), len(random_accs))
        acc_diff = [h - r for h, r in zip(high_rq_accs[:min_len], random_accs[:min_len])]
        
        # Calculate the percentage improvement
        rel_diff = [(h - r) / r * 100 if r > 0 else 0 for h, r in zip(high_rq_accs[:min_len], random_accs[:min_len])]
        
        # Plot absolute difference
        line1 = ax4.plot(fractions[:min_len], acc_diff, 'o-', color="#ff7f0e", linewidth=2, markersize=6, label="Absolute Difference")
        
        # Add a second y-axis for relative difference (%)
        ax4_right = ax4.twinx()
        line2 = ax4_right.plot(fractions[:min_len], rel_diff, 's--', color="#8c564b", linewidth=1.5, markersize=6, label="Relative Improvement (%)")
        
        # Set labels and title
        ax4.set_xlabel("Dropout Fraction", fontsize=12)
        ax4.set_ylabel("Accuracy Difference (High RQ - Random)", fontsize=12)
        ax4_right.set_ylabel("Relative Improvement (%)", fontsize=12)
        ax4.set_title("High RQ vs Random Comparison", fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add a horizontal line at y=0
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Combine legends from both axes
        lines = []
        labels = []
        
        if 'line1' in locals(): 
            lines.extend(line1)
            labels.append("Absolute Difference")
        if 'line2' in locals(): 
            lines.extend(line2)
            labels.append("Relative Improvement (%)")
        
        ax4.legend(lines, labels, fontsize=10, loc="best")
    else:
        ax4.text(0.5, 0.5, "Insufficient Data for Comparison", 
                fontsize=12, ha='center', va='center')
        ax4.axis('off')
    
    # Adjust layout and save/show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if figure_path:
        # Create directory if it doesn't exist
        os.makedirs(figure_path, exist_ok=True)
        
        filepath = os.path.join(figure_path, "experiment_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return filepath
    else:
        plt.show()
        plt.close()
        return None

def plot_dropout_comparison(
    progressive_results: Dict,
    eigenvector_results: Dict,
    figure_path: Optional[str] = None,
    title: str = "Dropout Comparison"
) -> Optional[str]:
    """
    Generate comparison plots between progressive and eigenvector dropout.
    
    Args:
        progressive_results: Progressive dropout results
        eigenvector_results: Eigenvector dropout results
        figure_path: Path to save the figure
        title: Title for the plot
        
    Returns:
        Path to the saved figure, or None if not saved
    """
    # Check if we have necessary data
    if not (progressive_results and eigenvector_results and 
            "dropout_fractions" in progressive_results and 
            "dropout_fractions" in eigenvector_results and
            "accuracies" in progressive_results and
            "accuracies" in eigenvector_results):
        return None
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Get data for plotting
    prog_fractions = progressive_results["dropout_fractions"]
    eig_fractions = eigenvector_results["dropout_fractions"]
    
    # Get progressive high RQ accuracies
    prog_accs = progressive_results["accuracies"]["high_rq"] if "high_rq" in progressive_results["accuracies"] else None
    
    # Get eigenvector accuracies
    eig_accs = eigenvector_results["accuracies"]["eigenvector"] if "eigenvector" in eigenvector_results["accuracies"] else None
    
    # Panel 1: Accuracy comparison
    if prog_accs is not None and eig_accs is not None:
        ax1.plot(prog_fractions, prog_accs, 'o-', color="#1f77b4", linewidth=2, markersize=8, label="Progressive (High RQ)")
        ax1.plot(eig_fractions, eig_accs, 'D-', color="#9467bd", linewidth=2, markersize=8, label="Eigenvector")
        ax1.set_xlabel("Dropout Fraction", fontsize=14)
        ax1.set_ylabel("Accuracy (%)", fontsize=14)
        ax1.set_title("Accuracy vs. Dropout", fontsize=15)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        ax1.legend(fontsize=12)
        
        # Panel 2: Accuracy difference
        # Use the min length if they differ
        min_len = min(len(prog_fractions), len(eig_fractions))
        accuracy_diff = [e - p for p, e in zip(prog_accs[:min_len], eig_accs[:min_len])]
        
        ax2.plot(prog_fractions[:min_len], accuracy_diff, 'o-', color="#d62728", linewidth=2, markersize=8)
        ax2.set_xlabel("Dropout Fraction", fontsize=14)
        ax2.set_ylabel("Accuracy Difference (Eigenvector - Progressive)", fontsize=14)
        ax2.set_title("Accuracy Difference", fontsize=15)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Add text for max difference
        max_diff_idx = np.argmax(np.abs(accuracy_diff))
        max_diff = accuracy_diff[max_diff_idx]
        max_frac = prog_fractions[max_diff_idx]
        
        ax2.plot(max_frac, max_diff, 'o', color='red', markersize=10)
        ax2.annotate(
            f"Max diff: {max_diff:.2f}%\nat {max_frac:.2f}",
            xy=(max_frac, max_diff),
            xytext=(max_frac + 0.05, max_diff + (5 if max_diff > 0 else -5)),
            fontsize=12,
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8)
        )
    else:
        ax1.text(0.5, 0.5, "Insufficient Data for Comparison", 
                fontsize=12, ha='center', va='center')
        ax1.axis('off')
        ax2.text(0.5, 0.5, "Insufficient Data for Comparison", 
                fontsize=12, ha='center', va='center')
        ax2.axis('off')
    
    # Adjust layout and save/show
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if figure_path:
        os.makedirs(figure_path, exist_ok=True)
        filepath = os.path.join(figure_path, "dropout_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return filepath
    else:
        plt.show()
        return None

def log_plots_to_wandb(plot_files: List[str], tags: Optional[Dict[str, str]] = None):
    """
    Log plot files to wandb if available.
    
    Args:
        plot_files: List of plot file paths
        tags: Optional tags to add to the plots
    """
    try:
        # import wandb # Removed from here
        if not WANDB_AVAILABLE or wandb is None or wandb.run is None: # Check WANDB_AVAILABLE and if wandb.run is active
            return
            
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                plot_name = os.path.basename(plot_file).replace(".png", "")
                
                # Add tags if provided
                wandb_kwargs = {}
                if tags:
                    wandb_kwargs["caption"] = tags.get(plot_name, "")
                
                # Log to wandb
                wandb.log({plot_name: wandb.Image(plot_file, **wandb_kwargs)})
    except Exception as e:
        # Other error
        logger.error(f"Error logging to wandb: {str(e)}")

def plot_mean_rq_of_pruned_nodes(
    experiment_results: Dict[str, Any],
    save_dir: str,
    title_prefix: str = "Mean RQ of Pruned Nodes by Layer", # Updated title
    show_plots: bool = False
) -> Optional[str]: # Returns a single filename for the multi-panel plot
    if "pruning_details" not in experiment_results:
        logger.warning("'pruning_details' not found. Skipping mean RQ plot.")
        return None

    pruning_details = experiment_results["pruning_details"]
    dropout_fractions = experiment_results.get("dropout_fractions", [])
    pruned_fractions = dropout_fractions[1:] if dropout_fractions and len(dropout_fractions) > 1 else dropout_fractions

    if not pruned_fractions:
        logger.warning("No pruned_fractions for plotting mean RQ.")
        return None

    # Include "random" strategy now, and also "cascading" if present
    strategies_to_plot = [s for s in pruning_details.keys() if s in ["high_rq", "low_rq", "random", "cascading"]]
    if not strategies_to_plot:
        logger.warning("No relevant strategies (high_rq, low_rq, random, cascading) found in pruning_details for mean RQ plot.")
        return None 

    num_networks = 0
    num_layers = 0 # This will now represent the count of unique layer names
    first_valid_strategy = strategies_to_plot[0]
    sorted_layer_name_keys = []

    if first_valid_strategy in pruning_details and pruning_details[first_valid_strategy]:
        valid_net_indices = list(pruning_details[first_valid_strategy].keys())
        if valid_net_indices:
            first_net_idx = valid_net_indices[0] # Should be 0 for layer_isolated averaged data
            num_networks = len(pruning_details[first_valid_strategy]) # Should be 1 for layer_isolated
            if first_net_idx in pruning_details[first_valid_strategy] and pruning_details[first_valid_strategy][first_net_idx]:
                valid_frac_indices = list(pruning_details[first_valid_strategy][first_net_idx].keys())
                if valid_frac_indices:
                    first_frac_idx = valid_frac_indices[0]
                    if first_frac_idx in pruning_details[first_valid_strategy][first_net_idx] and pruning_details[first_valid_strategy][first_net_idx][first_frac_idx]:
                        # Get layer names (keys) and sort them for consistent panel ordering
                        sorted_layer_name_keys = sorted(list(pruning_details[first_valid_strategy][first_net_idx][first_frac_idx].keys()))
                        num_layers = len(sorted_layer_name_keys)

    if num_networks == 0 or num_layers == 0:
        logger.warning("Could not determine num_networks/num_layers for mean RQ plot.")
        return None

    # Create N panels (subplots), one for each layer
    # Adjust layout if num_layers is large, e.g., 2 columns
    ncols = 2 if num_layers > 2 else 1
    nrows = (num_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows), sharex=True, squeeze=False)
    # squeeze=False ensures axes is always 2D array
    axes_flat = axes.flatten()

    fig.suptitle(f"{title_prefix}: Mean RQ of Pruned Nodes vs. Dropout Fraction", fontsize=16)

    for panel_idx, layer_name_key in enumerate(sorted_layer_name_keys):
        ax = axes_flat[panel_idx]
        ax.set_title(f"Layer: {layer_name_key}") # Use actual layer name for title
        plotted_something_in_ax = False

        for strategy in strategies_to_plot:
            # mean_rq_per_fraction_this_layer will store avg (over networks) mean RQ of nodes dropped in this layer
            mean_rq_per_fraction_this_layer = []
            
            for frac_idx in range(len(pruned_fractions)):
                mean_rq_for_this_frac_layer_all_nets = []
                for net_idx in range(num_networks):
                    # Fetch layer data using layer_name_key instead of integer layer_idx
                    layer_data_for_net_frac_layer = pruning_details.get(strategy, {}).get(net_idx, {}).get(frac_idx, {}).get(layer_name_key)
                    
                    if layer_data_for_net_frac_layer and not layer_data_for_net_frac_layer.get("skipped", False):
                        num_dropped = layer_data_for_net_frac_layer.get("num_dropped", 0)
                        scores_sum = layer_data_for_net_frac_layer.get("dropped_scores_sum", 0.0)
                        if num_dropped > 0:
                            mean_rq_for_this_frac_layer_all_nets.append(scores_sum / num_dropped)
                        else:
                            # If no nodes dropped in this specific layer for this net/frac/strat, append NaN
                            mean_rq_for_this_frac_layer_all_nets.append(np.nan)
                    else:
                        mean_rq_for_this_frac_layer_all_nets.append(np.nan) # Data missing or skipped
                
                # Average the mean RQs for this layer/frac across all networks
                if mean_rq_for_this_frac_layer_all_nets:
                    # Check if all elements are NaN to avoid RuntimeWarning from np.nanmean
                    if all(np.isnan(x) for x in mean_rq_for_this_frac_layer_all_nets):
                        mean_rq_per_fraction_this_layer.append(np.nan)
                    else:
                        mean_rq_per_fraction_this_layer.append(np.nanmean(mean_rq_for_this_frac_layer_all_nets))
                else:
                    mean_rq_per_fraction_this_layer.append(np.nan)
            
            if any(not np.isnan(val) for val in mean_rq_per_fraction_this_layer):
                 ax.plot(pruned_fractions, mean_rq_per_fraction_this_layer, marker='o', linestyle='-', label=f"Strat: {strategy}")
                 plotted_something_in_ax = True
        
        if plotted_something_in_ax:
            ax.legend()
        ax.grid(True)
        # Use panel_idx for layout checks, assuming sorted_layer_name_keys determines the panels
        if panel_idx // ncols == nrows -1 or nrows == 1 : 
             ax.set_xlabel("Dropout Fraction")
        if panel_idx % ncols == 0: 
             ax.set_ylabel("Mean RQ of Pruned Nodes")

    # Hide any unused subplots if num_layers doesn't fill the grid
    for i in range(num_layers, nrows * ncols):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    
    plot_filename = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_by_layer_mean_rq_pruned.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved mean RQ of pruned nodes by layer plot to {plot_filename}")
    if show_plots: plt.show()
    plt.close()
    return plot_filename

def plot_per_layer_pruning_percentage(
    experiment_results: Dict[str, Any],
    save_dir: str,
    title_prefix: str = "Per-Layer Pruning", # Shorter default title
    show_plots: bool = False
) -> Optional[str]: # Returns a single filename for the combined plot
    if "pruning_details" not in experiment_results:
        logger.warning("'pruning_details' not found. Skipping per-layer pruning plot.")
        return None

    pruning_details = experiment_results["pruning_details"]
    dropout_fractions = experiment_results.get("dropout_fractions", [])
    # Use fractions corresponding to pruning steps (after baseline)
    pruned_fractions = dropout_fractions[1:] if dropout_fractions and len(dropout_fractions) > 1 else dropout_fractions

    if not pruned_fractions:
        logger.warning("No pruned_fractions for plotting per-layer pruning.")
        return None

    strategies_to_plot = [s for s in pruning_details.keys() if s in ["high_rq", "low_rq", "random"]]
    if not strategies_to_plot:
        logger.warning("No relevant strategies found in pruning_details.")
        return None

    # Determine num_networks and num_layers from the data structure
    num_networks = 0
    num_layers = 0
    if not strategies_to_plot:
        logger.warning("No strategies to plot for per-layer pruning percentages.")
        return [] # Already handled, but good to be explicit

    first_valid_strategy = strategies_to_plot[0]
    # Check if the strategy key exists and its value (dict of net_idx) is not empty
    if first_valid_strategy in pruning_details and pruning_details[first_valid_strategy]:
        network_data_for_strategy = pruning_details[first_valid_strategy]
        valid_net_indices = list(network_data_for_strategy.keys())
        if valid_net_indices:
            first_net_idx = valid_net_indices[0]
            num_networks = len(network_data_for_strategy)
            
            # Check if this net_idx has entries for fractions (and the entry is not empty)
            fraction_data_for_net = network_data_for_strategy.get(first_net_idx)
            if fraction_data_for_net: # Check if dict for fractions is not empty
                valid_frac_indices = list(fraction_data_for_net.keys())
                if valid_frac_indices:
                    first_frac_idx = valid_frac_indices[0]
                    layer_info_dict = fraction_data_for_net.get(first_frac_idx)
                    
                    if layer_info_dict and isinstance(layer_info_dict, dict):
                        num_layers = len(layer_info_dict.keys())
                    else:
                        logger.warning(f"Plotting: layer_info_dict for strategy '{first_valid_strategy}', net {first_net_idx}, frac {first_frac_idx} is None or not a dict: {layer_info_dict}. Cannot determine num_layers.")
                else:
                    logger.warning(f"Plotting: No fraction data found for strategy '{first_valid_strategy}', net {first_net_idx}. Cannot determine num_layers.")
            else:
                logger.warning(f"Plotting: No fraction data (or empty dict) found for strategy '{first_valid_strategy}', net {first_net_idx}. Cannot determine num_layers.")
        else:
            logger.warning(f"Plotting: No network data found for strategy '{first_valid_strategy}'. Cannot determine num_layers.")
    else:
        logger.warning(f"Plotting: Strategy '{first_valid_strategy}' not found in pruning_details or has no data. Cannot determine num_layers.")

    if num_networks == 0 or num_layers == 0:
        logger.warning("Could not determine num_networks/num_layers for per-layer pruning plot.")
        return None

    fig, axes = plt.subplots(len(strategies_to_plot), 1, figsize=(10, 6 * len(strategies_to_plot)), sharex=True)
    if len(strategies_to_plot) == 1:
        axes = [axes] # Ensure axes is always a list

    fig.suptitle(f"{title_prefix}: % Nodes Pruned Per Layer vs. Overall Fraction", fontsize=16)

    for i, strategy in enumerate(strategies_to_plot):
        ax = axes[i]
        layer_pruning_percentages_for_strat = [[] for _ in range(num_layers)]

        for frac_idx in range(len(pruned_fractions)):
            # For each layer, this will hold a list of pruning percentages from each network replicate for the current fraction
            pruning_percentages_for_all_replicates_this_frac_layer = [[] for _ in range(num_layers)]

            for net_idx in range(num_networks):
                # Try to get data for current net, strat, frac
                layer_data_for_net_frac = pruning_details.get(strategy, {}).get(net_idx, {}).get(frac_idx, {})

                for layer_idx in range(num_layers):
                    if layer_idx in layer_data_for_net_frac:
                        layer_detail = layer_data_for_net_frac[layer_idx]
                        if not layer_detail.get("skipped", False) and layer_detail.get("total_nodes_in_layer", 0) > 0:
                            perc = (layer_detail["num_dropped"] / layer_detail["total_nodes_in_layer"]) * 100.0
                            pruning_percentages_for_all_replicates_this_frac_layer[layer_idx].append(perc)
                        else:
                            pruning_percentages_for_all_replicates_this_frac_layer[layer_idx].append(0.0) # Skipped or no nodes, 0% pruned
                    else:
                        # If layer_idx is not in layer_data_for_net_frac (e.g., global pruning didn't touch it, or data truly missing)
                        # We should append a NaN to signify missing data for this replicate, so nanmean works correctly.
                        pruning_percentages_for_all_replicates_this_frac_layer[layer_idx].append(np.nan)
            
            # Average over networks for this fraction for each layer
            for layer_idx in range(num_layers):
                # nanmean will correctly average, ignoring NaNs. If all are NaN, result is NaN.
                current_layer_replicate_percentages = pruning_percentages_for_all_replicates_this_frac_layer[layer_idx]
                if not current_layer_replicate_percentages or all(np.isnan(x) for x in current_layer_replicate_percentages):
                    mean_val = np.nan
                else:
                    mean_val = np.nanmean(current_layer_replicate_percentages)
                layer_pruning_percentages_for_strat[layer_idx].append(mean_val)
        
        for layer_idx in range(num_layers):
            # Only plot if there's some non-NaN data for this layer
            if any(not np.isnan(val) for val in layer_pruning_percentages_for_strat[layer_idx]):
                ax.plot(pruned_fractions, layer_pruning_percentages_for_strat[layer_idx], marker='.', linestyle='-', label=f"Layer {layer_idx}")
        
        ax.set_ylabel("% Nodes Pruned in Layer")
        ax.set_title(f"Strategy: {strategy.replace('_', ' ').title()}")
        # Only add legend if there are plotted lines with labels
        handles, labels = ax.get_legend_handles_labels()
        if handles: # or if labels:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        ax.set_ylim(0, 105) # Percentage

    axes[-1].set_xlabel("Overall Dropout Fraction Target")
    plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust for suptitle and legend
    
    plot_filename = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_per_layer_pruning.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved combined per-layer pruning percentage plot to {plot_filename}")
    if show_plots: plt.show()
    plt.close()
    return plot_filename

def plot_per_layer_contribution_to_pruning(
    experiment_results: Dict[str, Any],
    save_dir: str,
    title_prefix: str = "Layer Contribution to Total Pruning",
    show_plots: bool = False
) -> Optional[str]: # Returns a single filename for the combined plot
    if "pruning_details" not in experiment_results:
        logger.warning("'pruning_details' not found. Skipping layer contribution plot.")
        return None

    pruning_details = experiment_results["pruning_details"]
    dropout_fractions = experiment_results.get("dropout_fractions", [])
    pruned_fractions = dropout_fractions[1:] if dropout_fractions and len(dropout_fractions) > 1 else dropout_fractions

    if not pruned_fractions:
        logger.warning("No pruned_fractions for plotting layer contribution.")
        return None

    strategies_to_plot = [s for s in pruning_details.keys() if s in ["high_rq", "low_rq", "random"]]
    if not strategies_to_plot:
        logger.warning("No relevant strategies found in pruning_details for layer contribution plot.")
        return None

    num_networks = 0
    num_layers = 0
    first_valid_strategy = strategies_to_plot[0]
    if first_valid_strategy in pruning_details and pruning_details[first_valid_strategy]:
        first_net_idx = list(pruning_details[first_valid_strategy].keys())[0]
        num_networks = len(pruning_details[first_valid_strategy])
        if pruning_details[first_valid_strategy][first_net_idx]:
            first_frac_idx = list(pruning_details[first_valid_strategy][first_net_idx].keys())[0]
            if pruning_details[first_valid_strategy][first_net_idx][first_frac_idx]:
                num_layers = len(pruning_details[first_valid_strategy][first_net_idx][first_frac_idx].keys())

    if num_networks == 0 or num_layers == 0:
        logger.warning("Could not determine num_networks/num_layers for layer contribution plot.")
        return None

    fig, axes = plt.subplots(len(strategies_to_plot), 1, figsize=(10, 6 * len(strategies_to_plot)), sharex=True)
    if len(strategies_to_plot) == 1: axes = [axes]

    fig.suptitle(f"{title_prefix}: % Contribution of Each Layer to Total Pruned Nodes", fontsize=16)

    for i, strategy in enumerate(strategies_to_plot):
        ax = axes[i]
        # layer_contribution_percentages[layer_idx][frac_idx] = avg_contribution_percentage
        layer_contribution_percentages_for_strat = [[] for _ in range(num_layers)]

        for frac_idx in range(len(pruned_fractions)):
            contribution_this_frac_all_layers_all_nets = [[] for _ in range(num_layers)]

            for net_idx in range(num_networks):
                layer_data_for_net_frac = pruning_details.get(strategy, {}).get(net_idx, {}).get(frac_idx, {})
                
                total_nodes_pruned_this_net_this_frac = 0
                num_dropped_per_layer_this_net_this_frac = [0] * num_layers

                if not layer_data_for_net_frac: # If no data for this net/frac, contributions are NaN for all layers
                    for l_idx_fill in range(num_layers):
                        contribution_this_frac_all_layers_all_nets[l_idx_fill].append(np.nan)
                    continue

                for layer_idx in range(num_layers):
                    if layer_idx in layer_data_for_net_frac:
                        layer_detail = layer_data_for_net_frac[layer_idx]
                        if not layer_detail.get("skipped", False):
                            dropped_in_layer = layer_detail.get("num_dropped", 0)
                            num_dropped_per_layer_this_net_this_frac[layer_idx] = dropped_in_layer
                            total_nodes_pruned_this_net_this_frac += dropped_in_layer
                
                for layer_idx in range(num_layers):
                    if total_nodes_pruned_this_net_this_frac > 0:
                        contrib = (num_dropped_per_layer_this_net_this_frac[layer_idx] / total_nodes_pruned_this_net_this_frac) * 100.0
                        contribution_this_frac_all_layers_all_nets[layer_idx].append(contrib)
                    else: # No nodes pruned in this net for this frac, or data missing
                        contribution_this_frac_all_layers_all_nets[layer_idx].append(0.0 if not layer_data_for_net_frac else np.nan)
            
            for layer_idx in range(num_layers):
                current_layer_contributions = contribution_this_frac_all_layers_all_nets[layer_idx]
                if not current_layer_contributions or all(np.isnan(x) for x in current_layer_contributions):
                    mean_val = np.nan
                else:
                    mean_val = np.nanmean(current_layer_contributions)
                layer_contribution_percentages_for_strat[layer_idx].append(mean_val)
        
        for layer_idx in range(num_layers):
            if any(not np.isnan(val) for val in layer_contribution_percentages_for_strat[layer_idx]):
                ax.plot(pruned_fractions, layer_contribution_percentages_for_strat[layer_idx], marker='.', linestyle='-', label=f"Layer {layer_idx}")
        
        ax.set_ylabel("% Contribution to Total Pruned")
        ax.set_title(f"Strategy: {strategy.replace('_', ' ').title()}")
        # Only add legend if there are plotted lines with labels
        handles, labels = ax.get_legend_handles_labels()
        if handles: # or if labels:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        ax.set_ylim(0, 105)

    axes[-1].set_xlabel("Overall Dropout Fraction Target")
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    plot_filename = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_layer_contribution.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved layer contribution to pruning plot to {plot_filename}")
    if show_plots: plt.show()
    plt.close()
    return plot_filename 

def plot_rq_stats_per_layer(
    experiment_results: Dict[str, Any],
    save_dir: str,
    title_prefix: str = "Pre-Pruning RQ Stats by Layer",
    show_plots: bool = False
) -> Optional[str]:
    """
    Plots the mean and std deviation of RQ scores for all nodes within each layer, 
    averaged over network replicates, before any pruning is applied.
    Assumes results["pre_pruning_layer_stats"] exists with structure:
    {layer_idx: {"avg_mean_rq": val, "avg_std_rq": val}}
    """
    pre_pruning_stats = experiment_results.get("pre_pruning_layer_stats")
    if not pre_pruning_stats or not isinstance(pre_pruning_stats, dict):
        logger.warning("'pre_pruning_layer_stats' not found or invalid in experiment_results. Skipping RQ stats plot.")
        return None

    layer_indices = sorted(pre_pruning_stats.keys())
    if not layer_indices:
        logger.warning("No layer data in pre_pruning_layer_stats. Skipping RQ stats plot.")
        return None

    avg_means = [pre_pruning_stats[l_idx].get("avg_mean_rq", np.nan) for l_idx in layer_indices]
    avg_stds = [pre_pruning_stats[l_idx].get("avg_std_rq", np.nan) for l_idx in layer_indices]

    x = np.arange(len(layer_indices))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(max(10, len(layer_indices) * 1.5), 7))
    rects1 = ax.bar(x - width/2, avg_means, width, label='Avg. Mean RQ', yerr=avg_stds, capsize=5, color='skyblue', ecolor='gray')
    # Optionally plot std as separate bars if preferred over error bars:
    # rects2 = ax.bar(x + width/2, avg_stds, width, label='Avg. Std Dev RQ', color='lightcoral')

    ax.set_ylabel('RQ Score Value')
    ax.set_title(title_prefix)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l_idx}" for l_idx in layer_indices])
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    # if using separate bars for std: autolabel(rects2)

    fig.tight_layout()
    
    plot_filename = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_rq_stats.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved pre-pruning RQ stats plot to {plot_filename}")
    if show_plots: plt.show()
    plt.close()
    return plot_filename 

def plot_layer_isolated_dropout_results(
    experiment_results: Dict[str, Any],
    save_dir: str,
    title_prefix: str = "Layer Isolated Pruning",
    show_plots: bool = False
) -> List[str]: # Returns list of filenames, one per strategy
    if ("accuracies_isolated" not in experiment_results or 
        not isinstance(experiment_results.get("accuracies_isolated"), dict)):
        logger.warning("'accuracies_isolated' not found or is not a dictionary. Skipping layer-isolated plot.")
        return []

    accuracies_isolated = experiment_results["accuracies_isolated"]
    stds_isolated = experiment_results.get("stds_isolated", {})
    dropout_fractions = experiment_results.get("dropout_fractions", [])

    if not dropout_fractions:
        logger.warning("No dropout_fractions for plotting layer-isolated results.")
        return []

    strategies_present = list(accuracies_isolated.keys())
    if not strategies_present:
        logger.warning("No strategies found in accuracies_isolated.")
        return []

    # Determine all unique layer names that were isolated across any strategy
    all_isolated_layer_names = set()
    for strategy in strategies_present:
        if accuracies_isolated[strategy]: # Check if strategy data exists
            all_isolated_layer_names.update(accuracies_isolated[strategy].keys())
    
    sorted_layer_names = sorted(list(all_isolated_layer_names))
    num_unique_layers = len(sorted_layer_names)

    if num_unique_layers == 0:
        logger.warning("Could not determine any layers for layer-isolated plot.")
        return []

    plot_files = []
    
    # Strategy colors and markers (consistent with plot_dropout_results)
    strategy_colors = {"high_rq": "#d62728", "low_rq": "#2ca02c", "random": "#1f77b4", "eigenvector": "#ff7f0e"}
    strategy_markers = {"high_rq": "o", "low_rq": "s", "random": "^", "eigenvector": "D"}
    strategy_labels = {
        "high_rq": "Prune Highest RQ", 
        "low_rq": "Prune Lowest RQ", 
        "random": "Random Pruning",
        "eigenvector": "Eigenvector Pruning"
    }

    # Create ONE figure with N panels (one per isolated layer)
    ncols_fig = 2 if num_unique_layers > 2 else 1
    nrows_fig = (num_unique_layers + ncols_fig - 1) // ncols_fig
    fig, axes = plt.subplots(nrows_fig, ncols_fig, figsize=(8 * ncols_fig, 6 * nrows_fig), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(f"{title_prefix}: Impact of Isolating Layer Pruning\n(Network Accuracy vs. Fraction Pruned from Single Layer)", fontsize=16)

    for panel_idx, layer_name_key in enumerate(sorted_layer_names):
        ax = axes_flat[panel_idx]
        ax.set_title(f"Isolating Layer: {layer_name_key}")
        plotted_something_on_ax = False

        for strategy in strategies_present: # Iterate through available strategies
            if not accuracies_isolated.get(strategy) or not accuracies_isolated[strategy].get(layer_name_key):
                continue # Skip if this strategy or layer_name not present for this strategy

            accs_this_layer_strat = accuracies_isolated[strategy].get(layer_name_key, [])
            stds_this_layer_strat = stds_isolated.get(strategy, {}).get(layer_name_key, [])

            if accs_this_layer_strat:
                valid_len = min(len(dropout_fractions), len(accs_this_layer_strat))
                plot_fractions = dropout_fractions[:valid_len]
                plot_accs = accs_this_layer_strat[:valid_len]
                plot_stds = stds_this_layer_strat[:valid_len] if len(stds_this_layer_strat) >= valid_len else np.zeros(valid_len)
                
                ax.errorbar(plot_fractions, plot_accs, yerr=plot_stds, 
                            marker=strategy_markers.get(strategy, 'x'), 
                            color=strategy_colors.get(strategy, 'k'),
                            label=strategy_labels.get(strategy, strategy),
                            linestyle='-', capsize=3)
                plotted_something_on_ax = True
        
        if plotted_something_on_ax:
            ax.legend(fontsize=9)
            ax.grid(True)
            ax.set_ylim(0, 105) # Assuming accuracy percentage
        else:
            ax.text(0.5, 0.5, "No data for this layer", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        if panel_idx // ncols_fig == nrows_fig - 1 or nrows_fig == 1: # X-label for bottom row panels
            ax.set_xlabel("Fraction Pruned from This Layer")
        if panel_idx % ncols_fig == 0: # Y-label for first column panels
            ax.set_ylabel("Network Accuracy (%)")
    
    # Hide unused subplots
    for i in range(num_unique_layers, nrows_fig * ncols_fig):
        fig.delaxes(axes_flat[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make space for suptitle
    plot_filename = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_isolated_layers_summary.png")
    plt.savefig(plot_filename)
    logger.info(f"Saved consolidated layer-isolated plot to {plot_filename}")
    if show_plots:
        plt.show()
    plt.close(fig)
    plot_files.append(plot_filename) # Now returns a list with a single filename

    return plot_files 

def plot_metric_evolution(
    metric_evolution_data: Dict[str, Dict[str, List[Any]]], 
    metric_name: str, 
    save_dir: str, 
    title_prefix: str = "", 
    show_plots: bool = False
) -> Optional[str]:
    """
    Plots the evolution of a given metric over epochs for different layers.

    Args:
        metric_evolution_data: Data for a single metric. Expected format:
                               {layer_name: {'epochs': [...], 'mean_scores': [...], 'std_scores': [...]}}
        metric_name: The name of the metric being plotted (e.g., "RQ", "MI_G").
        save_dir: Directory to save the plot.
        title_prefix: Prefix for the plot title.
        show_plots: Whether to display the plot.

    Returns:
        Path to the saved plot file, or None.
    """
    if metric_name in ["RQ", "MI_G"]: # Keep the debug log if needed for future
        logger.debug(f"plot_metric_evolution START ({metric_name}): Received metric_evolution_data: {metric_evolution_data}")

    if not metric_evolution_data:
        logger.warning(f"No evolution data provided (metric_evolution_data is empty) for {metric_name}. Skipping plot.")
        return None

    # Check if any layer actually has epoch data.
    has_valid_layer_data = False
    for layer_name, layer_data_dict in metric_evolution_data.items():
        if isinstance(layer_data_dict, dict) and layer_data_dict.get("epochs") and layer_data_dict.get("mean_scores"):
            if len(layer_data_dict["epochs"]) == len(layer_data_dict["mean_scores"]):
                has_valid_layer_data = True
                break 
    
    if not has_valid_layer_data:
        logger.warning(f"No valid layer data with epochs and mean_scores found in metric_evolution_data for {metric_name}. Skipping plot.")
        return None
    
    plt.figure(figsize=(12, 7))
    
    # Determine common epochs across all layers that have data, or handle layers individually
    # For simplicity, we'll plot each layer against its own epochs if they differ.
    # However, it's more common for epochs to be consistent.

    for layer_name, layer_data in metric_evolution_data.items():
        if not isinstance(layer_data, dict):
            logger.warning(f"Data for layer {layer_name} in metric {metric_name} is not a dict. Skipping this layer.")
            continue

        epochs = layer_data.get("epochs")
        mean_scores = layer_data.get("mean_scores")
        std_scores = layer_data.get("std_scores") # Optional

        if not epochs or not mean_scores:
            # This specific layer has no epochs or mean_scores, skip plotting it
            # but don't skip the whole plot if other layers have data.
            logger.debug(f"Layer {layer_name} for metric {metric_name} has no epochs or mean_scores. Skipping this layer line.")
            continue
        
        if len(epochs) != len(mean_scores):
            logger.warning(f"Mismatch in length of epochs and mean_scores for layer {layer_name}, metric {metric_name}. Skipping this layer line.")
            continue

        # Ensure data is sorted by epoch for correct line plotting
        if epochs: # Should be true if we reached here
            sorted_indices = np.argsort(epochs)
            epochs_sorted = np.array(epochs)[sorted_indices]
            mean_scores_sorted = np.array(mean_scores)[sorted_indices]
            
            line, = plt.plot(epochs_sorted, mean_scores_sorted, marker='o', linestyle='-', label=f'Layer {layer_name}')
            
            if std_scores and len(std_scores) == len(epochs):
                std_scores_sorted = np.array(std_scores)[sorted_indices]
                plt.fill_between(epochs_sorted, 
                                 mean_scores_sorted - std_scores_sorted, 
                                 mean_scores_sorted + std_scores_sorted, 
                                 color=line.get_color(), alpha=0.2)
            elif std_scores:
                logger.warning(f"Length of std_scores does not match epochs for layer {layer_name}, metric {metric_name}. Std dev will not be plotted for this layer.")

    plt.xlabel("Epoch")
    plt.ylabel(f"Mean {metric_name} Score")
    plot_title = f"{title_prefix} Evolution of Mean {metric_name} per Layer"
    plt.title(plot_title)
    
    # Only add legend if there are labeled lines
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc='best', fontsize='small')
    else:
        logger.info(f"No data plotted for metric {metric_name}, legend will be empty.")

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = f"{title_prefix.lower().replace(' ', '_').replace('.', '')}_{metric_name.lower().replace('.', '')}_evolution.png"
    plot_filepath = os.path.join(save_dir, plot_filename)
    
    try:
        plt.savefig(plot_filepath)
        logger.info(f"Saved metric evolution plot to {plot_filepath}")
        if show_plots:
            plt.show()
    except Exception as e:
        logger.error(f"Error saving/showing metric evolution plot: {e}")
        plot_filepath = None
    finally:
        plt.close()
        
    return plot_filepath 