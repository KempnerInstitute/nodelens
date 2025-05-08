import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any
import logging

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
    
    This enhanced version ensures error bars are shown for all strategies.
    
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
    
    # Set strategy colors and markers
    colors = {"high_rq": "red", "low_rq": "green", "random": "blue"}
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
    
    for strategy in strategies:
        if strategy in results["accuracies"] and len(results["accuracies"][strategy]) > 0:
            # Get accuracy and standard deviation
            accs = results["accuracies"][strategy]
            
            # Ensure we have standard deviations
            if "stds" in results and strategy in results["stds"] and len(results["stds"][strategy]) > 0:
                stds = results["stds"][strategy]
            else:
                stds = np.zeros_like(accs)  # Use zeros if no stds available
                
            # Plot with error bars
            plt.errorbar(
                dropout_fractions[:len(accs)],  # Use only as many fractions as we have accuracies
                accs,
                yerr=stds,  # Include error bars
                label=labels.get(strategy, strategy),
                marker=markers.get(strategy, 'o'),
                color=colors.get(strategy, 'black'),
                capsize=4,  # Add caps to error bars
                markersize=8,
                linewidth=2
            )
    
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
    
    # Loss Plot (if available)
    if "losses" in results and any(len(results["losses"].get(s, [])) > 0 for s in strategies):
        plt.figure(figsize=(10, 6))
        
        for strategy in strategies:
            if strategy in results["losses"] and len(results["losses"][strategy]) > 0:
                losses = results["losses"][strategy]
                
                # Ensure we have standard deviations (for losses we might not have them)
                if "loss_stds" in results and strategy in results["loss_stds"] and len(results["loss_stds"][strategy]) > 0:
                    loss_stds = results["loss_stds"][strategy]
                else:
                    loss_stds = None
                    
                if loss_stds:
                    plt.errorbar(
                        dropout_fractions[:len(losses)],
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
                        dropout_fractions[:len(losses)],
                        losses,
                        label=labels.get(strategy, strategy),
                        marker=markers.get(strategy, 'o'),
                        color=colors.get(strategy, 'black'),
                        markersize=8,
                        linewidth=2
                    )
        
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
    
    return saved_files

# Alias for backward compatibility
custom_plot_dropout = plot_dropout_results

def plot_experiment_summary(
    results: Dict,
    figure_path: Optional[str] = None,
    experiment_name: str = "Experiment Summary"
) -> Optional[str]:
    """
    Generate a comprehensive summary plot of experiment results.
    
    Args:
        results: Dictionary of experiment results
        figure_path: Path to save the figure
        experiment_name: Name of the experiment for the title
        
    Returns:
        Path to the saved figure, or None if not saved
    """
    # Create a subplot grid for the summary
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Extract key information from results
    config = results.get("config", {})
    prog_results = results.get("progressive_dropout", {})
    eig_results = results.get("eigenvector_dropout", {})
    
    # Panel 1: Configuration Summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    # Collect configuration details
    config_text = [
        f"Experiment: {experiment_name}",
        f"Model: {config.get('model', {}).get('model_name', 'Unknown')}",
        f"Dataset: {config.get('dataset', {}).get('dataset_name', 'Unknown')}",
        f"Pruning Mode: {config.get('extra', {}).get('dropout_pruning_mode', 'Unknown')}",
        f"Dropout Mode: {config.get('extra', {}).get('dropout_mode', 'Unknown')}",
        f"Alignment Metric: {config.get('alignment', {}).get('metric', 'Unknown')}"
    ]
    
    ax1.text(0.05, 0.95, "\n".join(config_text), fontsize=12, 
            verticalalignment='top', horizontalalignment='left',
            transform=ax1.transAxes)
    ax1.set_title("Configuration Summary", fontsize=14)
    
    # Panel 2: Progressive Dropout Accuracy
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
    
    if prog_results and "dropout_fractions" in prog_results and "accuracies" in prog_results:
        if "high_rq" in prog_results["accuracies"] and "random" in prog_results["accuracies"]:
            fractions = prog_results["dropout_fractions"]
            high_accs = prog_results["accuracies"]["high_rq"]
            rand_accs = prog_results["accuracies"]["random"]
            
            # Calculate difference
            diff = [h - r for h, r in zip(high_accs, rand_accs)]
            
            ax4.plot(fractions, diff, 'D-', color="purple", linewidth=2, markersize=6)
            ax4.set_xlabel("Dropout Fraction", fontsize=12)
            ax4.set_ylabel("Accuracy Difference (%)", fontsize=12)
            ax4.set_title("High RQ vs Random Difference", fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        elif "alignment_values" in prog_results and "high_rq" in prog_results["alignment_values"]:
            # Plot alignment values by layer
            align_vals = prog_results["alignment_values"]["high_rq"][0]  # First fraction
            
            if isinstance(align_vals, list) and len(align_vals) > 0:
                x = np.arange(len(align_vals))
                ax4.bar(x, align_vals, color="#1f77b4", alpha=0.7)
                ax4.set_xlabel("Layer", fontsize=12)
                ax4.set_ylabel("Alignment Value", fontsize=12)
                ax4.set_title("Alignment by Layer", fontsize=14)
                ax4.set_xticks(x)
                ax4.set_xticklabels([f"Layer {i+1}" for i in range(len(align_vals))])
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, "No Alignment Values Available", 
                        fontsize=12, ha='center', va='center')
                ax4.axis('off')
        else:
            ax4.text(0.5, 0.5, "Insufficient Data for Comparison", 
                    fontsize=12, ha='center', va='center')
            ax4.axis('off')
    else:
        ax4.text(0.5, 0.5, "No Data for Comparison", 
                fontsize=12, ha='center', va='center')
        ax4.axis('off')
    
    # Add overall title
    plt.suptitle(experiment_name, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the figure
    if figure_path:
        os.makedirs(figure_path, exist_ok=True)
        filepath = os.path.join(figure_path, "experiment_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return filepath
    else:
        plt.show()
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
        import wandb
        if wandb.run is None:
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
    except ImportError:
        # wandb not available
        pass
    except Exception as e:
        # Other error
        print(f"Error logging to wandb: {str(e)}") 