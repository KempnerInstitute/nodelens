import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any

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
    figure_path=None, 
    pruning_mode="global_joint", 
    dropout_mode="global", 
    title_prefix="Progressive Dropout"
):
    """Plot results from progressive dropout.
    
    Args:
        results (dict): Results dictionary from progressive_dropout.
        figure_path (str, optional): Path to save figures. If None, figures are not saved.
        pruning_mode (str): Mode of pruning used ("global_joint", "layer_wise", "layer_isolated", "cascading_layer").
        dropout_mode (str): Mode of dropout used ("global", "rescaled", "layerwise").
        title_prefix (str): Prefix for plot titles.
        
    Returns:
        list: List of saved figure filenames.
    """
    saved_figures = []
    
    # Map old pruning modes to new ones for backward compatibility
    if pruning_mode == "global":
        pruning_mode = "global_joint"
    elif pruning_mode == "per_layer_combined":
        pruning_mode = "layer_wise" 
    elif pruning_mode == "per_layer_independent":
        pruning_mode = "layer_isolated"
    
    # Get human-readable pruning mode for plot titles
    pruning_mode_display = {
        "global_joint": "Global Joint Pruning",
        "layer_wise": "Layer-wise Pruning",
        "layer_isolated": "Layer Isolation Pruning",
        "cascading_layer": "Cascading Layer Pruning"
    }.get(pruning_mode, pruning_mode)
    
    dropout_mode_display = {
        "global": "Global",
        "rescaled": "Rescaled",
        "layerwise": "Layer-wise",
        "scaled": "Scaled"
    }.get(dropout_mode, dropout_mode)

    # Create figure directory if it doesn't exist
    if figure_path is not None:
        os.makedirs(figure_path, exist_ok=True)
    
    # Extract data
    dropout_fractions = results["dropout_fractions"]
    
    # Define preref style colors and markers
    strategies = ["high_rq", "low_rq", "random", "eigenvector"]
    colors = {
        "high_rq": "#1f77b4",  # blue
        "low_rq": "#d62728",   # red
        "random": "#2ca02c",   # green
        "eigenvector": "#9467bd"  # purple
    }
    markers = {
        "high_rq": "o",
        "low_rq": "s",
        "random": "^",
        "eigenvector": "d"
    }
    labels = {
        "high_rq": "High RQ",
        "low_rq": "Low RQ", 
        "random": "Random",
        "eigenvector": "Eigenvector"
    }
    
    # Plot accuracy vs. dropout fraction
    plt.figure(figsize=(12, 8))
    
    # Plot each strategy
    has_data = False
    for strategy in strategies:
        if strategy in results["accuracies"]:
            has_data = True
            accs = results["accuracies"][strategy]
            mean_accs = np.mean(accs, axis=0) if isinstance(accs, np.ndarray) and accs.ndim > 1 else accs
            std_accs = np.std(accs, axis=0) if isinstance(accs, np.ndarray) and accs.ndim > 1 else np.zeros_like(mean_accs)
            
            plt.plot(
                dropout_fractions, 
                mean_accs, 
                marker=markers.get(strategy, "o"),
                linestyle="-",
                linewidth=2.5,
                markersize=8,
                color=colors.get(strategy, "black"),
                label=labels.get(strategy, strategy.replace("_", " ").title())
            )
            
            # Add error bands
            plt.fill_between(
                dropout_fractions,
                mean_accs - std_accs,
                mean_accs + std_accs,
                alpha=0.2,
                color=colors.get(strategy, "black")
            )
    
    if not has_data:
        if "accuracies" in results and isinstance(results["accuracies"], np.ndarray):
            # Handle direct accuracy array
            mean_accs = np.mean(results["accuracies"], axis=0)
            std_accs = np.std(results["accuracies"], axis=0)
            
            plt.plot(
                dropout_fractions, 
                mean_accs, 
                marker="o",
                linestyle="-",
                linewidth=2.5,
                markersize=8,
                color="#1f77b4",
                label="Mean Accuracy"
            )
            
            # Add error bands
            plt.fill_between(
                dropout_fractions,
                mean_accs - std_accs,
                mean_accs + std_accs,
                alpha=0.2,
                color="#1f77b4"
            )
            
    # Style the plot (preref style)
    plt.xlabel("Dropout Fraction", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title(f"{title_prefix}: {pruning_mode_display} ({dropout_mode_display})", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Set consistent y-axis range
    plt.legend(loc="best", fontsize=12, framealpha=0.7)
    
    # Save the accuracy figure
    if figure_path is not None:
        filename = os.path.join(
            figure_path, 
            f"dropout_{pruning_mode}_{dropout_mode}_accuracy.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_figures.append(filename)
        plt.close()
    else:
        plt.show()
    
    # Plot loss vs. dropout (preref style)
    plt.figure(figsize=(12, 8))
    
    # Plot each strategy
    has_data = False
    for strategy in strategies:
        if strategy in results.get("losses", {}):
            has_data = True
            losses = results["losses"][strategy]
            mean_losses = np.mean(losses, axis=0) if isinstance(losses, np.ndarray) and losses.ndim > 1 else losses
            std_losses = np.std(losses, axis=0) if isinstance(losses, np.ndarray) and losses.ndim > 1 else np.zeros_like(mean_losses)
            
            plt.plot(
                dropout_fractions, 
                mean_losses, 
                marker=markers.get(strategy, "o"),
                linestyle="-",
                linewidth=2.5,
                markersize=8,
                color=colors.get(strategy, "black"),
                label=labels.get(strategy, strategy.replace("_", " ").title())
            )
            
            # Add error bands
            plt.fill_between(
                dropout_fractions,
                mean_losses - std_losses,
                mean_losses + std_losses,
                alpha=0.2,
                color=colors.get(strategy, "black")
            )
    
    if has_data:
        # Style the plot (preref style)
        plt.xlabel("Dropout Fraction", fontsize=14)
        plt.ylabel("Loss (%)", fontsize=14)
        plt.title(f"{title_prefix} Loss: {pruning_mode_display} ({dropout_mode_display})", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=12, framealpha=0.7)
        
        # Save the loss figure
        if figure_path is not None:
            filename = os.path.join(
                figure_path, 
                f"dropout_{pruning_mode}_{dropout_mode}_loss.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            saved_figures.append(filename)
            plt.close()
        else:
            plt.show()
    
    # For layer_isolated mode, plot accuracy for each layer separately
    if pruning_mode == "layer_isolated":
        n_layers = results["accuracies"]["high_rq"].shape[1] - 1 if "high_rq" in results["accuracies"] else 0
        if n_layers > 0:
            for layer_idx in range(n_layers):
                plt.figure(figsize=(10, 6))
                
                for strategy in ["high_rq", "low_rq", "random"]:
                    if strategy in results["accuracies"]:
                        accs = results["accuracies"][strategy]
                        mean_accs = np.mean(accs[:, layer_idx, :], axis=0)
                        std_accs = np.std(accs[:, layer_idx, :], axis=0)
                        
                        plt.plot(
                            dropout_fractions, 
                            mean_accs, 
                            marker=markers.get(strategy, "o"),
                            linestyle="-",
                            linewidth=2.5,
                            markersize=8,
                            color=colors.get(strategy, "black"),
                            label=labels.get(strategy, strategy.capitalize())
                        )
                        
                        # Add error bands
                        plt.fill_between(
                            dropout_fractions,
                            mean_accs - std_accs,
                            mean_accs + std_accs,
                            alpha=0.2,
                            color=colors.get(strategy, "black")
                        )
                
                plt.xlabel("Dropout Fraction", fontsize=14)
                plt.ylabel("Accuracy (%)", fontsize=14)
                plt.title(f"{title_prefix}: {pruning_mode_display} - Layer {layer_idx + 1} ({dropout_mode_display})", fontsize=16)
                plt.grid(True, alpha=0.3)
                plt.ylim(0, 100)
                plt.legend(loc="best", fontsize=12, framealpha=0.7)
                
                if figure_path is not None:
                    filename = os.path.join(
                        figure_path,
                        f"dropout_{pruning_mode}_layer{layer_idx+1}_{dropout_mode}.png"
                    )
                    plt.savefig(filename, dpi=300, bbox_inches="tight")
                    saved_figures.append(filename)
                    plt.close()
                else:
                    plt.show()
            
            # Plot combined case
            plt.figure(figsize=(10, 6))
            
            for strategy in ["high_rq", "low_rq", "random"]:
                if strategy in results["accuracies"]:
                    accs = results["accuracies"][strategy]
                    mean_accs = np.mean(accs[:, -1, :], axis=0)
                    std_accs = np.std(accs[:, -1, :], axis=0)
                    
                    plt.plot(
                        dropout_fractions, 
                        mean_accs, 
                        marker=markers.get(strategy, "o"),
                        linestyle="-",
                        linewidth=2.5,
                        markersize=8,
                        color=colors.get(strategy, "black"),
                        label=labels.get(strategy, strategy.capitalize())
                    )
                    
                    # Add error bands
                    plt.fill_between(
                        dropout_fractions,
                        mean_accs - std_accs,
                        mean_accs + std_accs,
                        alpha=0.2,
                        color=colors.get(strategy, "black")
                    )
            
            plt.xlabel("Dropout Fraction", fontsize=14)
            plt.ylabel("Accuracy (%)", fontsize=14)
            plt.title(f"{title_prefix}: {pruning_mode_display} - All Layers ({dropout_mode_display})", fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.legend(loc="best", fontsize=12, framealpha=0.7)
            
            if figure_path is not None:
                filename = os.path.join(
                    figure_path,
                    f"dropout_{pruning_mode}_all_layers_{dropout_mode}.png"
                )
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                saved_figures.append(filename)
                plt.close()
            else:
                plt.show()
    
    # Save results as JSON for future analysis
    if figure_path is not None:
        try:
            # Helper function to safely convert types for JSON serialization
            def safe_convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'tolist'):  # Check for torch tensors
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: safe_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [safe_convert(item) for item in obj]
                else:
                    return obj
                    
            # Convert data
            json_results = safe_convert(results)
            
            # Save as JSON
            json_path = os.path.join(figure_path, f"dropout_{pruning_mode}_{dropout_mode}_results.json")
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON results: {str(e)}")
    
    return saved_figures

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