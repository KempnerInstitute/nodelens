#!/usr/bin/env python
"""
Quick fix for matplotlib plotting issues in alignment experiments.
"""
import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def process_specific_file(results_file):
    """Process a specific results file."""
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return
    
    logger.info(f"Processing file: {results_file}")
    
    # Load the results
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return
    
    # Get parent directory for figures
    results_dir = os.path.dirname(results_file)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get pruning mode
    pruning_mode = "global_joint"
    if "config" in results and hasattr(results["config"], "extra") and hasattr(results["config"].extra, "dropout_pruning_mode"):
        pruning_mode = results["config"].extra.dropout_pruning_mode
        
    logger.info(f"Using pruning mode: {pruning_mode}")
    
    # Print detailed structure
    if "accuracies" in results:
        for strategy, values in results["accuracies"].items():
            if isinstance(values, dict):
                # This is in multi-strategy format - network_idx -> accuracies list
                logger.info(f"Multi-strategy format detected for {strategy}")
                # Create a flattened view for plotting
                
                # Initialize an empty list for each strategy
                if "flattened_accuracies" not in results:
                    results["flattened_accuracies"] = {}
                results["flattened_accuracies"][strategy] = []
                
                # Calculate mean accuracy for each fraction across all networks
                for net_idx, accs in values.items():
                    logger.info(f"  Network {net_idx}: {len(accs)} values")
                    if len(accs) > 0 and len(results["flattened_accuracies"][strategy]) == 0:
                        # Initialize with the first network's data
                        results["flattened_accuracies"][strategy] = accs.copy()
                    elif len(accs) > 0:
                        # Average with existing data
                        for i, acc in enumerate(accs):
                            if i < len(results["flattened_accuracies"][strategy]):
                                results["flattened_accuracies"][strategy][i] += acc
                
                # Calculate average
                num_networks = len(values)
                if num_networks > 0:
                    results["flattened_accuracies"][strategy] = [x / num_networks for x in results["flattened_accuracies"][strategy]]
                    
                # Overwrite the original format with the flattened version for plotting
                results["accuracies"][strategy] = results["flattened_accuracies"][strategy]
                logger.info(f"Flattened {strategy} data: {results['accuracies'][strategy][:3]}...")
    
    # Create plots
    create_fixed_plots(results, figures_dir, pruning_mode)

def create_fixed_plots(results, save_dir, pruning_mode="global_joint"):
    """Create fixed plots from the results."""
    if not isinstance(results, dict) or "accuracies" not in results or "dropout_fractions" not in results:
        logger.error(f"Invalid results format")
        return False
    
    # Get data
    dropout_fractions = results["dropout_fractions"]
    strategies = list(results["accuracies"].keys())
    
    logger.info(f"Found strategies: {strategies}")
    logger.info(f"Dropout fractions: {dropout_fractions}")
    
    # Set strategy info - use specific colors for each strategy
    colors = {"high_rq": "red", "low_rq": "green", "random": "blue"}
    markers = {"high_rq": "o", "low_rq": "s", "random": "^"}
    labels = {
        "high_rq": "Prune Highest Magnitude", 
        "low_rq": "Prune Lowest Magnitude", 
        "random": "Random Pruning"
    }
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    plt.clf()  # Clear the figure
    
    plot_count = 0  # Track if we've plotted anything
    
    for strategy in strategies:
        if strategy in results["accuracies"] and len(results["accuracies"][strategy]) > 0:
            # Get values
            accs = results["accuracies"][strategy]
            
            # Debug
            logger.info(f"Strategy {strategy} has {len(accs)} values")
            if len(accs) > 0:
                logger.info(f"Sample values: {accs[:3]}")
            
            # Ensure the data is numeric
            try:
                accs = np.array(accs, dtype=np.float32)
            except Exception as e:
                logger.error(f"Error converting accuracy values for {strategy}: {str(e)}")
                continue
                
            # Get stds if available
            if "stds" in results and strategy in results["stds"] and len(results["stds"][strategy]) > 0:
                stds = results["stds"][strategy]
                try:
                    stds = np.array(stds, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error converting std values for {strategy}: {str(e)}")
                    stds = np.zeros_like(accs)
            else:
                stds = np.zeros_like(accs)
            
            # Make sure arrays match
            x_values = np.array(dropout_fractions, dtype=np.float32)
            if len(x_values) > len(accs):
                logger.info(f"Truncating x_values from {len(x_values)} to {len(accs)}")
                x_values = x_values[:len(accs)]
            elif len(x_values) < len(accs):
                logger.info(f"Truncating accs from {len(accs)} to {len(x_values)}")
                accs = accs[:len(x_values)]
                stds = stds[:len(x_values)]
            
            # Plot the data - use specific colors
            color = colors.get(strategy, None)
            logger.info(f"Plotting {strategy} with color {color} and {len(accs)} data points")
            
            try:
                plt.errorbar(
                    x_values,
                    accs,
                    yerr=stds,
                    label=labels.get(strategy, strategy),
                    marker=markers.get(strategy, 'o'),
                    color=color,  # Use the assigned color
                    capsize=4,
                    markersize=8,
                    linewidth=2
                )
                plot_count += 1
            except Exception as e:
                logger.error(f"Error plotting {strategy}: {str(e)}")
    
    # Only add labels if we plotted something
    if plot_count > 0:
        logger.info(f"Successfully plotted {plot_count} strategies")
        plt.title(f"Progressive Dropout: Accuracy vs Dropout Fraction", fontsize=14)
        plt.xlabel("Dropout Fraction", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Save
        file_path = os.path.join(save_dir, f"accuracy_vs_dropout_{pruning_mode}_fixed.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved accuracy plot to {file_path}")
    else:
        logger.error("No data could be plotted for accuracy!")
    
    plt.close()
    
    # Create loss plot if available
    if "losses" in results and any(len(results["losses"].get(s, [])) > 0 for s in strategies):
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        plot_count = 0
        
        for strategy in strategies:
            if strategy in results["losses"] and len(results["losses"][strategy]) > 0:
                losses = results["losses"][strategy]
                
                # Debug
                logger.info(f"Strategy {strategy} has {len(losses)} loss values")
                if len(losses) > 0:
                    logger.info(f"Sample loss values: {losses[:3]}")
                
                # Ensure the data is numeric
                try:
                    losses = np.array(losses, dtype=np.float32)
                except Exception as e:
                    logger.error(f"Error converting loss values for {strategy}: {str(e)}")
                    continue
                
                # Make sure arrays match
                x_values = np.array(dropout_fractions, dtype=np.float32)
                if len(x_values) > len(losses):
                    x_values = x_values[:len(losses)]
                elif len(x_values) < len(losses):
                    losses = losses[:len(x_values)]
                
                # Plot with specific color
                color = colors.get(strategy, None)
                try:
                    plt.plot(
                        x_values,
                        losses,
                        label=labels.get(strategy, strategy),
                        marker=markers.get(strategy, 'o'),
                        color=color,  # Use the assigned color
                        markersize=8,
                        linewidth=2
                    )
                    plot_count += 1
                except Exception as e:
                    logger.error(f"Error plotting loss for {strategy}: {str(e)}")
        
        # Only add labels if we plotted something
        if plot_count > 0:
            logger.info(f"Successfully plotted {plot_count} loss series")
            plt.title(f"Progressive Dropout: Loss vs Dropout Fraction", fontsize=14)
            plt.xlabel("Dropout Fraction", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            # Save
            file_path = os.path.join(save_dir, f"loss_vs_dropout_{pruning_mode}_fixed.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved loss plot to {file_path}")
        else:
            logger.error("No data could be plotted for loss!")
            
        plt.close()
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process the specified file
        results_file = sys.argv[1]
        process_specific_file(results_file)
    else:
        # Find and process the most recent results
        all_dirs = glob.glob('results/alignment_*_*_*')
        if not all_dirs:
            logger.error("No experiment results found")
            sys.exit(1)
        
        # Sort by creation time
        sorted_dirs = sorted(all_dirs, key=os.path.getctime, reverse=True)
        results_dir = sorted_dirs[0]
        logger.info(f"Processing results in: {results_dir}")
        
        # Look for progressive_dropout_results.pkl
        results_file = os.path.join(results_dir, 'progressive_dropout_results.pkl')
        if not os.path.exists(results_file):
            logger.error(f"No results file found at {results_file}")
            sys.exit(1)
        
        process_specific_file(results_file) 