#!/usr/bin/env python
"""
Run alignment experiment with cascading layer pruning and enhanced visualization
"""

import os
import sys
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "src"))

# Import alignment modules
from alignment.config import ExperimentConfig
from alignment.experiments.alignment_experiments import AlignmentExperiment, set_logging_level
from alignment.utils.plotting import plot_dropout_results

# Set style to match the alignment_preref version
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

# Ensure wandb is available
try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. Install with 'pip install wandb' for full functionality.")
    wandb = None

def enhanced_plotting(results, figure_path, pruning_mode, dropout_mode, experiment_name):
    """Enhanced plotting function with consistent colors and styles"""
    
    # Define consistent colors and styles to match alignment_preref
    saved_figures = []
    
    # Extract data
    dropout_fractions = results["dropout_fractions"]
    accuracies = np.array(results["accuracies"])
    
    # Plot mean accuracy vs. dropout fraction with enhanced styling
    plt.figure(figsize=(12, 8))
    
    # Use a richer color palette and add markers for better visibility
    plt.plot(dropout_fractions, np.mean(accuracies, axis=0), 'o-', 
             color='#1f77b4', linewidth=2.5, markersize=8, 
             label="Mean Accuracy")
    
    # Add error bands with semi-transparency
    plt.fill_between(
        dropout_fractions,
        np.mean(accuracies, axis=0) - np.std(accuracies, axis=0),
        np.mean(accuracies, axis=0) + np.std(accuracies, axis=0),
        alpha=0.25, color='#1f77b4'
    )
    
    # Enhance plot formatting
    plt.xlabel("Dropout Fraction", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title(f"{experiment_name}: {pruning_mode} Pruning", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Set consistent y-axis range
    plt.xlim(0, max(dropout_fractions) * 1.05)  # Add a small margin
    plt.legend(loc="upper right", fontsize=12)
    
    # Add more information to the plot
    plt.text(0.02, 0.02, f"Dropout Mode: {dropout_mode}", transform=plt.gca().transAxes, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the enhanced figure
    if figure_path is not None:
        os.makedirs(figure_path, exist_ok=True)
        filename = os.path.join(
            figure_path, 
            f"enhanced_{pruning_mode}_{dropout_mode}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_figures.append(filename)
        
        # Log to wandb if available
        if wandb is not None and wandb.run is not None:
            wandb.log({"dropout_plot": wandb.Image(filename)})
            
        plt.close()
    else:
        plt.show()
    
    # Also generate the standard plots for comparison
    standard_plots = plot_dropout_results(
        results, 
        figure_path, 
        pruning_mode=pruning_mode, 
        dropout_mode=dropout_mode, 
        title_prefix=experiment_name
    )
    
    saved_figures.extend(standard_plots)
    return saved_figures

def run_experiment(config_path):
    """Run the experiment with enhanced visualization"""
    print(f"Starting cascading layer pruning experiment at {time.ctime()}")
    start_time = time.time()
    
    # Set up logging
    set_logging_level(logging.INFO)
    
    # Load configuration
    config = ExperimentConfig.load(config_path)
    
    # Ensure we're using progressive dropout with cascading layer
    config.experiment_type = "progressive_dropout"
    
    # Make sure the extra attribute exists
    if not hasattr(config, 'extra'):
        config.extra = type('ExtraConfig', (), {})()
    
    # Set the pruning mode to cascading layer
    config.extra.dropout_pruning_mode = "cascading_layer"
    
    # Initialize wandb if available
    if wandb is not None and getattr(config.checkpointing, "use_wandb", False):
        wandb.init(
            project=getattr(config, "wandb_project", "neural_alignment"),
            entity=getattr(config, "wandb_entity", None),
            name=getattr(config, "experiment_name", "cascading_layer_test"),
            config=config.to_dict()
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
    
    # Create and run experiment
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    # Enhance the default plots
    if "progressive_dropout" in results:
        dropout_results = results["progressive_dropout"]
        pruning_mode = config.extra.dropout_pruning_mode
        dropout_mode = getattr(config.extra, "dropout_mode", "scaled")
        
        # Create enhanced plots
        saved_figures = enhanced_plotting(
            dropout_results, 
            experiment.figure_path,
            pruning_mode, 
            dropout_mode,
            getattr(config, "experiment_name", "Progressive Dropout")
        )
        
        print(f"Generated {len(saved_figures)} plot files")
    
    end_time = time.time()
    print(f"Experiment finished at {time.ctime()}")
    print(f"Total duration: {end_time - start_time:.2f} seconds")
    
    # Finish wandb run
    if wandb is not None and wandb.run is not None:
        wandb.finish()
    
    return results, networks

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/config_alignment_experiment.yaml"
    
    results, _ = run_experiment(config_path) 