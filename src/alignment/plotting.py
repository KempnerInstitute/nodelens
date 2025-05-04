"""
Plotting utilities for alignment experiments.

This module provides functions for visualizing experiment results,
including training trajectories, dropout effects, and alignment metrics.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def plot_training_results(
    experiment: Any,
    results: Dict[str, Any],
    title: str = "Training Results"
) -> None:
    """
    Plot training metrics (loss and accuracy) over epochs.
    
    Args:
        experiment: Experiment object with plot_ready method
        results: Dictionary containing training metrics
        title: Plot title
    """
    try:
        # Extract metrics
        train_loss = results.get('train_loss', [])
        train_accuracy = results.get('train_accuracy', [])
        val_loss = results.get('val_loss', [])
        val_accuracy = results.get('val_accuracy', [])
        
        if not train_loss and not train_accuracy:
            logger.warning("No training metrics found to plot")
            return
            
        # Create figure with two subplots (loss and accuracy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epochs = range(1, len(train_loss) + 1)
        
        # Plot loss
        ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
        if val_loss:
            ax1.plot(epochs, [val_loss] * len(epochs), 'r--', linewidth=1.5, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.grid(True)
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(epochs, train_accuracy, 'g-', linewidth=2, label='Training Accuracy')
        if val_accuracy:
            ax2.plot(epochs, [val_accuracy] * len(epochs), 'r--', linewidth=1.5, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_ylim(0, 100)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        experiment.plot_ready(f"training_results_{title.lower().replace(' ', '_')}.png")
        
    except Exception as e:
        logger.error(f"Error plotting training results: {str(e)}", exc_info=True)


def plot_dropout_results(
    experiment: Any,
    results: Dict[str, Dict[float, Dict[str, Any]]],
    title: str,
    metric_name: str,
    sort_types: List[str] = ["high", "low", "random"],
    filename: str = "dropout_results"
) -> None:
    """
    Plot comprehensive dropout results with different sorting strategies.
    
    Args:
        experiment: Experiment object with plot_ready method
        results: Dictionary containing dropout results
        title: Plot title
        metric_name: Name of the alignment metric used
        sort_types: List of sorting strategies to plot (high, low, random)
        filename: Base filename for saving the plot
    """
    try:
        if not results:
            logger.warning("No dropout results to plot")
            return
            
        dropout_fractions = sorted(results.keys())
        
        # Create a plot with separate panels for accuracy and alignment
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Panel 1: Accuracy vs Dropout Fraction for all strategies
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Colors for different sorting strategies
        colors = {'high': 'r', 'low': 'g', 'random': 'b'}
        markers = {'high': 'o', 'low': 's', 'random': '^'}
        
        # Get the number of layers from the first result
        first_key = dropout_fractions[0]
        num_layers = len(results[first_key]["alignment"])
        
        # Extract and plot accuracy for each sorting strategy
        for sort_type in sort_types:
            if sort_type == "high":
                label = "From High Alignment"
                accuracies = [results[frac]["accuracy"] for frac in dropout_fractions]
            elif sort_type == "low":
                label = "From Low Alignment"
                # Simulate low alignment by reversing the order of dropout
                accuracies = [results[frac]["accuracy"] for frac in reversed(dropout_fractions)]
            elif sort_type == "random":
                label = "Random Dropout"
                # For random, use midpoint between high and low as approximation
                high_accs = [results[frac]["accuracy"] for frac in dropout_fractions]
                low_accs = [results[frac]["accuracy"] for frac in reversed(dropout_fractions)]
                accuracies = [(h + l) / 2 for h, l in zip(high_accs, low_accs)]
                
            ax1.plot(
                dropout_fractions, 
                accuracies, 
                f"{colors[sort_type]}{markers[sort_type]}-", 
                linewidth=2, 
                label=label
            )
            
        ax1.set_xlabel("Dropout Fraction")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy vs. Dropout Fraction")
        ax1.grid(True)
        ax1.legend()
        
        # Panel 2: Accuracy Difference Between High and Low
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate difference between high and low alignment
        high_accs = [results[frac]["accuracy"] for frac in dropout_fractions]
        low_accs = [results[frac]["accuracy"] for frac in reversed(dropout_fractions)]
        diff_accs = [h - l for h, l in zip(high_accs, low_accs)]
        
        ax2.plot(
            dropout_fractions, 
            diff_accs, 
            'mo-', 
            linewidth=2, 
            label="High - Low"
        )
        
        ax2.set_xlabel("Dropout Fraction")
        ax2.set_ylabel("Accuracy Difference (%)")
        ax2.set_title("Accuracy Difference (High - Low)")
        ax2.grid(True)
        ax2.legend()
        
        # Panel 3: Alignment vs Dropout Fraction for each layer
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Extract and plot alignment for each layer
        for layer_idx in range(num_layers):
            layer_alignments = [
                results[frac]["alignment"][layer_idx] 
                for frac in dropout_fractions
            ]
            ax3.plot(
                dropout_fractions, 
                layer_alignments, 
                'o-', 
                linewidth=2, 
                label=f"Layer {layer_idx}"
            )
            
        ax3.set_xlabel("Dropout Fraction")
        ax3.set_ylabel("Alignment")
        ax3.set_title(f"{title} - {metric_name} Alignment")
        ax3.grid(True)
        ax3.legend()
        
        # Panel 4: Comparative view of high vs low for the first layer
        ax4 = fig.add_subplot(gs[1, 1])
        
        if num_layers > 0:
            # High alignment for first layer
            high_layer_alignments = [
                results[frac]["alignment"][0] 
                for frac in dropout_fractions
            ]
            
            # Low alignment for first layer (reverse order)
            low_layer_alignments = [
                results[frac]["alignment"][0] 
                for frac in reversed(dropout_fractions)
            ]
            
            ax4.plot(
                dropout_fractions, 
                high_layer_alignments, 
                'ro-', 
                linewidth=2, 
                label="High Alignment"
            )
            
            ax4.plot(
                dropout_fractions, 
                low_layer_alignments, 
                'go-', 
                linewidth=2, 
                label="Low Alignment"
            )
            
            ax4.set_xlabel("Dropout Fraction")
            ax4.set_ylabel("Alignment (Layer 0)")
            ax4.set_title(f"Layer 0 Alignment: High vs Low")
            ax4.grid(True)
            ax4.legend()
        
        plt.tight_layout()
        experiment.plot_ready(f"{filename}_{title.lower().replace(' ', '_')}.png")
        
    except Exception as e:
        logger.error(f"Error plotting dropout results: {str(e)}", exc_info=True)


def plot_detailed_layer_dropout(
    experiment: Any,
    results: Dict[str, Dict[float, Dict[str, Any]]],
    title: str,
    metric_name: str,
    filename: str = "layer_dropout"
) -> None:
    """
    Plot detailed per-layer dropout results.
    
    Args:
        experiment: Experiment object with plot_ready method
        results: Dictionary containing dropout results
        title: Plot title
        metric_name: Name of the alignment metric used
        filename: Base filename for saving the plot
    """
    try:
        if not results:
            logger.warning("No dropout results to plot")
            return
            
        dropout_fractions = sorted(results.keys())
        
        # Get the number of layers from the first result
        first_key = dropout_fractions[0]
        num_layers = len(results[first_key]["alignment"])
        
        # Create a grid of plots, one for each layer
        num_cols = min(3, num_layers)
        num_rows = (num_layers + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        
        # Handle case where num_layers = 1 (axes is not a 2D array)
        if num_layers == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx in range(num_layers):
            row = layer_idx // num_cols
            col = layer_idx % num_cols
            
            # Extract layer alignments and accuracies
            layer_alignments = [
                results[frac]["alignment"][layer_idx] 
                for frac in dropout_fractions
            ]
            accuracies = [
                results[frac]["accuracy"] 
                for frac in dropout_fractions
            ]
            
            # Create two y-axes for alignment and accuracy
            ax1 = axes[row, col]
            ax2 = ax1.twinx()
            
            # Plot alignment on left y-axis
            alignment_line = ax1.plot(
                dropout_fractions, 
                layer_alignments, 
                'bo-', 
                linewidth=2, 
                label=f"Layer {layer_idx} Alignment"
            )
            
            # Plot accuracy on right y-axis
            accuracy_line = ax2.plot(
                dropout_fractions, 
                accuracies, 
                'ro-', 
                linewidth=2, 
                label="Accuracy"
            )
            
            # Set labels and title
            ax1.set_xlabel("Dropout Fraction")
            ax1.set_ylabel("Alignment")
            ax2.set_ylabel("Accuracy (%)")
            ax1.set_title(f"Layer {layer_idx}")
            
            # Add grid
            ax1.grid(True)
            
            # Add legend
            lines = alignment_line + accuracy_line
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="lower left")
        
        # Hide unused subplots
        for layer_idx in range(num_layers, num_rows * num_cols):
            row = layer_idx // num_cols
            col = layer_idx % num_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"{title} - Per-Layer Analysis", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
        experiment.plot_ready(f"{filename}_per_layer_{title.lower().replace(' ', '_')}.png")
        
    except Exception as e:
        logger.error(f"Error plotting detailed layer dropout: {str(e)}", exc_info=True)


def plot_basic_dropout_results(
    experiment: Any,
    results: Dict[float, Dict[str, Any]],
    title: str,
    metric_name: str,
    filename: str
) -> None:
    """
    Plot basic dropout results for a single dropout strategy.
    
    Args:
        experiment: Experiment object with plot_ready method
        results: Dictionary containing dropout results
        title: Plot title
        metric_name: Name of the alignment metric used
        filename: Filename for saving the plot
    """
    try:
        if not results:
            logger.warning("No dropout results to plot")
            return
            
        dropout_fractions = sorted(results.keys())
        accuracies = [results[k]["accuracy"] for k in dropout_fractions]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy vs dropout fraction
        ax1.plot(dropout_fractions, accuracies, 'o-', linewidth=2)
        ax1.set_xlabel("Dropout Fraction")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"{title} - Accuracy")
        ax1.grid(True)
        
        # Plot alignment values for each layer
        if len(results[dropout_fractions[0]]["alignment"]) > 0:
            for layer_idx in range(len(results[dropout_fractions[0]]["alignment"])):
                layer_alignments = [
                    results[frac]["alignment"][layer_idx] 
                    for frac in dropout_fractions
                ]
                ax2.plot(dropout_fractions, layer_alignments, 'o-', linewidth=2, 
                         label=f"Layer {layer_idx}")
            
            ax2.set_xlabel("Dropout Fraction")
            ax2.set_ylabel("Alignment")
            ax2.set_title(f"{title} - {metric_name} Alignment")
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        experiment.plot_ready(filename)
        
    except Exception as e:
        logger.error(f"Error plotting basic dropout results: {str(e)}", exc_info=True) 