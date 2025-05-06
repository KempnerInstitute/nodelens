import os
import matplotlib.pyplot as plt
import numpy as np

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
        "layerwise": "Layer-wise"
    }.get(dropout_mode, dropout_mode)

    # Create figure directory if it doesn't exist
    if figure_path is not None:
        os.makedirs(figure_path, exist_ok=True)
    
    # Extract data
    dropout_fractions = results["dropout_fractions"]
    accuracies = results["accuracies"]
    
    # Plot mean accuracy vs. dropout fraction
    if pruning_mode in ["global_joint", "layer_wise", "cascading_layer"]:
        plt.figure(figsize=(10, 6))
        plt.plot(dropout_fractions, np.mean(accuracies, axis=0), 'o-', label="Mean Accuracy")
        plt.fill_between(
            dropout_fractions,
            np.mean(accuracies, axis=0) - np.std(accuracies, axis=0),
            np.mean(accuracies, axis=0) + np.std(accuracies, axis=0),
            alpha=0.2
        )
        plt.xlabel("Dropout Fraction")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix}: {pruning_mode_display} ({dropout_mode_display})")
        plt.grid(True, alpha=0.3)
        
        if figure_path is not None:
            filename = os.path.join(
                figure_path, 
                f"dropout_{pruning_mode}_{dropout_mode}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            saved_figures.append(filename)
            plt.close()
        else:
            plt.show()
    
    # For layer_isolated mode, plot accuracy for each layer separately
    if pruning_mode == "layer_isolated":
        n_layers = accuracies.shape[1] - 1  # Last dimension is the combined case
        for layer_idx in range(n_layers):
            plt.figure(figsize=(10, 6))
            plt.plot(dropout_fractions, np.mean(accuracies[:, layer_idx, :], axis=0), 'o-')
            plt.fill_between(
                dropout_fractions,
                np.mean(accuracies[:, layer_idx, :], axis=0) - np.std(accuracies[:, layer_idx, :], axis=0),
                np.mean(accuracies[:, layer_idx, :], axis=0) + np.std(accuracies[:, layer_idx, :], axis=0),
                alpha=0.2
            )
            plt.xlabel("Dropout Fraction")
            plt.ylabel("Accuracy")
            plt.title(f"{title_prefix}: {pruning_mode_display} - Layer {layer_idx + 1} ({dropout_mode_display})")
            plt.grid(True, alpha=0.3)
            
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
        plt.plot(dropout_fractions, np.mean(accuracies[:, -1, :], axis=0), 'o-')
        plt.fill_between(
            dropout_fractions,
            np.mean(accuracies[:, -1, :], axis=0) - np.std(accuracies[:, -1, :], axis=0),
            np.mean(accuracies[:, -1, :], axis=0) + np.std(accuracies[:, -1, :], axis=0),
            alpha=0.2
        )
        plt.xlabel("Dropout Fraction")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix}: {pruning_mode_display} - All Layers ({dropout_mode_display})")
        plt.grid(True, alpha=0.3)
        
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
    
    return saved_figures 