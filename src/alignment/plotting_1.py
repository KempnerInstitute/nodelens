# src/alignment/plotting.py

"""
Plotting utilities for alignment experiments.

This module provides functions for visualizing experiment results,
including training curves, dropout experiments (progressive dropout),
and alignment metrics.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional, Union

import matplotlib
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_pruning_experiments(
    model_constructor: callable,
    dataset_config: Any,
    metric,
    device,
    prune_fractions: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    pruning_mode: str = "global",
    exclude_classification_layer: bool = False,
    replicates: int = 3,
    title: str = "Pruning Experiment",
    batch_size: int = 128,
    save_path: Optional[str] = None,
):
    """
    Sweep multiple prune fractions (0 -> 1), prune the network, measure accuracy, 
    and plot the average accuracy with error bars across 'replicates' re-initialized networks.

    This replicates or extends the "preref" logic for generating a single figure
    of accuracy vs. prune fraction for a specified pruning_mode.

    Args:
        model_constructor: A callable (or class) that builds a fresh model each time.
        dataset_config: Configuration object or dict for the dataset. 
                        (Used by alignment.datasets.load_dataset under the hood)
        metric: AlignmentMetric instance (RQ, MI, etc.)
        device: Torch device
        prune_fractions: List of prune fractions to test
        pruning_mode: "global", "per_layer_combined", "per_layer_independent"
        exclude_classification_layer: If True, skip final classification layer pruning
        replicates: Number of replicates for each fraction (for error bars)
        title: Plot title
        batch_size: Batch size for evaluation
        save_path: If given, path (filename) to save the resulting figure
    """
    from alignment.dropout import progressive_dropout

    results = []
    for frac in prune_fractions:
        replicate_accuracies = []
        for r in range(replicates):
            # Build a fresh model (with random init)
            model = model_constructor().to(device)

            # Optionally load pretrained weights or partial training if needed
            # (Not shown here — do if you want each replicate to have the same or random init.)
            # ...

            # Now prune
            final_acc, _ = progressive_dropout(
                model,
                dataset_config=dataset_config,
                dropout_fraction=frac,
                metric=metric,
                batch_size=batch_size,
                device=device,
                dropout_pruning_mode=pruning_mode,
                exclude_classification_layer=exclude_classification_layer
            )
            replicate_accuracies.append(final_acc)

        mean_acc = np.mean(replicate_accuracies)
        std_acc = np.std(replicate_accuracies)
        results.append((frac, mean_acc, std_acc))
        logger.info(
            f"[PlotExp] fraction={frac:.2f} => accuracy mean={mean_acc:.2f}, std={std_acc:.2f}"
        )

    # Convert to arrays for plotting
    frac_arr = np.array([x[0] for x in results])
    mean_arr = np.array([x[1] for x in results])
    std_arr  = np.array([x[2] for x in results])

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(frac_arr, mean_arr, yerr=std_arr, fmt="o-", capsize=4, label="Accuracy")
    ax.set_xlabel("Prune Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title + f" ({pruning_mode})")
    ax.set_ylim([0, 100])
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved pruning experiment figure to {save_path}")
    else:
        plt.show()

    return results


def plot_per_layer_independent(
    model_constructor: callable,
    dataset_config: Any,
    metric,
    device,
    prune_fraction: float = 0.3,
    exclude_classification_layer: bool = False,
    replicates: int = 3,
    title: str = "Per-Layer Independent Pruning",
    batch_size: int = 128,
    save_path: Optional[str] = None,
):
    """
    Demonstration function that prunes each layer independently for a *fixed* prune_fraction
    and plots accuracy vs. layer_index or layer_name.

    This corresponds to the "per_layer_independent" mode from the refactored dropout code.
    We'll measure the accuracy drop for each layer individually.

    For multiple replicates, we'll average results across random inits of the same model.

    Args:
        model_constructor: Callable that returns a fresh model.
        dataset_config: dataset config or object
        metric: alignment metric instance
        device: Torch device
        prune_fraction: fraction to prune in each layer
        exclude_classification_layer: skip final classification layer if True
        replicates: number of random inits
        title: figure title
        batch_size: batch size for test
        save_path: path to save figure (optional)
    """
    from alignment.dropout import progressive_dropout

    # We'll gather a list (layer_idx => list_of_accuracies_across_replicates)
    # Then we can average them for a final plot.

    # First, we need a temporary model to see how many alignment layers exist
    tmp_model = model_constructor().to(device)
    n_layers = len(tmp_model.alignment_layers)
    tmp_model = None

    # We'll accumulate results in shape (n_layers, replicates)
    # for each replicate r, we do "per_layer_independent" => that returns an overall last accuracy,
    # but we want the accuracy for each layer. We'll store them manually.
    # => The "progressive_dropout" logs them, but to store them we might do an approach:
    # However, from the code we wrote, the function returns only the final accuracy 
    # and logs the per-layer. We might parse logs or modify progressive_dropout to return
    # detailed results_by_layer. For now, let's do a simpler approach:
    # We'll call progressive_dropout separately per layer. 
    # But that wouldn't exactly replicate the standard "per_layer_independent" approach 
    # that tries each layer in a single run.

    # For consistency with the refactor, we can do it in a single call and parse the logs.
    # The simpler approach here is to modify progressive_dropout if needed. 
    # We'll do a repeated approach: For each replicate, we run "per_layer_independent"
    # and use logs or an attribute. Or let's assume we do separate calls for each layer:

    # Quick approach: We'll do a single replicate approach for each layer. 
    # Then for replicates we'll do that multiple times. 
    # => This is simpler but less efficient. 
    # If you want it exactly in the single call, you'd need progressive_dropout to return a full list.

    # Implementation: We'll do the single-run approach that "progressive_dropout" does:
    #   returns the final accuracy, logs layer wise in results_by_layer, but we can intercept logs 
    #   or update progressive_dropout code to return that array. 
    # We'll do that. Let's just replicate the "layer wise" approach:

    # For final completeness, let's just do multiple calls, each time specifying the layer in question. 
    # Then store the final accuracy. That approach is shown below:

    # We'll store final data as: layer_accs[layer_idx] -> list of replicate accuracies
    layer_accs = [[] for _ in range(n_layers)]

    # We'll do replicate loops:
    for r in range(replicates):
        # Fresh model
        model = model_constructor().to(device)
        # We'll run the "per_layer_independent" mode once for all layers. 
        # That code runs a loop internally across each layer. 
        # The only catch: the original code returns only the last accuracy, 
        # but it logs each layer's result. 
        # Let's modify the approach:
        # We can monkey-patch the logger or read the info from the function. 
        # We'll do a local hack: We'll copy progressive_dropout code or update it to return 
        # a dictionary. 
        # For demonstration, let's do the naive approach of a single run for each layer.

        # => naive approach: for l_idx in [0..n_layers]: prune that layer alone, measure accuracy
        # Then revert. We'll do that in code right here, so we skip progressive_dropout. 
        # But let's see if we can just do the official approach:

        from alignment.dropout import _evaluate_model_accuracy, _compute_metric_for_all_nodes

        # 1) compute node-level alignment scores
        model.eval()
        node_scores_by_layer = _compute_metric_for_all_nodes(
            model, metric, device, 
            dataset_config.test_loader, # or load it properly
            exclude_classification_layer=exclude_classification_layer
        )

        for l_idx in range(n_layers):
            # skip if classification layer
            if exclude_classification_layer and (l_idx == n_layers - 1):
                # Evaluate no pruning for that layer
                no_prune_acc = _evaluate_model_accuracy(model, dataset_config.test_loader, device)
                layer_accs[l_idx].append(no_prune_acc)
                continue

            # backup layer weights
            layer_mod = model.alignment_layers[l_idx]
            original_w = layer_mod.weight.detach().clone()
            original_b = layer_mod.bias.detach().clone() if layer_mod.bias is not None else None

            # sort ascending
            scores_tsr = node_scores_by_layer[l_idx]
            sorted_idx = torch.argsort(scores_tsr, descending=False)
            layer_node_count = len(scores_tsr)
            num_prune_layer = int(round(layer_node_count * prune_fraction))
            to_prune = sorted_idx[:num_prune_layer]

            with torch.no_grad():
                for node_i in to_prune:
                    layer_mod.weight[node_i].zero_()
                    if layer_mod.bias is not None:
                        layer_mod.bias[node_i].zero_()

            acc_this_layer = _evaluate_model_accuracy(model, dataset_config.test_loader, device)
            layer_accs[l_idx].append(acc_this_layer)

            # revert
            with torch.no_grad():
                layer_mod.weight.copy_(original_w)
                if layer_mod.bias is not None and original_b is not None:
                    layer_mod.bias.copy_(original_b)

    # Now compute stats
    layer_means = []
    layer_stds = []
    for l_idx in range(n_layers):
        arr = np.array(layer_accs[l_idx])
        layer_means.append(arr.mean())
        layer_stds.append(arr.std())

    # Plot bar or line chart with error bars
    fig, ax = plt.subplots(figsize=(8, 4))
    x_idx = np.arange(n_layers)
    ax.errorbar(x_idx, layer_means, yerr=layer_stds, fmt="o-", capsize=4)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} (prune_fraction={prune_fraction:.2f})")
    ax.set_ylim(0, 100)
    ax.grid(True)

    # Label the x-axis with layer names if you want
    # But we only have layer_idx here. If you want names, fetch them from model.alignment_names.
    # We'll do a dummy label approach for demonstration:
    layer_labels = [f"L{i}" for i in range(n_layers)]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(layer_labels)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved per-layer independent figure to {save_path}")
    else:
        plt.show()

    # Return results if wanted
    return {
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "layer_accs": layer_accs,
    }


###############################################################################
# Additional utilities for training curves, etc. (optional)
###############################################################################

def plot_training_results(
    train_loss: List[float],
    train_accuracy: List[float],
    val_loss: Optional[List[float]] = None,
    val_accuracy: Optional[List[float]] = None,
    title: str = "Training Results",
    save_path: Optional[str] = None
):
    """
    Simple function to plot training and validation curves for loss & accuracy.
    This is a generic utility that can be adapted as needed.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = list(range(1, len(train_loss) + 1))

    # Plot Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    if val_loss is not None and len(val_loss) == len(epochs):
        ax1.plot(epochs, val_loss, 'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.grid(True)
    ax1.legend()

    # Plot Accuracy
    ax2.plot(epochs, train_accuracy, 'g-', linewidth=2, label='Train Accuracy')
    if val_accuracy is not None and len(val_accuracy) == len(epochs):
        ax2.plot(epochs, val_accuracy, 'r--', linewidth=2, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved training result figure to {save_path}")
    else:
        plt.show()