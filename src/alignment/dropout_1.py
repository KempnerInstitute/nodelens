# src/alignment/dropout.py

"""
Dropout (node pruning) implementation and analysis utilities.

Restores three pruning modes:
  1) "global": Prune x% of nodes across all layers by alignment score.
  2) "per_layer_combined": Prune x% from each layer (optionally skip classification).
  3) "per_layer_independent": For each layer, prune x% of that layer alone and measure accuracy.

Uses a general `_compute_metric_for_all_nodes(...)` that calls your selected
metric (RQ, MI, or any other) from alignment.metrics or alignment.alignment_metrics.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alignment.metrics import AlignmentMetric  # or your metric class
# Example: from alignment.metrics import get_metric, compute_per_node_metric

logger = logging.getLogger(__name__)


def _compute_metric_for_all_nodes(
    model: nn.Module,
    metric: AlignmentMetric,
    device: torch.device,
    data_loader: DataLoader,
    exclude_classification_layer: bool = False
) -> Dict[int, torch.Tensor]:
    """
    Compute a user-selected alignment metric (e.g. RQ, MI, etc.) *per node*
    for each alignment layer in `model`.

    We rely on the hooking mechanism (model.hidden[layer_name]) for storing
    layer outputs or inputs. This function:
      1) Runs the data through `model` in eval mode,
      2) Gathers the relevant per-layer (input or output) tensor from model.hidden,
      3) Calls 'metric.compute_node_scores(...)' or similar function to get
         a vector of shape (num_nodes_in_layer,).

    Returns:
        A dict keyed by layer index in `model.alignment_layers`.
        Each value is a 1D float tensor (#nodes_in_that_layer,).
    """
    if not hasattr(model, "alignment_layers") or not hasattr(model, "alignment_names"):
        raise ValueError("Model must define `alignment_layers` and `alignment_names` for alignment-based pruning.")

    if metric is None:
        raise ValueError("A valid AlignmentMetric instance is required for computing per-node alignment scores.")

    model.eval()
    model.to(device)

    # Clear old hooks/hiddens if needed
    # (Depending on your code, you might do model.hidden.clear() at the start)
    for k in model.hidden.keys():
        model.hidden[k] = None

    # We'll force a forward pass on the entire dataset, so hooking can store activations
    with torch.no_grad():
        for inputs, _targets in data_loader:
            inputs = inputs.to(device)
            model(inputs)  # The hooking in model should populate model.hidden[...] now

    # Now compute metric per-layer
    scores_per_layer = {}
    for layer_idx, layer_mod in enumerate(model.alignment_layers):
        layer_name = model.alignment_names[layer_idx]

        # If user says exclude classification layer and this is the last layer:
        if exclude_classification_layer and layer_idx == len(model.alignment_layers) - 1:
            # Return zeros or skip
            node_count = layer_mod.weight.shape[0]
            scores_per_layer[layer_idx] = torch.zeros(node_count, device=device)
            continue

        # Retrieve the activations from hooking
        # Often we want the *input* to this layer to compute RQ(w_i, input).
        # But if your hooking captures the *output* of the same layer, you might adjust accordingly.
        # For example, if we want the input to layer_mod, we might store it under model.hidden["...prev_layer..."].
        # Suppose we have it under model.hidden[layer_name] if we are hooking the layer *input*.
        if layer_name not in model.hidden:
            raise RuntimeError(f"No hooking data for layer '{layer_name}' in model.hidden")

        layer_input = model.hidden[layer_name]  # shape (batch_size, something,...)
        # Flatten if needed. For a linear layer with input dim D, we want shape (N, D).
        if layer_input.dim() > 2:
            # e.g. CNN features => flatten
            layer_input = layer_input.view(layer_input.size(0), -1)

        # Get the layer's weights for all nodes
        w = layer_mod.weight  # shape (#nodes, weight_dim)
        # Possibly flatten as well if layer_mod is conv => you might do w.view(...)
        # But if your metric code handles conv flattening, skip. Adjust as needed.

        # Now call a function from your metrics code that returns a 1D vector, one score per node
        # We'll assume we have something like metric.compute_per_node_scores(layer_input, w)
        node_scores = metric.compute_per_node_scores(layer_input, w, device=device)

        scores_per_layer[layer_idx] = node_scores.detach().to(device)

    return scores_per_layer


def _evaluate_model_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """
    Utility to evaluate model classification accuracy on a given DataLoader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # If model returns (outputs, hidden)
            _, predicted = outputs.max(dim=1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def progressive_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled",
    dropout_pruning_mode: str = "per_layer_combined",
    exclude_classification_layer: bool = False
) -> Tuple[float, List[float]]:
    """
    Main pruning function that prunes 'dropout_fraction' of neurons
    based on the alignment metric's per-node scores.

    Pruning modes:
      - "global": rank all nodes across all layers by their metric scores,
                  prune the bottom X% (or top X%, depends on convention) globally.
      - "per_layer_combined": prune X% from each layer (except if exclude_classification_layer).
      - "per_layer_independent": for each layer, prune X% in that layer alone, measure accuracy,
                                 then revert. Typically returns the last accuracy or logs results.

    Returns:
        (final_accuracy, list_of_alignment_means)
        For "per_layer_independent", 'final_accuracy' is the last measured accuracy
        after the final layer’s test. You can also log separate accuracies per layer.
    """
    # Import dataset loader here to avoid circular imports
    from alignment.datasets import load_dataset
    
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # 1) Compute node-level alignment scores for all layers
    node_scores_by_layer = _compute_metric_for_all_nodes(
        model, metric, device, test_loader,
        exclude_classification_layer=exclude_classification_layer
    )

    if dropout_pruning_mode == "global":
        #
        #   global => gather all node scores, sort them, prune the bottom fraction
        #
        global_list = []
        for l_idx, scores_tsr in node_scores_by_layer.items():
            for node_idx, score in enumerate(scores_tsr):
                global_list.append((score.item(), l_idx, node_idx))

        # Decide if we prune lowest or highest. Suppose we prune the *lowest* alignment.
        # Sort ascending
        global_list.sort(key=lambda x: x[0])
        total_nodes = len(global_list)
        num_prune = int(round(total_nodes * dropout_fraction))
        to_prune = global_list[:num_prune]

        for _, layer_idx, node_idx in to_prune:
            layer_mod = model.alignment_layers[layer_idx]
            with torch.no_grad():
                layer_mod.weight[node_idx].zero_()
                if layer_mod.bias is not None:
                    layer_mod.bias[node_idx].zero_()

        logger.info(f"[Global Pruning] Pruned {num_prune} nodes out of {total_nodes} total (~{100*dropout_fraction:.2f}%).")

        accuracy = _evaluate_model_accuracy(model, test_loader, device)
        alignment_per_layer = [tsr.mean().item() for tsr in node_scores_by_layer.values()]
        return accuracy, alignment_per_layer

    elif dropout_pruning_mode == "per_layer_combined":
        #
        #   per_layer_combined => prune exactly X% from each layer
        #                         (optionally skip the final classification layer)
        #
        for l_idx, scores_tsr in node_scores_by_layer.items():
            layer_mod = model.alignment_layers[l_idx]
            # If classification is excluded, skip last layer
            if exclude_classification_layer and l_idx == len(model.alignment_layers) - 1:
                logger.info(f"[Per-Layer Combined] Skipping classification layer {l_idx}.")
                continue

            # Sort ascending
            sorted_idx = torch.argsort(scores_tsr, descending=False)
            layer_node_count = len(scores_tsr)
            num_prune_layer = int(round(layer_node_count * dropout_fraction))
            to_prune = sorted_idx[:num_prune_layer]

            with torch.no_grad():
                for n_idx in to_prune:
                    layer_mod.weight[n_idx].zero_()
                    if layer_mod.bias is not None:
                        layer_mod.bias[n_idx].zero_()

            logger.info(f"[Per-Layer Combined] Layer {l_idx} pruned {num_prune_layer}/{layer_node_count} nodes.")

        accuracy = _evaluate_model_accuracy(model, test_loader, device)
        alignment_per_layer = [tsr.mean().item() for tsr in node_scores_by_layer.values()]
        return accuracy, alignment_per_layer

    elif dropout_pruning_mode == "per_layer_independent":
        #
        #   per_layer_independent => for each layer L, prune fraction X% of L alone,
        #                            measure, revert, then move on
        #
        # We'll store (layer_idx, pruned_count, accuracy) for each layer
        results_by_layer = []

        for l_idx, scores_tsr in node_scores_by_layer.items():
            layer_mod = model.alignment_layers[l_idx]
            if exclude_classification_layer and l_idx == len(model.alignment_layers) - 1:
                # skip it
                no_prune_acc = _evaluate_model_accuracy(model, test_loader, device)
                results_by_layer.append((l_idx, 0, no_prune_acc))
                logger.info(f"[Per-Layer Independent] Skipping classification layer {l_idx}. Accuracy = {no_prune_acc:.2f}%")
                continue

            # Backup original weights
            original_w = layer_mod.weight.detach().clone()
            original_b = layer_mod.bias.detach().clone() if layer_mod.bias is not None else None

            # Sort ascending
            sorted_idx = torch.argsort(scores_tsr, descending=False)
            layer_node_count = len(scores_tsr)
            num_prune_layer = int(round(layer_node_count * dropout_fraction))
            to_prune = sorted_idx[:num_prune_layer]

            # Prune
            with torch.no_grad():
                for n_idx in to_prune:
                    layer_mod.weight[n_idx].zero_()
                    if layer_mod.bias is not None:
                        layer_mod.bias[n_idx].zero_()

            acc_this_layer = _evaluate_model_accuracy(model, test_loader, device)
            results_by_layer.append((l_idx, num_prune_layer, acc_this_layer))
            logger.info(f"[Per-Layer Ind] L={l_idx}, pruned={num_prune_layer}, accuracy={acc_this_layer:.2f}%")

            # Revert
            with torch.no_grad():
                layer_mod.weight.copy_(original_w)
                if layer_mod.bias is not None and original_b is not None:
                    layer_mod.bias.copy_(original_b)

        # Return last accuracy or store them all externally
        if len(results_by_layer) > 0:
            final_accuracy = results_by_layer[-1][2]
        else:
            final_accuracy = _evaluate_model_accuracy(model, test_loader, device)

        alignment_per_layer = [tsr.mean().item() for tsr in node_scores_by_layer.values()]
        logger.info("[Per-Layer Independent] Results by layer: %s", results_by_layer)
        return final_accuracy, alignment_per_layer

    else:
        raise ValueError(f"Unknown dropout_pruning_mode: {dropout_pruning_mode}")

def eigenvector_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled",
    dropout_pruning_mode: str = "per_layer_combined"
) -> Tuple[float, List[float]]:
    """
    Apply eigenvector-based dropout to a network and measure accuracy and alignment.
    
    This function performs dropout based on the principal eigenvectors of each layer,
    targeting the nodes with the highest activation along the principal component.
    
    Args:
        model: Neural network to analyze
        dataset_config: Configuration for the dataset
        dropout_fraction: Fraction of nodes to drop out
        metric: Alignment metric to use for measuring alignment
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate
        device: Device to run the computation on
        dropout_mode: Mode for dropout application ('scaled' or 'unscaled')
        dropout_pruning_mode: How pruning is distributed:
            - "global": Prune across all neurons in network based on importance
            - "per_layer": Prune independently for each layer
            - "per_layer_combined": Combined approach
        
    Returns:
        Tuple of (accuracy, list of alignment values for each layer)
    """
    if not hasattr(model, "alignment_layers"):
        raise ValueError("Model must have alignment_layers attribute for eigenvector dropout")
    
    if device is None:
        device = next(model.parameters()).device
        
    # Import dataset loader here to avoid circular imports
    from alignment.datasets import load_dataset
    
    # Prepare dataset
    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Set model to evaluation mode
    model.eval()
    
    # Get activations for computing eigenvectors
    activation_stats = _collect_activations(model, test_loader, num_batches, device)
    
    # Calculate eigenvectors for each layer
    n_layers = len(model.alignment_layers)
    
    # For global mode, calculate importance scores across all layers
    if dropout_pruning_mode == "global":
        # First, calculate importance scores for each layer
        all_importance_scores = []
        all_nodes = []
        
        # Calculate importance for each node based on eigenvector components
        for layer_idx in range(n_layers):
            activations = activation_stats[layer_idx]["activations"]
            
            if activations.size(1) == 0:
                logger.debug(f"Layer {layer_idx} has no activations, skipping")
                continue
            
            # Calculate covariance matrix
            activations = activations - activations.mean(0, keepdim=True)
            cov = torch.mm(activations.T, activations) / activations.size(0)
            
            # Get principal eigenvector
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            principal_eigenvector = eigenvectors[:, -1]  # Last column is the principal eigenvector
            
            # Calculate importance scores based on eigenvector components
            importance = torch.abs(principal_eigenvector)
            
            # Store importance scores and node indices
            for i, score in enumerate(importance):
                all_importance_scores.append(score.item())
                all_nodes.append((layer_idx, i))
        
        # Sort all nodes by importance (descending)
        combined = sorted(zip(all_importance_scores, all_nodes), key=lambda x: x[0], reverse=True)
        
        # Calculate total number of nodes to dropout
        total_nodes = len(all_nodes)
        n_dropout = int(total_nodes * dropout_fraction)
        
        # Select top importance nodes to dropout
        nodes_to_dropout = [node for _, node in combined[:n_dropout]]
        
        # Organize dropout indices by layer
        dropout_indices = [[] for _ in range(n_layers)]
        for layer_idx, node_idx in nodes_to_dropout:
            dropout_indices[layer_idx].append(node_idx)
        
        # Convert to tensor
        for i in range(n_layers):
            if dropout_indices[i]:
                dropout_indices[i] = torch.tensor(dropout_indices[i], device=device, dtype=torch.long)
            else:
                dropout_indices[i] = torch.tensor([], device=device, dtype=torch.long)
    else:
        # Original per-layer approach
        dropout_indices = []
        
        for layer_idx in range(n_layers):
            activations = activation_stats[layer_idx]["activations"]
            
            if activations.size(1) == 0:
                logger.debug(f"Layer {layer_idx} has no activations, skipping")
                dropout_indices.append(torch.tensor([], device=device, dtype=torch.long))
                continue
            
            # Calculate covariance matrix
            activations = activations - activations.mean(0, keepdim=True)
            cov = torch.mm(activations.T, activations) / activations.size(0)
            
            # Get principal eigenvector
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            principal_eigenvector = eigenvectors[:, -1]  # Last column is the principal eigenvector
            
            # Calculate importance scores based on eigenvector components
            importance = torch.abs(principal_eigenvector)
            
            # Sort by importance and select top nodes
            output_dim = importance.size(0)
            n_dropout = int(output_dim * dropout_fraction)
            
            # Get indices of highest importance nodes
            _, indices = torch.topk(importance, n_dropout)
            dropout_indices.append(indices)
    
    # Validation loop with eigenvector dropout
    correct = 0
    total = 0
    alignment_values = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply dropout to all layers and get activations
        with torch.no_grad():
            outputs, hidden = model.forward_targeted_dropout(
                inputs, dropout_indices, list(range(n_layers)), dropout_mode=dropout_mode
            )
            
            # Calculate alignment if metric is provided
            if metric is not None:
                batch_alignment = metric.measure(hidden, targets, num_classes)
                if len(alignment_values) == 0:
                    alignment_values = batch_alignment
                else:
                    # Average with previous batches
                    for i, val in enumerate(batch_alignment):
                        alignment_values[i] = (alignment_values[i] * batch_idx + val) / (batch_idx + 1)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Only log at debug level
    logger.debug(f"Eigenvector dropout with fraction {dropout_fraction:.4f}: Accuracy = {accuracy:.4f}")
    
    return accuracy, alignment_values


def _collect_activations(
    model: nn.Module, 
    data_loader: DataLoader, 
    num_batches: int,
    device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    """
    Collect activations from all layers of the model.
    
    Args:
        model: Neural network
        data_loader: DataLoader for input data
        num_batches: Number of batches to process
        device: Device to run computation on
        
    Returns:
        List of dictionaries with activations for each layer
    """
    n_layers = len(model.alignment_layers)
    activation_stats = [{"activations": None} for _ in range(n_layers)]
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
                
            inputs = inputs.to(device)
            
            # Forward pass to get activations
            _, hidden = model(inputs)
            
            # Store activations
            for layer_idx, layer_output in enumerate(hidden):
                # Reshape to 2D: (batch_size, -1)
                flat_output = layer_output.view(layer_output.size(0), -1)
                
                if activation_stats[layer_idx]["activations"] is None:
                    activation_stats[layer_idx]["activations"] = flat_output.cpu()
                else:
                    activation_stats[layer_idx]["activations"] = torch.cat(
                        [activation_stats[layer_idx]["activations"], flat_output.cpu()], dim=0
                    )
    
    # Move activations back to device for computation
    for layer_idx in range(n_layers):
        if activation_stats[layer_idx]["activations"] is not None:
            activation_stats[layer_idx]["activations"] = activation_stats[layer_idx]["activations"].to(device)
    
    return activation_stats 