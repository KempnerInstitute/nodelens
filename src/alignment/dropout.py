"""
Dropout implementation and analysis utilities (fixed).

We remove any code that used a direct L2 weight norm for pruning. Now, "high_rq"
or "low_rq" rely on hooking-based alignment metrics. "random" ignores the metric
and shuffles. CNN layers are handled by flattening filters. 
"""

import logging
import os
import time
import copy
import random
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import all CNN layer types explicitly
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Linear, Sequential

from alignment.metrics import AlignmentMetric, get_metric, compute_all_node_scores
from alignment.datasets import DataSet
from alignment.utils.evaluation import evaluate_networks_ensemble, _evaluate_model_accuracy
from alignment.utils.model_utils import _normalize_device, _ensure_model_on_device, process_cnn_weights
from alignment.utils.core import _create_mask_from_indices

logger = logging.getLogger(__name__)


@dataclass
class DropoutResults:
    network_accuracies: Dict[int, List[float]]
    network_losses: Dict[int, List[float]]
    dropout_fractions: List[float]
    dropout_indices: Optional[Dict[int, List[int]]] = None
    timing_info: Dict[str, Union[float, List[float]]] = field(default_factory=dict)


def eigenvector_dropout(
    model: nn.Module,
    dataset_config: Any,
    dropout_fraction: float = 0.1,
    metric: Optional[AlignmentMetric] = None,
    batch_size: int = 128,
    num_batches: int = 10,
    device: Optional[torch.device] = None,
    dropout_mode: str = "scaled",
    dropout_pruning_mode: str = "layer_wise",
    debug_mode: bool = False,
) -> Tuple[float, List[float]]:
    """
    Apply eigenvector-based pruning to a model.

    This method prunes nodes based on the top eigenvector of the alignment metric.

    Args:
        model: Neural network model to prune
        dataset_config: Dataset configuration
        dropout_fraction: Fraction of nodes to prune
        metric: Alignment metric to use
        batch_size: Batch size for evaluation
        num_batches: Number of batches to use for alignment computation
        device: Device to run on
        dropout_mode: "scaled" or "unscaled"
        dropout_pruning_mode: "layer_wise" or "global_joint"
        debug_mode: If True, print detailed debug information

    Returns:
        Tuple of (accuracy after pruning, alignment values per layer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize device
    device = _normalize_device(device)

    from alignment.datasets import load_dataset

    dataset = load_dataset(dataset_config, batch_size=batch_size)
    test_loader = dataset.test_loader

    logger.info(f"[Eigenvector Dropout] fraction={dropout_fraction}, mode={dropout_mode}, pruning_mode={dropout_pruning_mode}")

    # Evaluate baseline accuracy
    model.to(device)
    model.eval()
    base_acc = _evaluate_model_accuracy(model, test_loader, device)

    # Get alignment values
    alignment_values = []

    # Initialize metadata dictionary
    layer_metadata_by_idx = {}

    # First, get architecture-specific metadata for all layers
    for layer_idx, layer_mod in enumerate(model.alignment_layers):
        # Process weights with architecture-specific awareness
        layer_weights, layer_metadata = process_cnn_weights(
            model, layer_idx, pruning_strategy="structure-aware"  # Eigenvector dropout uses structure-aware strategy
        )

        # Store metadata for later use
        layer_metadata_by_idx[layer_idx] = layer_metadata

        # Log meaningful information about CNN-specific processing
        if layer_metadata.get("type", "").startswith("conv_"):
            logger.info(f"CNN layer detected: {layer_idx} ({layer_metadata['name']}) - {layer_metadata['type']}")

            if layer_metadata.get("requires_dimensional_matching"):
                logger.info(f"  Residual connection detected: Ensuring dimensional compatibility")

            if layer_metadata.get("all_outputs_required"):
                logger.info(f"  Dense connection detected: Preserving channel connectivity")

    # Compute node scores using the metric
    node_scores_by_layer = compute_all_node_scores(
        model=model, metric_instance=metric, device=device, data_loader=test_loader, num_batches=num_batches, debug_mode=debug_mode
    )

    # Save original weights
    original_w = {}
    original_b = {}
    for l_i, layer_mod in enumerate(model.alignment_layers):
        original_w[l_i] = layer_mod.weight.data.clone()
        if layer_mod.bias is not None:
            original_b[l_i] = layer_mod.bias.data.clone()

    # Apply pruning based on the pruning mode
    if dropout_pruning_mode == "global_joint":
        # Gather all nodes' alignment scores
        all_nodes = []
        for l_i, scores in node_scores_by_layer.items():
            for node_idx, score in enumerate(scores):
                all_nodes.append((l_i, node_idx, score.item()))

        # Sort by score (highest first - we're using eigenvector alignment)
        all_nodes.sort(key=lambda x: x[2], reverse=True)

        # Determine how many nodes to drop
        total_nodes = len(all_nodes)
        nodes_to_drop = int(round(dropout_fraction * total_nodes))

        # Get nodes to drop
        drop_list = all_nodes[:nodes_to_drop]

        # Group by layer
        drop_by_layer = {}
        for layer_idx, node_idx, _ in drop_list:
            if layer_idx not in drop_by_layer:
                drop_by_layer[layer_idx] = []
            drop_by_layer[layer_idx].append(node_idx)

        # Apply dropout to each layer
        for layer_idx, nodes in drop_by_layer.items():
            layer_mod = model.alignment_layers[layer_idx]
            w_data = layer_mod.weight.data
            out_dim = w_data.shape[0]

            # Record alignment values for this layer
            alignment_values.append(float(torch.mean(node_scores_by_layer[layer_idx])))

            if dropout_mode == "scaled":
                # Create mask using vectorized operations
                mask = _create_mask_from_indices(w_data.shape, nodes, device)

                # Scale the remaining weights
                drop_frac = len(nodes) / float(out_dim)
                scale = 1.0 / (1.0 - drop_frac) if drop_frac < 0.9999 else 10.0
                layer_mod.weight.data = w_data * mask * scale

                # Handle bias
                if layer_mod.bias is not None:
                    bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, [i for i in nodes if i < layer_mod.bias.data.shape[0]], device)
                    layer_mod.bias.data = layer_mod.bias.data * bias_mask * scale
                    else:
                # Just zero out the weights without scaling
                for node_idx in nodes:
                    if node_idx < out_dim:
                        w_data[node_idx] = 0.0
                        if layer_mod.bias is not None and node_idx < layer_mod.bias.data.shape[0]:
                            layer_mod.bias.data[node_idx] = 0.0

    elif dropout_pruning_mode == "layer_wise":
        # Apply pruning to each layer independently
        for l_i, layer_mod in enumerate(model.alignment_layers):
            w_data = layer_mod.weight.data
            out_dim = w_data.shape[0]
            nodes_to_drop = int(round(dropout_fraction * out_dim))

            if nodes_to_drop <= 0:
                alignment_values.append(0.0)
            continue

            # Get scores for this layer
            scores = node_scores_by_layer[l_i]

            # Record alignment value
            alignment_values.append(float(torch.mean(scores)))

            # Sort indices by score (highest first)
            sorted_indices = torch.argsort(scores, descending=True)
            to_drop = sorted_indices[:nodes_to_drop]

            if dropout_mode == "scaled":
                # Create mask using vectorized operations
                mask = _create_mask_from_indices(w_data.shape, to_drop, device)

                # Scale the remaining weights
                drop_frac = nodes_to_drop / float(out_dim)
                scale = 1.0 / (1.0 - drop_frac) if drop_frac < 0.9999 else 10.0
                layer_mod.weight.data = w_data * mask * scale

                # Handle bias
                if layer_mod.bias is not None:
                    bmask = _create_mask_from_indices(layer_mod.bias.data.shape, [i for i in to_drop if i < layer_mod.bias.data.shape[0]], device)
                    layer_mod.bias.data = layer_mod.bias.data * bmask * scale
            else:
                # Just zero out the weights without scaling
                for node_idx in to_drop:
                    if node_idx < out_dim:
                        w_data[node_idx] = 0.0
                        if layer_mod.bias is not None and node_idx < layer_mod.bias.data.shape[0]:
                            layer_mod.bias.data[node_idx] = 0.0

    else:
        logger.warning(f"Unsupported pruning mode for eigenvector dropout: {dropout_pruning_mode}")
        return base_acc, []

    # Evaluate accuracy after pruning
    final_acc = _evaluate_model_accuracy(model, test_loader, device)

    # Restore original weights if needed
    # (Uncomment below if you want the model to be restored to original state after evaluation)
    # for l_i, layer_mod in enumerate(model.alignment_layers):
    #     layer_mod.weight.data = original_w[l_i].clone()
    #     if layer_mod.bias is not None and l_i in original_b:
    #         layer_mod.bias.data = original_b[l_i].clone()

    return final_acc, alignment_values


def progressive_dropout_multi_strategy(
    networks: List[nn.Module],
    dataset: DataSet,
    dropout_fractions: List[float],
    metric: AlignmentMetric,
    device="cuda",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "scaled",
    show_progress: bool = False,
    exclude_classification_layer_config: bool = True
) -> Tuple[Dict[str, Dict[int, List[float]]], Dict[str, Dict[int, List[float]]]]:
    strategies = ["high_rq", "low_rq", "random"]
    network_accuracies_by_strategy = {st: {} for st in strategies}
    network_losses_by_strategy = {st: {} for st in strategies}

    if not networks:
        logger.warning("No networks provided.")
        return network_accuracies_by_strategy, network_losses_by_strategy

    if pruning_mode == "cascading_layer":
        logger.info("Cascading layer pruning mode not supported in single pass.")
        for st in strategies:
            st_networks = [copy.deepcopy(net).to(device) for net in networks]
            accs_dict, losses_dict = progressive_dropout(
                st_networks,
                dataset,
                dropout_fractions,
                metric,
                device=device,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                strategy=st,
                show_progress=show_progress,
                use_multi_strategy=False,
                exclude_classification_layer_config=exclude_classification_layer_config
            )
            network_accuracies_by_strategy[st] = accs_dict
            network_losses_by_strategy[st] = losses_dict
        return network_accuracies_by_strategy, network_losses_by_strategy
    else:
        for st in strategies:
            st_networks = [copy.deepcopy(net).to(device) for net in networks]
            accs_dict, losses_dict = progressive_dropout(
                st_networks,
                dataset,
                dropout_fractions,
                metric,
                device=device,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                strategy=st,
                show_progress=show_progress,
                use_multi_strategy=False,
                exclude_classification_layer_config=exclude_classification_layer_config
            )
            network_accuracies_by_strategy[st] = accs_dict
            network_losses_by_strategy[st] = losses_dict

        return network_accuracies_by_strategy, network_losses_by_strategy


def progressive_dropout(
    networks: List[nn.Module],
    all_networks_scores_by_layer: List[Dict[int, torch.Tensor]],
    all_networks_ascend_indices: List[Dict[int, torch.Tensor]],
    all_networks_descend_indices: List[Dict[int, torch.Tensor]],
    all_networks_random_indices: List[Dict[int, torch.Tensor]],
    dataset,
    dropout_fractions: List[float],
    device: Union[str, torch.device] = "cuda",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "scaled",
    strategy: str = "low_rq",
    show_progress: bool = False,
    use_multi_strategy: bool = False,
    debug_mode: bool = False,
    exclude_classification_layer_config: bool = True,
) -> Tuple[
    Union[Dict[int, List[float]], Dict[str, Dict[int, List[float]]]],
    Union[Dict[int, List[float]], Dict[str, Dict[int, List[float]]]],
    Optional[Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]]],
]:
    device = _normalize_device(device)

    if use_multi_strategy:
        logger.info("Running progressive_dropout in multi-strategy mode with batched evaluation of replicates.")

        strategies_to_run = ["high_rq", "low_rq", "random"]
        network_accuracies_all: Dict[str, Dict[int, List[float]]] = {st: {} for st in strategies_to_run}
        network_losses_all: Dict[str, Dict[int, List[float]]] = {st: {} for st in strategies_to_run}
        pruning_details_all: Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]] = {
            st: {n_idx: {} for n_idx in range(len(networks))} for st in strategies_to_run
        }

        all_original_network_metrics_struct = []
        for net_idx, original_net_rep in enumerate(networks):
            _ensure_model_on_device(original_net_rep, device)
            original_net_rep.eval()
            baseline_acc = _evaluate_model_accuracy(original_net_rep, dataset.test_loader, device)
            baseline_loss = 100.0 - baseline_acc
            if debug_mode:
                logger.info(f"Multi-strategy: Baseline Acc for Net {net_idx}: {baseline_acc:.2f}%")
            for st_key in strategies_to_run:
                network_accuracies_all[st_key][net_idx] = [baseline_acc]
                network_losses_all[st_key][net_idx] = [baseline_loss]

            current_net_scores_by_layer = all_networks_scores_by_layer[net_idx]
            current_net_ascend_indices = all_networks_ascend_indices[net_idx]
            current_net_descend_indices = all_networks_descend_indices[net_idx]
            current_net_random_indices = all_networks_random_indices[net_idx]

            original_weights_this_net = {l_idx: lm.weight.data.clone() for l_idx, lm in enumerate(original_net_rep.alignment_layers)}
            original_biases_this_net = {
                l_idx: lm.bias.data.clone() for l_idx, lm in enumerate(original_net_rep.alignment_layers) if lm.bias is not None
            }

            all_original_network_metrics_struct.append(
                {
                    "net_idx": net_idx,
                    "scores_by_layer": current_net_scores_by_layer,
                    "ascend_indices": current_net_ascend_indices,
                    "descend_indices": current_net_descend_indices,
                    "random_indices": current_net_random_indices,
                    "original_weights": original_weights_this_net,
                    "original_biases": original_biases_this_net,
                }
            )

        for frac_idx, frac_val in enumerate(dropout_fractions[1:]):
            for st_key in strategies_to_run:
                pruned_networks_for_batch_eval = []
                current_frac_strat_pruning_details = []

                for net_metrics_info in all_original_network_metrics_struct:
                    original_net_idx = net_metrics_info["net_idx"]
                    net_to_prune = copy.deepcopy(networks[original_net_idx])
                    _ensure_model_on_device(net_to_prune, device)
                    net_to_prune.eval()

                    single_net_pruning_details = _apply_pruning_to_single_net(
                        net_to_prune,
                        frac_val,
                        st_key,
                        pruning_mode,
                        dropout_mode,
                        device,
                        net_metrics_info["scores_by_layer"],
                        net_metrics_info["ascend_indices"],
                        net_metrics_info["descend_indices"],
                        net_metrics_info["random_indices"],
                        debug_mode,
                        exclude_classification_layer=exclude_classification_layer_config,
                    )
                    pruned_networks_for_batch_eval.append(net_to_prune)
                    current_frac_strat_pruning_details.append(single_net_pruning_details["layer_info"])

                if pruned_networks_for_batch_eval:
                    batch_avg_losses, batch_avg_accuracies = evaluate_networks_ensemble(
                        pruned_networks_for_batch_eval, dataset.test_loader, device, nn.CrossEntropyLoss(reduction="sum")
                    )

                    for original_net_idx in range(len(networks)):
                        network_accuracies_all[st_key][original_net_idx].append(batch_avg_accuracies[original_net_idx])
                        network_losses_all[st_key][original_net_idx].append(100.0 - batch_avg_accuracies[original_net_idx])

                for original_net_idx in range(len(networks)):
                    if frac_idx not in pruning_details_all[st_key][original_net_idx]:
                        pruning_details_all[st_key][original_net_idx][frac_idx] = {}
                    pruning_details_all[st_key][original_net_idx][frac_idx] = current_frac_strat_pruning_details[original_net_idx]

        return network_accuracies_all, network_losses_all, pruning_details_all

                    else:
        logger.info(f"Starting progressive_dropout (single-strategy path): strategy={strategy}, " f"mode={pruning_mode}, dropout_mode={dropout_mode}")
    network_accuracies: Dict[int, List[float]] = {}
    network_losses: Dict[int, List[float]] = {}
        pruning_details_single_strat: Dict[int, Dict[int, Dict[int, Dict[str, Any]]]] = {n_idx: {} for n_idx in range(len(networks))}

    if not networks:
            return network_accuracies, network_losses, pruning_details_single_strat

        for net_idx, net in enumerate(networks):
            _ensure_model_on_device(net, device)
            net.eval()
            baseline_acc = _evaluate_model_accuracy(net, dataset.test_loader, device)
            baseline_loss = 100.0 - baseline_acc
            if debug_mode:
                logger.info(f"Single-strategy ({strategy}): Baseline Acc for Net {net_idx}: {baseline_acc:.2f}%")
            network_accuracies[net_idx] = [baseline_acc]
            network_losses[net_idx] = [baseline_loss]

            node_scores_by_layer_single_net = all_networks_scores_by_layer[net_idx]
            ascend_indices_single_net = all_networks_ascend_indices[net_idx]
            descend_indices_single_net = all_networks_descend_indices[net_idx]
            random_indices_single_net = all_networks_random_indices[net_idx]

            original_w_single_net = {l_i: lm.weight.data.clone() for l_i, lm in enumerate(net.alignment_layers)}
            original_b_single_net = {l_i: lm.bias.data.clone() for l_i, lm in enumerate(net.alignment_layers) if lm.bias is not None}

            frac_iter_actual_pruning = dropout_fractions[1:]
            for frac_idx_enum, frac_val_actual in enumerate(frac_iter_actual_pruning):
                for l_i, lm in enumerate(net.alignment_layers):
                    lm.weight.data = original_w_single_net[l_i].clone()
                    if lm.bias is not None and l_i in original_b_single_net:
                        lm.bias.data = original_b_single_net[l_i].clone()

                single_net_pruning_details_here = _apply_pruning_to_single_net(
                    net,
                    frac_val_actual,
                    strategy,
                    pruning_mode,
                    dropout_mode,
                    device,
                    node_scores_by_layer_single_net,
                    ascend_indices_single_net,
                    descend_indices_single_net,
                    random_indices_single_net,
                    debug_mode,
                    exclude_classification_layer=exclude_classification_layer_config,
                )
                acc_val = _evaluate_model_accuracy(net, dataset.test_loader, device)
                network_accuracies[net_idx].append(acc_val)
                network_losses[net_idx].append(100.0 - acc_val)

                pruning_details_single_strat[net_idx][frac_idx_enum] = single_net_pruning_details_here["layer_info"]

        return network_accuracies, network_losses, pruning_details_single_strat


# Helper function to contain the actual pruning logic for a single network
# This consolidates the if/elif pruning_mode blocks from the original progressive_dropout
def _apply_pruning_to_single_net(
    net_to_prune: nn.Module,
    frac_val: float,
    strategy_key: str,
    pruning_mode: str,
    dropout_mode_str: str,
    device: torch.device,
    scores_by_layer: Dict[int, torch.Tensor],
    ascend_indices: Dict[int, torch.Tensor],
    descend_indices: Dict[int, torch.Tensor],
    random_indices: Dict[int, torch.Tensor],
    debug_mode: bool,
    exclude_classification_layer: bool,
) -> Dict[str, Any]:
    # Correct indentation for the function body starts here
    pruning_details_for_this_call = {
        "layer_info": {}
    }
    classification_layer_idx = len(net_to_prune.alignment_layers) - 1

    if pruning_mode == "global_joint":
        all_nodes = []
        for l_i, sc_tensor in scores_by_layer.items():
            if exclude_classification_layer and l_i == classification_layer_idx:
                if debug_mode:
                    logger.info(f"Global Pruning: Skipping layer {l_i} (classification layer) from candidate pool.")
                pruning_details_for_this_call["layer_info"][l_i] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": sc_tensor.shape[0] if sc_tensor is not None else 0,
                    "skipped": True,
                }
                continue
            node_count = sc_tensor.shape[0]
            if strategy_key == "high_rq":
                sorted_ids_this_layer = descend_indices[l_i]
            elif strategy_key == "low_rq":
                sorted_ids_this_layer = ascend_indices[l_i]
            else:
                sorted_ids_this_layer = random_indices[l_i]
            
            for node_idx_in_layer in sorted_ids_this_layer:
                score = sc_tensor[node_idx_in_layer].item() if strategy_key != "random" else 0.0
                all_nodes.append((l_i, node_idx_in_layer.item(), score))
        
        if strategy_key == "random": 
            random.shuffle(all_nodes)
        elif strategy_key == "high_rq": 
            all_nodes.sort(key=lambda x: x[2], reverse=True)
        elif strategy_key == "low_rq":  
            all_nodes.sort(key=lambda x: x[2])

        total_count = len(all_nodes)
        num_drop = int(round(frac_val * total_count))
        nodes_to_drop_info = all_nodes[:num_drop]

        if debug_mode and num_drop > 0:
            logger.info(f"Debug Global Pruning: Strategy '{strategy_key}', Frac {frac_val:.2f}, NumDrop {num_drop}")
            log_limit = min(5, num_drop)
            logger.info(f"  Nodes to drop (first {log_limit}): {[ (n[0], n[1], round(n[2], 4)) for n in nodes_to_drop_info[:log_limit]]}")
            if num_drop > log_limit:
                logger.info(f"  Nodes to drop (last {log_limit}): {[ (n[0], n[1], round(n[2], 4)) for n in nodes_to_drop_info[-log_limit:]]}")

        drop_by_layer_indices = {}
        for l_idx, n_idx, score_val in nodes_to_drop_info:
            drop_by_layer_indices.setdefault(l_idx, []).append(n_idx)
            if l_idx not in pruning_details_for_this_call["layer_info"]:
                pruning_details_for_this_call["layer_info"][l_idx] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": scores_by_layer[l_idx].shape[0],
                }
            pruning_details_for_this_call["layer_info"][l_idx]["num_dropped"] += 1
            if strategy_key != "random":
                pruning_details_for_this_call["layer_info"][l_idx]["dropped_scores_sum"] += score_val

        for l_idx_init in scores_by_layer.keys():
            if exclude_classification_layer and l_idx_init == classification_layer_idx:
                if l_idx_init not in pruning_details_for_this_call["layer_info"]:
                    pruning_details_for_this_call["layer_info"][l_idx_init] = {
                        "num_dropped": 0,
                        "dropped_scores_sum": 0.0,
                        "total_nodes_in_layer": scores_by_layer[l_idx_init].shape[0],
                        "skipped": True,
                    }
                    continue
            if l_idx_init not in pruning_details_for_this_call["layer_info"]:
                pruning_details_for_this_call["layer_info"][l_idx_init] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": scores_by_layer[l_idx_init].shape[0],
                }

        for l_idx_prune, node_indices_to_prune in drop_by_layer_indices.items():
            layer_mod = net_to_prune.alignment_layers[l_idx_prune]
            wdat = layer_mod.weight.data
            out_dim_layer = wdat.shape[0]
            mask = _create_mask_from_indices(wdat.shape, node_indices_to_prune, device)
            if dropout_mode_str == "scaled":
                actual_dropped_count = len(node_indices_to_prune)
                fraction_dropped_layer = actual_dropped_count / float(out_dim_layer) if out_dim_layer > 0 else 0.0
                scale = 1.0 / (1.0 - fraction_dropped_layer) if fraction_dropped_layer < 0.9999 else 10.0
                layer_mod.weight.data *= mask * scale
                if layer_mod.bias is not None:
                    bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, node_indices_to_prune, device)
                    layer_mod.bias.data *= bias_mask * scale
            else:
                layer_mod.weight.data *= mask
                if layer_mod.bias is not None:
                    bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, node_indices_to_prune, device)
                    layer_mod.bias.data *= bias_mask

    elif pruning_mode in ["layer_wise", "layer_isolated"]:
        classification_layer_idx = len(net_to_prune.alignment_layers) - 1
        if debug_mode:
            logger.info(f"Debug Layer-Wise/Isolated Pruning: Strategy '{strategy_key}', Frac {frac_val:.2f}")
        for l_i, layer_mod in enumerate(net_to_prune.alignment_layers):
            if exclude_classification_layer and l_i == classification_layer_idx:
                if debug_mode:
                    logger.info(f"Layer-Wise/Isolated Pruning: Skipping layer {l_i} (classification layer) entirely.")
                pruning_details_for_this_call["layer_info"][l_i] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": layer_mod.weight.data.shape[0],
                    "skipped": True,
                }
                continue
            out_dim = layer_mod.weight.data.shape[0]
            n_drop = int(round(frac_val * out_dim))
            if n_drop <= 0:
                pruning_details_for_this_call["layer_info"][l_i] = {"num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": out_dim}
                continue
            pruned_node_indices_for_layer = []
            if strategy_key == "high_rq":
                pruned_node_indices_for_layer = descend_indices[l_i][:n_drop]
            elif strategy_key == "low_rq":
                pruned_node_indices_for_layer = ascend_indices[l_i][:n_drop]
            else:
                pruned_node_indices_for_layer = random_indices[l_i][:n_drop]

            # Ensure wdat is defined BEFORE the debug block and before its use
            wdat = layer_mod.weight.data  # Define wdat here, unconditionally for this layer

            if debug_mode and len(pruned_node_indices_for_layer) > 0:
                indices_to_log = (
                    pruned_node_indices_for_layer.tolist()
                    if isinstance(pruned_node_indices_for_layer, torch.Tensor)
                    else pruned_node_indices_for_layer
                )
                log_limit_layer = min(5, len(indices_to_log))
                scores_for_log = []
                if l_i in scores_by_layer:
                    layer_scores_tensor = scores_by_layer[l_i]
                    try:
                        scores_for_log = [round(layer_scores_tensor[idx].item(), 4) for idx in indices_to_log[:log_limit_layer]]
                    except IndexError:
                        logger.warning(f"Debug Layer-Wise: Index out of bounds while fetching scores for logging layer {l_i}")
                        scores_for_log = ["N/A"] * log_limit_layer
                else:
                    scores_for_log = ["N/A"] * log_limit_layer
                logger.info(
                    f"  Layer {l_i}: Dropping {len(indices_to_log)} nodes. Indices (first {log_limit_layer}): {indices_to_log[:log_limit_layer]} with scores: {scores_for_log}"
                )
            mask = _create_mask_from_indices(wdat.shape, pruned_node_indices_for_layer, device)
            if dropout_mode_str == "scaled":
                frac_d_layer = n_drop / float(out_dim) if out_dim > 0 else 0.0
                scale = 1.0 / (1.0 - frac_d_layer) if frac_d_layer < 0.9999 else 10.0
                layer_mod.weight.data *= mask * scale
                if layer_mod.bias is not None:
                    bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, pruned_node_indices_for_layer, device)
                    layer_mod.bias.data *= bias_mask * scale
            else:
                for node_idx in pruned_node_indices_for_layer:
                    if node_idx < out_dim:
                        wdat[node_idx] = 0.0  # Corrected direct modification for zero mode
                        if layer_mod.bias is not None and node_idx < layer_mod.bias.data.shape[0]:
                            layer_mod.bias.data[node_idx] = 0.0
            num_actually_dropped = len(pruned_node_indices_for_layer)
            sum_scores_of_dropped = 0.0
            if l_i in scores_by_layer and strategy_key != "random" and num_actually_dropped > 0:
                try:
                    sum_scores_of_dropped = scores_by_layer[l_i][pruned_node_indices_for_layer].sum().item()
                except IndexError:
                    logger.warning(f"Index error summing scores for layer {l_i} in {strategy_key}")
            pruning_details_for_this_call["layer_info"][l_i] = {
                "num_dropped": num_actually_dropped,
                "dropped_scores_sum": sum_scores_of_dropped,
                "total_nodes_in_layer": out_dim,
            }

    elif pruning_mode == "cascading_layer":
        logger.warning("Cascading_layer pruning within batched multi-strategy evaluation is not fully optimized by this helper.")
        for l_i, layer_mod in enumerate(net_to_prune.alignment_layers):
            # Placeholder: Actual cascading logic is complex and not fully implemented here for batched mode
            # For now, it effectively does nothing in this path if _apply_pruning_to_single_net is called for cascading
            # We should ensure pruning_details_for_this_call is populated for all layers even if skipped by cascading here.
            if l_i not in pruning_details_for_this_call["layer_info"]:
                pruning_details_for_this_call["layer_info"][l_i] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": layer_mod.weight.shape[0],
                    "skipped": True,  # Indicating this mode didn't prune it here
                }
            pass

    # Corrected indentation for the final else
    else:
        logger.warning(f"Unrecognized pruning_mode={pruning_mode} in _apply_pruning_to_single_net.")
        # Ensure all layers are in details if mode is unknown, to avoid key errors later
        for l_idx_unknown in range(len(net_to_prune.alignment_layers)):
            if l_idx_unknown not in pruning_details_for_this_call["layer_info"]:
                layer_node_count = net_to_prune.alignment_layers[l_idx_unknown].weight.shape[0]
                pruning_details_for_this_call["layer_info"][l_idx_unknown] = {
                    "num_dropped": 0,
                    "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": layer_node_count,
                }

    return pruning_details_for_this_call
