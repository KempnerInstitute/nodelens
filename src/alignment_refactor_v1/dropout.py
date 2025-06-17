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
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import all CNN layer types explicitly
from torch.nn import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Linear, Sequential

from alignment_refac1.metrics import AlignmentMetric, get_metric, compute_all_node_scores
from alignment_refac1.datasets import DataSet
from alignment_refac1.utils.evaluation import evaluate_networks_ensemble, _evaluate_model_accuracy
from alignment_refac1.utils.model_utils import _normalize_device, _ensure_model_on_device, process_cnn_weights
from alignment_refac1.utils.core import _create_mask_from_indices

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

    from alignment_refac1.datasets import load_dataset

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
    all_networks_scores_by_layer: List[Dict[str, torch.Tensor]],
    all_networks_ascend_indices: List[Dict[str, torch.Tensor]],
    all_networks_descend_indices: List[Dict[str, torch.Tensor]],
    all_networks_random_indices: List[Dict[str, torch.Tensor]],
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
    max_workers: Optional[int] = None,
) -> Tuple[
    Union[Dict[int, List[float]], Dict[str, Dict[int, List[float]]]],
    Union[Dict[int, List[float]], Dict[str, Dict[int, List[float]]]],
    Optional[Dict[str, Dict[int, Dict[int, Dict[int, Dict[str, Any]]]]]],
]:
    device = _normalize_device(device)

    if use_multi_strategy:
        logger.info("Running progressive_dropout in multi-strategy mode with batched evaluation of replicates.")
        logger.info(f"Parallel execution will use up to {max_workers or os.cpu_count()} workers.")

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
            
            for st_key_init in strategies_to_run:
                network_accuracies_all[st_key_init][net_idx] = [baseline_acc]
                network_losses_all[st_key_init][net_idx] = [baseline_loss]

            current_net_scores_by_layer = all_networks_scores_by_layer[net_idx]
            current_net_ascend_indices = all_networks_ascend_indices[net_idx]
            current_net_descend_indices = all_networks_descend_indices[net_idx]
            current_net_random_indices = all_networks_random_indices[net_idx]

            all_original_network_metrics_struct.append(
                {
                    "net_idx": net_idx,
                    "scores_by_layer": current_net_scores_by_layer,
                    "ascend_indices": current_net_ascend_indices,
                    "descend_indices": current_net_descend_indices,
                    "random_indices": current_net_random_indices,
                }
            )
        
        pruning_tasks = []
        for frac_idx_enum, frac_val_actual in enumerate(dropout_fractions[1:]):
            for st_key in strategies_to_run:
                for net_idx, net_metrics_info in enumerate(all_original_network_metrics_struct):
                    pruning_tasks.append({
                        "original_net_idx": net_idx,
                        "frac_val": frac_val_actual,
                        "frac_idx_enum": frac_idx_enum,
                        "strategy": st_key,
                        "net_metrics": net_metrics_info,
                        "original_network_ref": networks[net_idx],
                        "pruning_mode": pruning_mode,
                        "dropout_mode_str": dropout_mode,
                        "device": device,
                        "debug_mode": debug_mode,
                        "exclude_classification_layer": exclude_classification_layer_config,
                    })

        if pruning_tasks:
            logger.info(f"Created {len(pruning_tasks)} pruning tasks for parallel execution.")
        else:
            logger.info("No pruning tasks created (dropout_fractions[1:] was empty).")

        def _execute_single_pruning_task(task_args):
            original_net_idx = task_args["original_net_idx"]
            frac_val = task_args["frac_val"]
            frac_idx_enum = task_args["frac_idx_enum"]
            strategy = task_args["strategy"]
            net_metrics = task_args["net_metrics"]
            original_network_ref = task_args["original_network_ref"]

            pm = task_args["pruning_mode"]
            dm_str = task_args["dropout_mode_str"]
            dev = task_args["device"]
            dbg_md = task_args["debug_mode"]
            excl_cls = task_args["exclude_classification_layer"]

            if dbg_md:
                logger.debug(f"Starting task: Net {original_net_idx}, Frac {frac_val:.2f} (idx {frac_idx_enum}), Strat {strategy}")
            
            net_to_prune = copy.deepcopy(original_network_ref)
            _ensure_model_on_device(net_to_prune, dev)
            net_to_prune.eval()

            single_net_pruning_details = _apply_pruning_to_single_net(
                net_to_prune,
                frac_val,
                strategy,
                pm,
                dm_str,
                dev,
                net_metrics["scores_by_layer"],
                net_metrics["ascend_indices"],
                net_metrics["descend_indices"],
                net_metrics["random_indices"],
                dbg_md,
                exclude_classification_layer=excl_cls,
            )
            if dbg_md:
                logger.debug(f"Finished task: Net {original_net_idx}, Frac {frac_val:.2f}, Strat {strategy}")
            return (original_net_idx, frac_idx_enum, strategy, net_to_prune, single_net_pruning_details["layer_info"])

        grouped_pruned_nets_for_eval: Dict[Tuple[int, str], List[Optional[nn.Module]]] = {}
        
        num_workers = max_workers if max_workers is not None else os.cpu_count()
        
        for frac_idx_enum, _ in enumerate(dropout_fractions[1:]):
            for st_key in strategies_to_run:
                grouped_pruned_nets_for_eval[(frac_idx_enum, st_key)] = [None] * len(networks)

        if pruning_tasks:
            logger.info(f"Starting ThreadPoolExecutor with {num_workers} workers to process pruning tasks.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures_map = {executor.submit(_execute_single_pruning_task, task): task for task in pruning_tasks}
                
                iterable_futures = concurrent.futures.as_completed(futures_map)
                if show_progress:
                    iterable_futures = tqdm(iterable_futures, total=len(pruning_tasks), desc="Pruning Tasks")

                processed_tasks_count = 0
                for future in iterable_futures:
                    task_info_for_log = futures_map[future] # Get original task args for logging
                    try:
                        net_idx_res, frac_idx_enum_res, strategy_res, pruned_net_res, details_res = future.result()
                        if debug_mode:
                            logger.debug(f"Successfully completed task: Net {net_idx_res}, Frac Idx {frac_idx_enum_res}, Strat {strategy_res}")
                        
                        eval_key = (frac_idx_enum_res, strategy_res)
                        if eval_key in grouped_pruned_nets_for_eval:
                             grouped_pruned_nets_for_eval[eval_key][net_idx_res] = pruned_net_res
                        else:
                             logger.warning(f"Evaluation key {eval_key} not found during results collection. This is unexpected.")

                        if frac_idx_enum_res not in pruning_details_all[strategy_res][net_idx_res]:
                             pruning_details_all[strategy_res][net_idx_res][frac_idx_enum_res] = {}
                        pruning_details_all[strategy_res][net_idx_res][frac_idx_enum_res] = details_res
                    
                    except Exception as exc:
                        task_info = futures_map[future]
                        logger.error(f"Task original_net_idx={task_info['original_net_idx']}, frac={task_info['frac_val']:.2f}, strat={task_info['strategy']} generated an exception: {exc}")
                    finally:
                        processed_tasks_count += 1
                
                logger.info(f"All {processed_tasks_count}/{len(pruning_tasks)} submitted tasks processed by ThreadPoolExecutor.")
        
        # --- Batched Evaluation (after all pruning tasks are done) ---
        logger.info("Starting batched evaluation of pruned networks.")

        # Create a list of (frac_idx_enum, st_key) for the new progress bar
        evaluation_combos = []
        # frac_idx_enum will correspond to the index in the results list *after* the baseline.
        for frac_idx_enum_val, _ in enumerate(dropout_fractions[1:]):
            for st_key_val in strategies_to_run:
                evaluation_combos.append((frac_idx_enum_val, st_key_val))
        
        # New progress bar for the sequential evaluation combinations
        # This progress bar will only be active if show_progress is True.
        eval_combo_iterator = evaluation_combos
        if show_progress:
            eval_combo_iterator = tqdm(evaluation_combos, desc="Evaluating Dropped Ensembles", leave=False)

        for frac_idx_enum, st_key in eval_combo_iterator:
            eval_key_current = (frac_idx_enum, st_key)
            pruned_networks_for_batch_eval = grouped_pruned_nets_for_eval.get(eval_key_current)

            if pruned_networks_for_batch_eval and not all(net is None for net in pruned_networks_for_batch_eval) :
                valid_nets_for_eval = [n for n in pruned_networks_for_batch_eval if n is not None]
                if len(valid_nets_for_eval) < len(pruned_networks_for_batch_eval) and debug_mode:
                    logger.warning(f"Strategy {st_key}, Frac Idx {frac_idx_enum}: Found {len(pruned_networks_for_batch_eval) - len(valid_nets_for_eval)} None networks (due to errors). Evaluating only valid ones.")

                if valid_nets_for_eval:
                    # Call evaluate_networks_ensemble directly, using the original dataset.test_loader
                    # and disable its internal progress bar
                    batch_avg_losses, batch_avg_accuracies = evaluate_networks_ensemble(
                        valid_nets_for_eval, 
                        dataset.test_loader, 
                        device, 
                        nn.CrossEntropyLoss(reduction="sum"),
                        show_batch_progress=False # Disable internal bar
                    )
                    
                    current_net_idx_in_valid_batch = 0
                    for original_net_idx_eval_loop in range(len(networks)):
                        if pruned_networks_for_batch_eval[original_net_idx_eval_loop] is not None: 
                            network_accuracies_all[st_key][original_net_idx_eval_loop].append(batch_avg_accuracies[current_net_idx_in_valid_batch])
                            network_losses_all[st_key][original_net_idx_eval_loop].append(batch_avg_losses[current_net_idx_in_valid_batch]) # Assuming batch_avg_losses is also per-network
                            current_net_idx_in_valid_batch +=1
                        else:
                            network_accuracies_all[st_key][original_net_idx_eval_loop].append(float('nan'))
                            network_losses_all[st_key][original_net_idx_eval_loop].append(float('nan'))
                else: 
                    for original_net_idx_fill_nan in range(len(networks)):
                        network_accuracies_all[st_key][original_net_idx_fill_nan].append(float('nan'))
                        network_losses_all[st_key][original_net_idx_fill_nan].append(float('nan'))
            else: 
                if debug_mode:
                    logger.info(f"No networks to evaluate for strategy {st_key}, fraction index {frac_idx_enum}.")
                for net_idx_fill in range(len(networks)):
                    network_accuracies_all[st_key][net_idx_fill].append(float('nan'))
                    network_losses_all[st_key][net_idx_fill].append(float('nan'))

        # Map strategy keys to generic score-based keys for the output
        final_network_accuracies_all = {}
        final_network_losses_all = {}
        final_pruning_details_all = {}

        key_map = {
            "high_rq": "high_score",
            "low_rq": "low_score",
            "random": "random"
        }

        for old_key, generic_key in key_map.items():
            if old_key in network_accuracies_all:
                final_network_accuracies_all[generic_key] = network_accuracies_all[old_key]
            if old_key in network_losses_all:
                final_network_losses_all[generic_key] = network_losses_all[old_key]
            if old_key in pruning_details_all: # Ensure pruning_details_all also exists and is structured by these keys
                final_pruning_details_all[generic_key] = pruning_details_all[old_key]
            # If a generic key was expected but not found via mapping (e.g. if strategies_to_run was different)
            # and it's not 'random', it might indicate an issue or need to copy if already generic.
            # However, given strategies_to_run, this should cover it.

        # If any strategy was directly named generically and not in key_map, ensure it's carried over.
        # This is unlikely given the current hardcoded strategies_to_run = ["high_rq", "low_rq", "random"]
        for st_key in network_accuracies_all: # Check original dict
            if st_key not in key_map and st_key not in final_network_accuracies_all:
                final_network_accuracies_all[st_key] = network_accuracies_all[st_key]
        for st_key in network_losses_all:
            if st_key not in key_map and st_key not in final_network_losses_all:
                final_network_losses_all[st_key] = network_losses_all[st_key]
        for st_key in pruning_details_all:
             if st_key not in key_map and st_key not in final_pruning_details_all:
                final_pruning_details_all[st_key] = pruning_details_all[st_key]

        return final_network_accuracies_all, final_network_losses_all, final_pruning_details_all

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


def _apply_pruning_to_single_net(
    net_to_prune: nn.Module,
    frac_val: float,
    strategy_key: str,
    pruning_mode: str,
    dropout_mode_str: str,
    device: torch.device,
    scores_by_layer: Dict[str, torch.Tensor],
    ascend_indices: Dict[str, torch.Tensor],
    descend_indices: Dict[str, torch.Tensor],
    random_indices: Dict[str, torch.Tensor],
    debug_mode: bool,
    exclude_classification_layer: bool,
) -> Dict[str, Any]:
    pruning_details_for_this_call = {"layer_info": {}}
    classification_layer_name = net_to_prune.alignment_names[-1] if net_to_prune.alignment_names else None

    if pruning_mode == "global_joint":
        all_nodes = []
        for layer_name_str, sc_tensor in scores_by_layer.items():
            if exclude_classification_layer and layer_name_str == classification_layer_name:
                if debug_mode:
                    logger.info(f"Global Pruning: Skipping layer {layer_name_str} (classification layer) from candidate pool.")
                pruning_details_for_this_call["layer_info"][layer_name_str] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": sc_tensor.shape[0] if sc_tensor is not None else 0,
                    "skipped": True,
                }
                continue
            
            if sc_tensor is None or sc_tensor.numel() == 0:
                logger.warning(f"Global Pruning: Skipping layer '{layer_name_str}' due to missing or empty scores tensor.")
                pruning_details_for_this_call["layer_info"][layer_name_str] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0,
                    "total_nodes_in_layer": 0, "skipped_due_to_missing_scores": True,
                }
                continue

            node_count = sc_tensor.shape[0]
            current_layer_indices_sorted: Optional[torch.Tensor] = None
            if strategy_key == "high_rq": current_layer_indices_sorted = descend_indices.get(layer_name_str)
            elif strategy_key == "low_rq": current_layer_indices_sorted = ascend_indices.get(layer_name_str)
            else: current_layer_indices_sorted = random_indices.get(layer_name_str)

            if current_layer_indices_sorted is None:
                logger.warning(f"Global Pruning: Sorted indices not found for layer '{layer_name_str}' and strategy '{strategy_key}'. Skipping layer.")
                continue

            for node_idx_in_layer_tensor in current_layer_indices_sorted:
                node_idx_in_layer = node_idx_in_layer_tensor.item()
                score = sc_tensor[node_idx_in_layer].item() if strategy_key != "random" else 0.0
                all_nodes.append((layer_name_str, node_idx_in_layer, score))
        
        if strategy_key == "random": random.shuffle(all_nodes)
        elif strategy_key == "high_rq": all_nodes.sort(key=lambda x: x[2], reverse=True)
        elif strategy_key == "low_rq": all_nodes.sort(key=lambda x: x[2])

        total_prunable_nodes = len(all_nodes)
        num_drop_total = int(round(frac_val * total_prunable_nodes))
        nodes_to_drop_info_global = all_nodes[:num_drop_total]

        if debug_mode and num_drop_total > 0:
            logger.info(f"Debug Global Pruning: Strategy '{strategy_key}', Frac {frac_val:.2f}, NumDrop {num_drop_total}")
            log_limit = min(5, num_drop_total)
            logger.info(f"  Nodes to drop (first {log_limit}): {[ (n[0], n[1], round(n[2], 4)) for n in nodes_to_drop_info_global[:log_limit]]}")
            if num_drop_total > log_limit:
                logger.info(f"  Nodes to drop (last {log_limit}): {[ (n[0], n[1], round(n[2], 4)) for n in nodes_to_drop_info_global[-log_limit:]]}")

        drop_by_layer_indices_map: Dict[str, List[int]] = {}
        for layer_name_str_global, n_idx_global, score_val_global in nodes_to_drop_info_global:
            drop_by_layer_indices_map.setdefault(layer_name_str_global, []).append(n_idx_global)
            if layer_name_str_global not in pruning_details_for_this_call["layer_info"]:
                total_nodes = scores_by_layer[layer_name_str_global].shape[0] if layer_name_str_global in scores_by_layer and scores_by_layer[layer_name_str_global] is not None else 0
                pruning_details_for_this_call["layer_info"][layer_name_str_global] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes,
                }
            pruning_details_for_this_call["layer_info"][layer_name_str_global]["num_dropped"] += 1
            if strategy_key != "random":
                pruning_details_for_this_call["layer_info"][layer_name_str_global]["dropped_scores_sum"] += score_val_global

        # Create a direct map from alignment_names to alignment_layers for reliable module fetching
        if not hasattr(net_to_prune, 'alignment_names') or not hasattr(net_to_prune, 'alignment_layers') or \
           len(net_to_prune.alignment_names) != len(net_to_prune.alignment_layers):
            logger.error("Global Pruning: Mismatch or missing alignment_names/alignment_layers in net_to_prune. Cannot reliably map names to modules for pruning.")
            # Populate details for all layers indicating skip if critical structure is missing
            for l_name_str_skip in scores_by_layer.keys(): # Use keys from scores_by_layer as they were intended for pruning
                if l_name_str_skip not in pruning_details_for_this_call["layer_info"]:
                    sc_tensor_skip = scores_by_layer.get(l_name_str_skip)
                    total_nodes_skip = sc_tensor_skip.shape[0] if sc_tensor_skip is not None else 0
                    pruning_details_for_this_call["layer_info"][l_name_str_skip] = {
                        "num_dropped": 0, "dropped_scores_sum": 0.0, 
                        "total_nodes_in_layer": total_nodes_skip,
                        "skipped_due_to_mapping_error": True
                    }
                else: # If already exists, just add the skipped flag
                    pruning_details_for_this_call["layer_info"][l_name_str_skip]["skipped_due_to_mapping_error"] = True
            return pruning_details_for_this_call

        module_map_from_alignment_lists = {
            name: net_to_prune.alignment_layers[i]
            for i, name in enumerate(net_to_prune.alignment_names)
        }

        for layer_name_str_to_prune, node_indices_list in drop_by_layer_indices_map.items():
            actual_layer_module = module_map_from_alignment_lists.get(layer_name_str_to_prune)
            if actual_layer_module and node_indices_list:
                _apply_pruning_to_layer_module(actual_layer_module, node_indices_list, dropout_mode_str, device)
            elif not actual_layer_module:
                # This warning should ideally not be hit if the above checks pass and layer_name_str_to_prune was derived from scores_by_layer.keys()
                # which should align with alignment_names.
                logger.warning(f"Global Pruning: Could not find module for layer name '{layer_name_str_to_prune}' using alignment_names mapping. This might indicate an inconsistency.")
                # Update details for this specific layer if it was targeted for pruning but module not found
                if layer_name_str_to_prune in pruning_details_for_this_call["layer_info"]:
                    pruning_details_for_this_call["layer_info"][layer_name_str_to_prune]["skipped_module_not_found_in_map"] = True
                else: # Should already be in details from earlier loop, but as a fallback
                    pruning_details_for_this_call["layer_info"][layer_name_str_to_prune] = {
                        "num_dropped": 0, "dropped_scores_sum": 0.0, 
                        "total_nodes_in_layer": scores_by_layer.get(layer_name_str_to_prune).shape[0] if scores_by_layer.get(layer_name_str_to_prune) is not None else 0,
                        "skipped_module_not_found_in_map": True
                    }

    elif pruning_mode in ["layer_wise", "layer_isolated"]:
        if debug_mode: logger.info(f"Debug Layer-Wise/Isolated Pruning: Strategy '{strategy_key}', Frac {frac_val:.2f}")
        
        for l_i, layer_mod in enumerate(net_to_prune.alignment_layers):
            layer_name_str = net_to_prune.alignment_names[l_i]

            if exclude_classification_layer and layer_name_str == classification_layer_name:
                if debug_mode:
                    logger.info(f"Layer-Wise/Isolated Pruning: Skipping layer {l_i} (classification layer) '{layer_name_str}' entirely.")
                total_nodes = layer_mod.weight.data.shape[0] if hasattr(layer_mod, 'weight') and layer_mod.weight is not None else 0
                pruning_details_for_this_call["layer_info"][layer_name_str] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes, "skipped": True,
                }
                continue
            
            current_layer_scores = scores_by_layer.get(layer_name_str)
            if current_layer_scores is None or current_layer_scores.numel() == 0:
                logger.warning(f"Layer-Wise Pruning: Scores missing or empty for layer '{layer_name_str}'. Skipping.")
                total_nodes = layer_mod.weight.data.shape[0] if hasattr(layer_mod, 'weight') and layer_mod.weight is not None else 0
                pruning_details_for_this_call["layer_info"][layer_name_str] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes, "skipped_due_to_missing_scores": True,
                }
                continue

            out_dim = layer_mod.weight.data.shape[0]
            n_drop = int(round(frac_val * out_dim))
            pruning_details_for_this_call["layer_info"][layer_name_str] = {"num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": out_dim}

            if n_drop <= 0:
                continue
            
            indices_for_layer: Optional[torch.Tensor] = None
            if strategy_key == "high_rq": indices_for_layer = descend_indices.get(layer_name_str)
            elif strategy_key == "low_rq": indices_for_layer = ascend_indices.get(layer_name_str)
            else: indices_for_layer = random_indices.get(layer_name_str)

            if indices_for_layer is None or indices_for_layer.numel() == 0:
                logger.warning(f"Layer-Wise Pruning: Sorted indices not found or empty for layer '{layer_name_str}' (strategy: {strategy_key}). Skipping pruning for this layer.")
                continue
                
            pruned_node_indices_for_layer = indices_for_layer[:n_drop]

            if debug_mode and pruned_node_indices_for_layer.numel() > 0:
                indices_to_log = (
                    pruned_node_indices_for_layer.tolist()
                    if isinstance(pruned_node_indices_for_layer, torch.Tensor)
                    else pruned_node_indices_for_layer
                )
                log_limit_layer = min(5, len(indices_to_log))
                scores_for_log = []
                if layer_name_str in scores_by_layer and scores_by_layer[layer_name_str] is not None:
                    layer_scores_tensor = scores_by_layer[layer_name_str]
                    try:
                        scores_for_log = [round(layer_scores_tensor[idx].item(), 4) for idx in indices_to_log[:log_limit_layer]]
                    except IndexError:
                        logger.warning(f"Debug Layer-Wise: Index out of bounds while fetching scores for logging layer {layer_name_str}")
                        scores_for_log = ["N/A"] * log_limit_layer
                else:
                    scores_for_log = ["N/A"] * log_limit_layer
                logger.info(
                    f"  Layer {layer_name_str} (idx {l_i}): Dropping {len(indices_to_log)} nodes. Indices (first {log_limit_layer}): {indices_to_log[:log_limit_layer]} with scores: {scores_for_log}"
                )
            
            _apply_pruning_to_layer_module(layer_mod, pruned_node_indices_for_layer, dropout_mode_str, device, out_dim_for_scaling=out_dim)
            
            # Update details
            pruning_details_for_this_call["layer_info"][layer_name_str]["num_dropped"] = pruned_node_indices_for_layer.numel()
            if strategy_key != "random" and current_layer_scores is not None and pruned_node_indices_for_layer.numel() > 0:
                try:
                    pruning_details_for_this_call["layer_info"][layer_name_str]["dropped_scores_sum"] = current_layer_scores[pruned_node_indices_for_layer].sum().item()
                except IndexError:
                    logger.warning(f"Index error summing scores for layer {layer_name_str} in {strategy_key}")

    elif pruning_mode == "cascading_layer":
        logger.warning("Cascading_layer pruning logic within _apply_pruning_to_single_net is complex and not fully refactored to use _apply_pruning_to_layer_module in this pass.")
        for l_name in net_to_prune.alignment_names:
            if l_name not in pruning_details_for_this_call["layer_info"]:
                 sc_tensor_temp = scores_by_layer.get(l_name)
                 total_nodes_temp = sc_tensor_temp.shape[0] if sc_tensor_temp is not None else net_to_prune.alignment_layers[net_to_prune.alignment_names.index(l_name)].weight.shape[0]
                 pruning_details_for_this_call["layer_info"][l_name] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes_temp, "skipped_cascading_placeholder": True
                }
    else:
        logger.warning(f"Unrecognized pruning_mode={pruning_mode} in _apply_pruning_to_single_net.")
        for l_name in net_to_prune.alignment_names:
            if l_name not in pruning_details_for_this_call["layer_info"]:
                sc_tensor_temp = scores_by_layer.get(l_name)
                total_nodes_temp = sc_tensor_temp.shape[0] if sc_tensor_temp is not None else net_to_prune.alignment_layers[net_to_prune.alignment_names.index(l_name)].weight.shape[0]
                pruning_details_for_this_call["layer_info"][l_name] = {
                    "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes_temp
                }

    return pruning_details_for_this_call


def _apply_pruning_to_layer_module(
    layer_mod: nn.Module,
    indices_to_drop: Union[List[int], torch.Tensor],
    dropout_mode_str: str,
    device: torch.device,
    out_dim_for_scaling: Optional[int] = None
) -> None:
    if not hasattr(layer_mod, 'weight') or layer_mod.weight is None:
        logger.warning(f"Layer {layer_mod} has no weight attribute or weight is None. Skipping pruning.")
        return

    wdat = layer_mod.weight.data
    actual_out_dim = wdat.shape[0]
    
    if isinstance(indices_to_drop, list):
        if not indices_to_drop:
            return
        indices_to_drop_tensor = torch.tensor(indices_to_drop, device=device, dtype=torch.long)
    elif isinstance(indices_to_drop, torch.Tensor):
        indices_to_drop_tensor = indices_to_drop.to(device=device, dtype=torch.long)
    else:
        logger.error(f"_apply_pruning_to_layer_module: indices_to_drop must be a list or tensor, got {type(indices_to_drop)}")
        return

    if indices_to_drop_tensor.numel() == 0:
        return

    mask = _create_mask_from_indices(wdat.shape, indices_to_drop_tensor, device)
    
    if dropout_mode_str == "scaled":
        reference_out_dim = out_dim_for_scaling if out_dim_for_scaling is not None else actual_out_dim
        if reference_out_dim == 0:
            logger.warning("Reference output dimension for scaling is 0. Skipping scaling.")
            scale = 1.0
        else:
            valid_indices_for_fraction = indices_to_drop_tensor[indices_to_drop_tensor < reference_out_dim]
            num_actually_dropped_for_scaling = valid_indices_for_fraction.numel()
            
            fraction_dropped_layer = num_actually_dropped_for_scaling / float(reference_out_dim)
            scale = 1.0 / (1.0 - fraction_dropped_layer) if fraction_dropped_layer < 0.9999 else 10.0
        
        layer_mod.weight.data.mul_(mask).mul_(scale)
        if layer_mod.bias is not None:
            bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, indices_to_drop_tensor, device)
            layer_mod.bias.data.mul_(bias_mask).mul_(scale)
    else:
        layer_mod.weight.data.mul_(mask)
        if layer_mod.bias is not None:
            bias_mask = _create_mask_from_indices(layer_mod.bias.data.shape, indices_to_drop_tensor, device)
            layer_mod.bias.data.mul_(bias_mask)
