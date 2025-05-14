"""
Dropout experiment management.

This module provides high-level functions for running dropout experiments
with multiple networks and strategies.
"""

import logging
import time
import copy
from typing import Dict, List, Tuple, Union, Optional, Any
import traceback

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from alignment.metrics import AlignmentMetric, compute_all_node_scores
from alignment.dropout import progressive_dropout, eigenvector_dropout, _normalize_device, _create_mask_from_indices, _evaluate_model_accuracy, _ensure_model_on_device, _apply_pruning_to_layer_module
from alignment.utils.evaluation import evaluate_networks, evaluate_on_loader

logger = logging.getLogger(__name__)


def run_progressive_dropout_experiment(
    networks: List[nn.Module],
    dataset,
    dropout_fractions: List[float],
    metric_instance: AlignmentMetric,
    device="cuda",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "scaled",
    show_progress: bool = True,
    debug_mode: bool = False,
    exclude_classification_layer_config: bool = True,
    num_batches_for_pre_scoring: int = 5,
    force_cpu_for_large_metric_ops: bool = True,
    configured_cnn_mode: Optional[str] = "unfold",
    configured_cnn_rq_op: Optional[str] = "mean"
) -> Dict:
    """
    Run progressive dropout experiment on multiple networks with multiple strategies.
    
    Args:
        networks: List of networks to evaluate
        dataset: Dataset object
        dropout_fractions: List of dropout fractions to test
        metric_instance: Alignment metric to use
        device: Device to run on
        pruning_mode: Pruning mode to use
        dropout_mode: Dropout mode to use
        show_progress: Whether to show progress bars
        debug_mode: Whether to print additional debug information
        exclude_classification_layer_config: Whether to exclude the classification layer from the experiment
        num_batches_for_pre_scoring: Number of batches to use for pre-scoring
        force_cpu_for_large_metric_ops: Whether to force CPU for large metric operations
        configured_cnn_mode: Optional CNN metric computation mode
        configured_cnn_rq_op: Optional CNN RQ operation
        
    Returns:
        Dictionary with dropout experiment results
    """
    device = _normalize_device(device)
    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies": {"high_rq": [], "low_rq": [], "random": []},
        "losses": {"high_rq": [], "low_rq": [], "random": []},
        "stds": {"high_rq": [], "low_rq": [], "random": []},
        "pruning_details": {st: {} for st in ["high_rq", "low_rq", "random"]},
        "pre_pruning_layer_stats": {}
    }

    if not networks:
        logger.warning("No networks provided to run_progressive_dropout_experiment.")
        return results

    # --- Pre-compute scores and indices for ALL networks ONCE --- 
    all_networks_scores_by_layer_list = []
    all_networks_ascend_indices_list = []
    all_networks_descend_indices_list = []
    all_networks_random_indices_list = []
    
    pre_pruning_stats_accumulator = {}

    logger.info(f"Pre-computing scores & indices for {len(networks)} network replicates.")
    # Construct metric_configs for the single metric used by this experiment
    # Pass down additional relevant configs for metric computation.
    metric_config_for_pruning = [{
        "name": metric_instance.name, 
        "scale_by_norm": metric_instance.scale_by_norm,
        "force_cpu_for_large_metric_ops": force_cpu_for_large_metric_ops,
        "configured_cnn_mode": configured_cnn_mode,
        "configured_cnn_rq_op": configured_cnn_rq_op
    }]

    for net_idx, net_rep in enumerate(tqdm(networks, desc="Preparing Network Metrics", disable=not show_progress)):
        _ensure_model_on_device(net_rep, device)
        net_rep.eval()
        
        scores_dict_of_dict = compute_all_node_scores(
            model=net_rep, 
            metric_configs=metric_config_for_pruning, 
            device=device, 
            data_loader=dataset.test_loader,
            num_batches=num_batches_for_pre_scoring,
            debug_mode=debug_mode
        )
        
        # +++ START NEW DETAILED DEBUG LOGGING +++
        if debug_mode:
            logger.debug(f"Dropout Manager (Net {net_idx}): Received scores_dict_of_dict from compute_all_node_scores:")
            if not scores_dict_of_dict:
                logger.debug("  scores_dict_of_dict is EMPTY.")
            else:
                for l_name_debug, m_dict_debug in scores_dict_of_dict.items():
                    logger.debug(f"  Layer '{l_name_debug}': Contains metrics: {list(m_dict_debug.keys())}")
                    for m_name_debug, val_debug in m_dict_debug.items():
                        val_type = type(val_debug)
                        val_shape = "N/A"
                        is_valid_tensor = False
                        if isinstance(val_debug, torch.Tensor):
                            val_shape = str(val_debug.shape)
                            is_valid_tensor = val_debug.numel() > 0 and not torch.isnan(val_debug).all()
                        logger.debug(f"    Metric '{m_name_debug}': type={val_type}, shape={val_shape}, is_valid_tensor_for_pruning={is_valid_tensor}")
        # +++ END NEW DETAILED DEBUG LOGGING +++
        
        current_net_scores = {} 
        if scores_dict_of_dict:
            metric_found = False
            for layer_idx_key in scores_dict_of_dict: 
                metric_data_for_layer = scores_dict_of_dict[layer_idx_key]
                
                if metric_instance.name in metric_data_for_layer: 
                    score_value = metric_data_for_layer[metric_instance.name]
                    if score_value is not None and isinstance(score_value, torch.Tensor) and score_value.numel() > 0 and not torch.isnan(score_value).all():
                        current_net_scores[layer_idx_key] = score_value
                        metric_found = True
                        if debug_mode:
                            logger.debug(f"Dropout Manager: Found valid score for metric '{metric_instance.name}' in layer '{layer_idx_key}'.")
                    else:
                        current_net_scores[layer_idx_key] = None 
                        if debug_mode:
                            logger.debug(f"Dropout Manager: Score for metric '{metric_instance.name}' in layer '{layer_idx_key}' is None, not a tensor, empty, or all NaNs.")
                else:
                    current_net_scores[layer_idx_key] = None
                    if debug_mode:
                        logger.debug(f"Dropout Manager: Metric '{metric_instance.name}' (expected) not found in scores for layer '{layer_idx_key}'. Available metrics: {list(metric_data_for_layer.keys())}")
            
            if not metric_found: 
                logger.warning(f"Progressive Dropout: Primary metric '{metric_instance.name}' not found with valid scores in any layer for network {net_idx}. Attempting fallback (fallback logic not shown here but was empty).")
        else: 
            logger.error(f"Progressive Dropout: compute_all_node_scores returned an empty dictionary for network {net_idx}.")

        all_networks_scores_by_layer_list.append(current_net_scores)

        asc_indices, desc_indices, rand_indices = {}, {}, {}
        for l_idx, scores_tensor in current_net_scores.items():
            count = scores_tensor.shape[0]
            asc_indices[l_idx] = torch.argsort(scores_tensor, descending=False)
            desc_indices[l_idx] = torch.argsort(scores_tensor, descending=True)
            all_layer_node_indices = list(range(count))
            random.shuffle(all_layer_node_indices)
            rand_indices[l_idx] = torch.tensor(all_layer_node_indices, device=device, dtype=torch.long)
            
            if l_idx not in pre_pruning_stats_accumulator:
                pre_pruning_stats_accumulator[l_idx] = {"means": [], "stds": []}
            pre_pruning_stats_accumulator[l_idx]["means"].append(torch.mean(scores_tensor).item())
            pre_pruning_stats_accumulator[l_idx]["stds"].append(torch.std(scores_tensor).item())

        all_networks_ascend_indices_list.append(asc_indices)
        all_networks_descend_indices_list.append(desc_indices)
        all_networks_random_indices_list.append(rand_indices)

    for l_idx, stats_lists in pre_pruning_stats_accumulator.items():
        results["pre_pruning_layer_stats"][l_idx] = {
            "avg_mean_rq": np.mean(stats_lists["means"]) if stats_lists["means"] else np.nan,
            "avg_std_rq": np.mean(stats_lists["stds"]) if stats_lists["stds"] else np.nan
        }
    # --- End of Pre-computation --- 

    try:
        strategies = ["high_rq", "low_rq", "random"]
        # Remove the tqdm wrapper here as it's no longer directly iterated in a way that updates it.
        # strategy_pbar = tqdm(strategies, desc="Processing Strategies", position=0, disable=not show_progress)
        
        # If AlignmentExperiment.use_multi_strategy_dropout is True, we can call progressive_dropout ONCE.
        # Otherwise, we loop here and call it per strategy.
        # The `use_multi_strategy_dropout` flag is in AlignmentConfig, read by AlignmentExperiment.
        # This manager function should ideally receive that flag or be simplified.
        # For now, let's assume this manager is called when multi-strategy is DESIRED for the overall experiment.
        # So, call progressive_dropout ONCE with use_multi_strategy=True.

        # Create deep copies of original networks to pass to progressive_dropout, 
        # as progressive_dropout will modify them (or its own copies).
        networks_for_pruning = [copy.deepcopy(net) for net in networks]
        for net in networks_for_pruning: # Ensure copies are on device
            _ensure_model_on_device(net, device)

        logger.info("Calling progressive_dropout with use_multi_strategy=True for all strategies.")
        accuracies_all_strategies, losses_all_strategies, details_all_strategies = progressive_dropout(
            networks_for_pruning, # Pass copies
            all_networks_scores_by_layer_list,
            all_networks_ascend_indices_list,
            all_networks_descend_indices_list,
            all_networks_random_indices_list,
            dataset=dataset,
            dropout_fractions=dropout_fractions,
            device=device,
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode,
            strategy='high_rq', # strategy arg ignored if use_multi_strategy=True
            show_progress=show_progress, 
            use_multi_strategy=True, # Key change: leverage multi-strategy path in progressive_dropout
            debug_mode=debug_mode,
            exclude_classification_layer_config=exclude_classification_layer_config
        )

        # `accuracies_all_strategies` is {strategy: {net_idx: [values_for_each_dropout_fraction]}}
        # `details_all_strategies` is {strategy: {net_idx: {frac_idx_pruned: layer_info}}}
        results["pruning_details"] = details_all_strategies

        num_fractions_total = len(dropout_fractions) # This includes the baseline 0.0

        for strategy_key in strategies: # strategies = ["high_rq", "low_rq", "random"]
            if strategy_key in accuracies_all_strategies:
                # This dict contains {net_idx: [acc_baseline, acc_frac1, acc_frac2, ...]}
                accs_per_net_for_this_strategy = accuracies_all_strategies[strategy_key]
                
                # Initialize lists for mean/std accuracies for this strategy for each fraction
                mean_accuracies_for_strategy = []
                std_accuracies_for_strategy = []

                for frac_idx_overall in range(num_fractions_total):
                    # For this specific fraction index, gather accuracies from all network replicates
                    accuracies_this_frac_all_replicates = []
                    for net_idx_rep in range(len(networks)): # Iterate 0 to num_replicates-1
                        if net_idx_rep in accs_per_net_for_this_strategy and \
                           len(accs_per_net_for_this_strategy[net_idx_rep]) > frac_idx_overall:
                            accuracies_this_frac_all_replicates.append(accs_per_net_for_this_strategy[net_idx_rep][frac_idx_overall])
                        else:
                            # This case (missing data for a specific net/fraction) should ideally not happen 
                            # if progressive_dropout populates results for all nets and all fractions (incl. baseline).
                            # Adding NaN allows np.nanmean/np.nanstd to work gracefully.
                            accuracies_this_frac_all_replicates.append(np.nan)
                            if debug_mode:
                                logger.warning(f"Missing accuracy data for strategy '{strategy_key}', net {net_idx_rep}, fraction_idx {frac_idx_overall}")

                    if accuracies_this_frac_all_replicates:
                        mean_acc = np.nanmean(accuracies_this_frac_all_replicates)
                        std_acc = np.nanstd(accuracies_this_frac_all_replicates)
                        mean_accuracies_for_strategy.append(mean_acc)
                        std_accuracies_for_strategy.append(std_acc)
                        
                        # Optional: Log losses similarly if losses_all_strategies is processed
                        # For now, focusing on accuracies and stds as per original results structure.
                        if debug_mode:
                            actual_frac_val_log = dropout_fractions[frac_idx_overall]
                            logger.info(f"  Aggregated for {strategy_key}, fraction={actual_frac_val_log:.2f}: "
                                       f"acc={mean_acc:.2f} ± {std_acc:.2f}%")
                    else:
                        # All replicates had missing data for this fraction (highly unlikely for baseline)
                        mean_accuracies_for_strategy.append(np.nan)
                        std_accuracies_for_strategy.append(np.nan)
                
                results["accuracies"][strategy_key] = mean_accuracies_for_strategy
                results["stds"][strategy_key] = std_accuracies_for_strategy
                # results["losses"][strategy_key] would be populated similarly if losses are tracked

            else:
                logger.warning(f"Strategy {strategy_key} not found in progressive_dropout results. Filling with NaNs.")
                results["accuracies"][strategy_key] = [np.nan] * num_fractions_total
                results["stds"][strategy_key] = [np.nan] * num_fractions_total

    except Exception as e:
        logger.error(f"Error in run_progressive_dropout_experiment: {str(e)}")
        logger.error(traceback.format_exc())
        results["error"] = str(e)
    
    return results


def run_eigenvector_dropout_experiment(
    network: nn.Module,
    dataset,
    dropout_fractions: List[float],
    metric,
    device="cuda",
    dropout_mode: str = "scaled",
    pruning_mode: str = "global_joint",
    show_progress: bool = True,
    debug_mode: bool = False
) -> Dict:
    """
    Run eigenvector dropout experiment on a network.
    
    Args:
        network: Network to evaluate
        dataset: Dataset object
        dropout_fractions: List of dropout fractions to test
        metric: Alignment metric to use
        device: Device to run on
        dropout_mode: Dropout mode to use
        pruning_mode: Pruning mode to use
        show_progress: Whether to show progress bars
        debug_mode: Whether to print additional debug information
        
    Returns:
        Dictionary with eigenvector dropout results
    """
    # Normalize device
    device = _normalize_device(device)
    
    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies": {"eigenvector": []},
        "losses": {"eigenvector": []},
        "alignment_values": {"eigenvector": []}
    }
    
    fraction_pbar = tqdm(dropout_fractions, desc="Eigenvector Dropout", position=0) if show_progress else dropout_fractions
    
    for dropout_fraction in fraction_pbar:
        try:
            if debug_mode:
                logger.info(f"Running eigenvector_dropout with fraction={dropout_fraction:.2f}, "
                           f"dropout_mode={dropout_mode}, pruning_mode={pruning_mode}")
                
            accuracy, alignment_values = eigenvector_dropout(
                network,
                dataset,
                dropout_fraction=dropout_fraction,
                metric=metric,
                device=device,
                dropout_mode=dropout_mode,
                dropout_pruning_mode=pruning_mode,
                debug_mode=debug_mode
            )
            
            results["accuracies"]["eigenvector"].append(accuracy)
            results["losses"]["eigenvector"].append(100.0 - accuracy)
            results["alignment_values"]["eigenvector"].append(alignment_values)
            
            if debug_mode:
                logger.info(f"  Eigenvector dropout, fraction={dropout_fraction:.2f}: acc={accuracy:.2f}%")
            
            if show_progress:
                fraction_pbar.set_postfix({"acc": f"{accuracy:.2f}%"})
            
        except Exception as e:
            logger.error(f"Error in eigenvector dropout at fraction {dropout_fraction}: {str(e)}")
            logger.error(traceback.format_exc())
            results["accuracies"]["eigenvector"].append(0.0)
            results["losses"]["eigenvector"].append(100.0)
            results["alignment_values"]["eigenvector"].append(None)
    
    return results


def run_layer_isolated_dropout_experiment(
    original_networks: List[nn.Module],
    dataset,
    dropout_fractions: List[float],
    metric: AlignmentMetric,
    device: Union[str, torch.device] = "cuda",
    dropout_mode: str = "scaled",
    show_progress: bool = True,
    debug_mode: bool = False,
    exclude_classification_layer_config: bool = True,
    num_batches_for_pre_scoring: int = 5,
    force_cpu_for_large_metric_ops: bool = True,
    configured_cnn_mode: Optional[str] = "unfold",
    configured_cnn_rq_op: Optional[str] = "mean"
) -> Dict:
    logger.info("Starting Layer Isolated Dropout Experiment")
    device = _normalize_device(device)

    num_original_networks = len(original_networks)
    if num_original_networks == 0:
        logger.warning("No networks provided for layer isolated experiment.")
        return {}

    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies_isolated": {},
        "stds_isolated": {}
    }

    all_network_metrics_precomputed = [] 
    metric_config_for_isolated_pruning = [{
        "name": metric.name, 
        "scale_by_norm": metric.scale_by_norm,
        "force_cpu_for_large_metric_ops": force_cpu_for_large_metric_ops,
        "configured_cnn_mode": configured_cnn_mode,
        "configured_cnn_rq_op": configured_cnn_rq_op
    }]

    for net_idx, net_rep in enumerate(original_networks):
        _ensure_model_on_device(net_rep, device)
        net_rep.eval()
        scores_dict_this_rep = compute_all_node_scores( 
            model=net_rep, 
            metric_configs=metric_config_for_isolated_pruning,
            device=device, 
            data_loader=dataset.test_loader,
            debug_mode=debug_mode, 
            num_batches=num_batches_for_pre_scoring
        )
        
        scores_this_rep_single_metric = {} 
        if scores_dict_this_rep:
            metric_found_isolated = False
            for layer_idx_key in scores_dict_this_rep:
                if metric.name in scores_dict_this_rep[layer_idx_key]:
                    score_value = scores_dict_this_rep[layer_idx_key][metric.name]
                    if score_value is not None and isinstance(score_value, torch.Tensor) and score_value.numel() > 0:
                         scores_this_rep_single_metric[layer_idx_key] = score_value
                         metric_found_isolated = True
                    else:
                         scores_this_rep_single_metric[layer_idx_key] = None # Store None if invalid
                else:
                    scores_this_rep_single_metric[layer_idx_key] = None # Store None if metric not found

            if not metric_found_isolated:
                logger.warning(f"Layer Isolated: Primary metric '{metric.name}' not found for network {net_idx}. Attempting fallback.")
                first_metric_name_found_iso = None
                first_layer_idx_for_fallback_iso = next(iter(scores_dict_this_rep), None)
                if first_layer_idx_for_fallback_iso is not None and scores_dict_this_rep[first_layer_idx_for_fallback_iso]:
                    first_metric_name_found_iso = next(iter(scores_dict_this_rep[first_layer_idx_for_fallback_iso].keys()), None)
                
                if first_metric_name_found_iso:
                    logger.warning(f"Layer Isolated: Using fallback metric '{first_metric_name_found_iso}' for network {net_idx}.")
                    scores_this_rep_single_metric = {} # Reset and rebuild
                    for layer_idx_key, metrics_in_layer_val in scores_dict_this_rep.items():
                        scores_this_rep_single_metric[layer_idx_key] = metrics_in_layer_val.get(first_metric_name_found_iso)
                else:
                    logger.error(f"Layer Isolated: Failed to obtain any scores for network {net_idx} for any metric.")
        else:
            logger.error(f"Layer Isolated: compute_all_node_scores returned empty dict for network {net_idx}.")
        
        asc_indices_this_rep = {}
        desc_indices_this_rep = {}
        rand_indices_this_rep = {}
        for l_idx_scores, score_tensor_val in scores_this_rep_single_metric.items(): # Renamed to avoid conflict
            if score_tensor_val is not None and isinstance(score_tensor_val, torch.Tensor) and score_tensor_val.numel() > 0:
                count = score_tensor_val.shape[0]
                asc_indices_this_rep[l_idx_scores] = torch.argsort(score_tensor_val, descending=False)
                desc_indices_this_rep[l_idx_scores] = torch.argsort(score_tensor_val, descending=True)
                allidx_layer = list(range(count))
                random.shuffle(allidx_layer)
                rand_indices_this_rep[l_idx_scores] = torch.tensor(allidx_layer, device=device, dtype=torch.long)
            else:
                # Ensure keys exist even if scores are None or invalid, to prevent KeyErrors later when accessing .get(layer_name_str)
                asc_indices_this_rep[l_idx_scores] = torch.empty(0, dtype=torch.long, device=device)
                desc_indices_this_rep[l_idx_scores] = torch.empty(0, dtype=torch.long, device=device)
                rand_indices_this_rep[l_idx_scores] = torch.empty(0, dtype=torch.long, device=device)

        all_network_metrics_precomputed.append({
            "scores": scores_this_rep_single_metric, 
            "asc_indices": asc_indices_this_rep, 
            "desc_indices": desc_indices_this_rep, 
            "rand_indices": rand_indices_this_rep,
            "alignment_names": net_rep.alignment_names if hasattr(net_rep, 'alignment_names') else [name for name, _ in net_rep.named_modules() if hasattr(_, 'weight')] # Store alignment names
        })

    if not all_network_metrics_precomputed or not all_network_metrics_precomputed[0]["scores"]:
        logger.error("Failed to pre-compute metrics for any network.")
        return {}
    
    # Use alignment_names from the first precomputed network to determine num_layers consistently
    # This assumes all networks have the same alignment_names structure if more than one original_network
    first_net_alignment_names = all_network_metrics_precomputed[0]["alignment_names"]
    num_layers = len(first_net_alignment_names)
    classification_layer_name = first_net_alignment_names[-1] if first_net_alignment_names else None

    strategies_to_run = ["high_rq", "low_rq", "random"]
    for strategy in strategies_to_run:
        results["accuracies_isolated"][strategy] = {layer_name: [] for layer_name in first_net_alignment_names}
        results["stds_isolated"][strategy] = {layer_name: [] for layer_name in first_net_alignment_names}

        # Iterate by layer_name_str using the order from the first network's alignment_names
        for layer_to_isolate_name_str in tqdm(first_net_alignment_names, desc=f"Isolating Layers (Strat: {strategy})"):
            # Find the integer index corresponding to this name for consistent iteration logic if needed
            # For this refactor, we primarily use layer_name_str directly.
            layer_to_isolate_idx = first_net_alignment_names.index(layer_to_isolate_name_str)

            if exclude_classification_layer_config and layer_to_isolate_name_str == classification_layer_name:
                # ... (baseline calculation for skipped classification layer) ...
                baseline_accuracies_replicates = [_evaluate_model_accuracy(net, dataset.test_loader, device) for net in original_networks]
                avg_baseline_acc = np.mean(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan
                std_baseline_acc = np.std(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan
                results["accuracies_isolated"][strategy][layer_to_isolate_name_str] = [avg_baseline_acc] * len(dropout_fractions)
                results["stds_isolated"][strategy][layer_to_isolate_name_str] = [std_baseline_acc] * len(dropout_fractions)
                continue

            accuracies_this_layer_all_fracs_avg_reps = []
            stds_this_layer_all_fracs_avg_reps = []

            baseline_accuracies_replicates = [_evaluate_model_accuracy(net, dataset.test_loader, device) for net in original_networks]
            accuracies_this_layer_all_fracs_avg_reps.append(np.mean(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan)
            stds_this_layer_all_fracs_avg_reps.append(np.std(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan)

            for frac_val in dropout_fractions[1:]:
                current_frac_accuracies_over_replicates = []
                for net_rep_idx in range(num_original_networks):
                    net_copy = copy.deepcopy(original_networks[net_rep_idx])
                    _ensure_model_on_device(net_copy, device)
                    net_copy.eval()
                    
                    metrics_for_this_rep = all_network_metrics_precomputed[net_rep_idx]
                    # Find the module corresponding to layer_to_isolate_name_str in net_copy
                    target_layer_module = None
                    module_map = {name: mod for name, mod in net_copy.named_modules()}
                    target_layer_module = module_map.get(layer_to_isolate_name_str)

                    if not target_layer_module or not hasattr(target_layer_module, 'weight') or target_layer_module.weight is None:
                        logger.warning(f"Layer '{layer_to_isolate_name_str}' not found or has no weights in network copy {net_rep_idx}. Skipping pruning for this rep.")
                        current_frac_accuracies_over_replicates.append(_evaluate_model_accuracy(net_copy, dataset.test_loader, device))
                        continue
                        
                    out_dim = target_layer_module.weight.data.shape[0]
                    n_drop = int(round(frac_val * out_dim))

                    # Use layer_to_isolate_name_str to access score/index dictionaries
                    if n_drop > 0 and layer_to_isolate_name_str in metrics_for_this_rep["scores"] and metrics_for_this_rep["scores"][layer_to_isolate_name_str] is not None:
                        indices_map = {
                            "high_rq": metrics_for_this_rep["desc_indices"],
                            "low_rq": metrics_for_this_rep["asc_indices"],
                            "random": metrics_for_this_rep["rand_indices"]
                        }
                        sorted_indices_for_layer = indices_map[strategy].get(layer_to_isolate_name_str)
                        
                        if sorted_indices_for_layer is not None and sorted_indices_for_layer.numel() > 0:
                            indices_to_drop = sorted_indices_for_layer[:n_drop]
                            # Call the new shared pruning utility
                            _apply_pruning_to_layer_module(target_layer_module, indices_to_drop, dropout_mode, device, out_dim_for_scaling=out_dim)
                        else:
                            if debug_mode: logger.warning(f"Layer '{layer_to_isolate_name_str}' (idx {layer_to_isolate_idx}) has no valid sorted indices for strategy {strategy}, rep {net_rep_idx}. No pruning applied.")
                    
                    current_frac_accuracies_over_replicates.append(_evaluate_model_accuracy(net_copy, dataset.test_loader, device))
                
                accuracies_this_layer_all_fracs_avg_reps.append(np.mean(current_frac_accuracies_over_replicates) if current_frac_accuracies_over_replicates else np.nan)
                stds_this_layer_all_fracs_avg_reps.append(np.std(current_frac_accuracies_over_replicates) if current_frac_accuracies_over_replicates else np.nan)
            
            results["accuracies_isolated"][strategy][layer_to_isolate_name_str] = accuracies_this_layer_all_fracs_avg_reps
            results["stds_isolated"][strategy][layer_to_isolate_name_str] = stds_this_layer_all_fracs_avg_reps

    return results