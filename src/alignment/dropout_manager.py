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
import os
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from alignment.metrics import AlignmentMetric, compute_all_node_scores
from alignment.dropout import progressive_dropout, eigenvector_dropout, _normalize_device, _create_mask_from_indices, _evaluate_model_accuracy, _ensure_model_on_device, _apply_pruning_to_layer_module
from alignment.utils.evaluation import evaluate_networks, evaluate_on_loader, evaluate_networks_ensemble

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
    
    # --- Potential Fix: Limit PyTorch internal threading when using Python threads --- 
    try:
        previous_torch_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        logger.info(f"Set PyTorch num_threads to 1 for layer-isolated experiment. Previous: {previous_torch_threads}")
    except Exception as e:
        logger.warning(f"Could not set PyTorch num_threads: {e}")
    # --- End Potential Fix ---

    device = _normalize_device(device)

    num_original_networks = len(original_networks)
    if num_original_networks == 0:
        logger.warning("No networks provided for layer isolated experiment.")
        return {}

    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies_isolated": {},
        "stds_isolated": {},
        "pre_pruning_layer_stats": {},
        "pruning_details": {st: {0: {}} for st in ["high_rq", "low_rq", "random"]} # Store avg details under a single net_idx like 0
    }

    # Initialize the pruning_details structure for each fraction and layer for averaging later
    for strategy in ["high_rq", "low_rq", "random"]:
        for frac_idx in range(len(dropout_fractions) -1): # For non-baseline fractions where pruning happens
            results["pruning_details"][strategy][0][frac_idx] = {}
            for layer_name_key_init in original_networks[0].alignment_names if hasattr(original_networks[0], 'alignment_names') else [name for name, _ in original_networks[0].named_modules() if hasattr(_, 'weight')]:
                results["pruning_details"][strategy][0][frac_idx][layer_name_key_init] = {
                    "num_dropped_sum_reps": 0,
                    "dropped_scores_sum_total_reps": 0.0,
                    "total_nodes_in_layer_sum_reps": 0, # Will be same for all reps, but good to sum and avg
                    "replicates_contributing": 0
                }

    all_network_metrics_precomputed = [] 
    pre_pruning_stats_accumulator = {}

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

        # Accumulate pre-pruning stats for this replicate
        for l_name_stat, score_tensor_stat in scores_this_rep_single_metric.items():
            if score_tensor_stat is not None and isinstance(score_tensor_stat, torch.Tensor) and score_tensor_stat.numel() > 0:
                if l_name_stat not in pre_pruning_stats_accumulator:
                    pre_pruning_stats_accumulator[l_name_stat] = {"means": [], "stds": []}
                pre_pruning_stats_accumulator[l_name_stat]["means"].append(torch.mean(score_tensor_stat).item())
                pre_pruning_stats_accumulator[l_name_stat]["stds"].append(torch.std(score_tensor_stat).item())

    # Store the averaged pre_pruning_layer_stats in results
    for l_name_stat_avg, stats_lists_avg in pre_pruning_stats_accumulator.items():
        results["pre_pruning_layer_stats"][l_name_stat_avg] = {
            "avg_mean_rq": np.mean(stats_lists_avg["means"]) if stats_lists_avg["means"] else np.nan,
            "avg_std_rq": np.mean(stats_lists_avg["stds"]) if stats_lists_avg["stds"] else np.nan
        }

    if not all_network_metrics_precomputed or not all_network_metrics_precomputed[0]["scores"]:
        logger.error("Failed to pre-compute metrics for any network.")
        return {}
    
    # Use alignment_names from the first precomputed network to determine num_layers consistently
    # This assumes all networks have the same alignment_names structure if more than one original_network
    first_net_alignment_names = all_network_metrics_precomputed[0]["alignment_names"]
    classification_layer_name = first_net_alignment_names[-1] if first_net_alignment_names else None

    strategies_to_run = ["high_rq", "low_rq", "random"]

    # --- Sequential Layer Isolation --- 
    logger.info("Starting sequential layer isolation logic.")

    for strategy in strategies_to_run:
        results["accuracies_isolated"][strategy] = {layer_name: [] for layer_name in first_net_alignment_names}
        results["stds_isolated"][strategy] = {layer_name: [] for layer_name in first_net_alignment_names}

        pbar_desc = f"Isolating Layers (Strat: {strategy})"
        layer_iterator = tqdm(first_net_alignment_names, desc=pbar_desc, disable=not show_progress, leave=False)
        
        for layer_to_isolate_name_str in layer_iterator:
            if show_progress: layer_iterator.set_postfix_str(f"Processing {layer_to_isolate_name_str}")

            if exclude_classification_layer_config and layer_to_isolate_name_str == classification_layer_name:
                baseline_accuracies_replicates_cls = []
                # Ensure _shared_baseline_cls_accs is an attribute of the function for persistence across calls if needed,
                # or compute fresh if this function is always called once per experiment.
                # For simplicity, assume it's okay to re-evaluate original networks if this branch is hit multiple times (e.g. per strategy)
                # though ideally, this baseline is computed once for the whole experiment.
                # However, the structure here is per-strategy, so let's get baseline for all original nets.
                if not hasattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}'):
                    if debug_mode: logger.debug(f"[{strategy}] Evaluating all original networks for baseline once.")
                    _, baselines = evaluate_networks_ensemble(original_networks, dataset.test_loader, device, show_batch_progress=False)
                    setattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}', baselines)
                
                baseline_accuracies_replicates_cls = getattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}')
                avg_baseline_acc = np.mean(baseline_accuracies_replicates_cls) if baseline_accuracies_replicates_cls else np.nan
                std_baseline_acc = np.std(baseline_accuracies_replicates_cls) if baseline_accuracies_replicates_cls else np.nan
                results["accuracies_isolated"][strategy][layer_to_isolate_name_str] = [avg_baseline_acc] * len(dropout_fractions)
                results["stds_isolated"][strategy][layer_to_isolate_name_str] = [std_baseline_acc] * len(dropout_fractions)
                if show_progress: layer_iterator.set_postfix_str(f"{layer_to_isolate_name_str} (CLS baseline: {avg_baseline_acc:.2f}%)")
                continue

            # To store results for the current (layer_to_isolate, strategy) across all fractions
            mean_accuracies_for_this_layer_strat = []
            std_accuracies_for_this_layer_strat = []

            # Handle fraction 0.0 (baseline for this isolated layer context)
            # This means evaluating all original networks without any pruning.
            # This baseline should be the same for all isolated layers (when frac=0 for that layer).
            # We can reuse the _shared_baseline_accs_strat_{strategy} if available and computed above.
            if hasattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}'):
                baselines_for_0_frac = getattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}')
            else: # Should have been computed if CLS layer was hit, but compute if not.
                if debug_mode: logger.debug(f"[{strategy}] Evaluating all original networks for 0.0 fraction baseline.")
                _, baselines_for_0_frac = evaluate_networks_ensemble(original_networks, dataset.test_loader, device, show_batch_progress=False)
                setattr(run_layer_isolated_dropout_experiment, f'_shared_baseline_accs_strat_{strategy}', baselines_for_0_frac)

            mean_accuracies_for_this_layer_strat.append(np.mean(baselines_for_0_frac) if baselines_for_0_frac else np.nan)
            std_accuracies_for_this_layer_strat.append(np.std(baselines_for_0_frac) if baselines_for_0_frac else np.nan)

            # Iterate through non-zero dropout fractions
            for frac_val in dropout_fractions[1:]: 
                if debug_mode: logger.debug(f"[{strategy}/{layer_to_isolate_name_str}] Processing frac {frac_val:.2f}")
                replicates_pruned_for_this_fraction = []
                valid_replicate_indices_for_this_fraction = [] # To map results back if some reps fail

                for net_rep_idx in range(num_original_networks):
                    if debug_mode: task_id_str_rep = f"Net {net_rep_idx}, Layer {layer_to_isolate_name_str}, Strat {strategy}, Frac {frac_val:.2f}"
                    
                    t_start_deepcopy_frac = time.time()
                    net_copy_pruning = copy.deepcopy(original_networks[net_rep_idx])
                    if debug_mode: logger.debug(f"[{task_id_str_rep}] Deepcopy time: {time.time() - t_start_deepcopy_frac:.4f}s")
                    
                    _ensure_model_on_device(net_copy_pruning, device)
                    net_copy_pruning.eval()

                    metrics_for_this_rep = all_network_metrics_precomputed[net_rep_idx]
                    target_layer_module_prune = None
                    if not hasattr(net_copy_pruning, 'alignment_names') or not hasattr(net_copy_pruning, 'alignment_layers') or \
                       len(net_copy_pruning.alignment_names) != len(net_copy_pruning.alignment_layers):
                        logger.error(f"[{task_id_str_rep}] Mismatch in alignment attributes. Skipping this replicate for this fraction.")
                        # replicates_pruned_for_this_fraction.append(None) # Placeholder for failed rep
                        continue # Skip this replicate
                    
                    module_map_copy = {name: net_copy_pruning.alignment_layers[i] for i, name in enumerate(net_copy_pruning.alignment_names)}
                    target_layer_module_prune = module_map_copy.get(layer_to_isolate_name_str)

                    if not target_layer_module_prune or not hasattr(target_layer_module_prune, 'weight') or target_layer_module_prune.weight is None:
                        logger.warning(f"[{task_id_str_rep}] Target layer not found/no weights. Adding unpruned network to batch for this fraction.")
                        # Fallback: add the unpruned copy to the batch to get its accuracy
                        replicates_pruned_for_this_fraction.append(net_copy_pruning) 
                        valid_replicate_indices_for_this_fraction.append(net_rep_idx)
                        continue # Skip pruning for this rep, use original

                    out_dim = target_layer_module_prune.weight.data.shape[0]
                    n_drop = int(round(frac_val * out_dim))

                    if n_drop > 0:
                        indices_map = {
                            "high_rq": metrics_for_this_rep["desc_indices"],
                            "low_rq": metrics_for_this_rep["asc_indices"],
                            "random": metrics_for_this_rep["rand_indices"]
                        }
                        sorted_indices_for_layer = indices_map[strategy].get(layer_to_isolate_name_str)
                        
                        if sorted_indices_for_layer is not None and sorted_indices_for_layer.numel() > 0:
                            indices_to_drop = sorted_indices_for_layer[:n_drop]
                            # Record pruning details for this replicate, for this isolated layer
                            # These will be aggregated later.
                            current_layer_scores = metrics_for_this_rep["scores"].get(layer_to_isolate_name_str)
                            scores_of_dropped_nodes_sum = 0.0
                            if current_layer_scores is not None and strategy != "random":
                                try:
                                    scores_of_dropped_nodes_sum = current_layer_scores[indices_to_drop].sum().item()
                                except IndexError:
                                    logger.warning(f"[{task_id_str_rep}] Index error summing scores for dropped nodes.")
                            
                            # Store details for later aggregation
                            frac_idx_for_details = dropout_fractions.index(frac_val) -1 # 0-indexed for pruned_fractions
                            if frac_idx_for_details >=0:
                                layer_details_agg = results["pruning_details"][strategy][0][frac_idx_for_details].setdefault(layer_to_isolate_name_str, {
                                    "num_dropped_sum_reps": 0,
                                    "dropped_scores_sum_total_reps": 0.0,
                                    "total_nodes_in_layer_sum_reps": 0,
                                    "replicates_contributing": 0
                                })
                                layer_details_agg["num_dropped_sum_reps"] += n_drop
                                layer_details_agg["dropped_scores_sum_total_reps"] += scores_of_dropped_nodes_sum
                                layer_details_agg["total_nodes_in_layer_sum_reps"] += out_dim
                                layer_details_agg["replicates_contributing"] += 1

                            t_start_prune_apply = time.time()
                            _apply_pruning_to_layer_module(target_layer_module_prune, indices_to_drop, dropout_mode, device, out_dim_for_scaling=out_dim)
                            if debug_mode: logger.debug(f"[{task_id_str_rep}] Apply Pruning time: {time.time() - t_start_prune_apply:.4f}s")
                        elif debug_mode:
                             logger.warning(f"[{task_id_str_rep}] No valid sorted indices. No pruning applied for frac {frac_val}.")
                    
                    replicates_pruned_for_this_fraction.append(net_copy_pruning)
                    valid_replicate_indices_for_this_fraction.append(net_rep_idx)
                
                # Batch evaluate all replicates for this fraction
                if replicates_pruned_for_this_fraction:
                    if debug_mode: logger.debug(f"[{strategy}/{layer_to_isolate_name_str}/Frac {frac_val:.2f}] Evaluating batch of {len(replicates_pruned_for_this_fraction)} replicates.")
                    t_start_eval_batch = time.time()
                    _, accuracies_batch = evaluate_networks_ensemble(replicates_pruned_for_this_fraction, dataset.test_loader, device, show_batch_progress=False)
                    if debug_mode: logger.debug(f"[{strategy}/{layer_to_isolate_name_str}/Frac {frac_val:.2f}] Batch eval time: {time.time() - t_start_eval_batch:.4f}s")
                    
                    # Store results, mapping back to original number of replicates with NaNs if any failed
                    all_replicate_accuracies_this_fraction = [np.nan] * num_original_networks
                    for i, original_idx in enumerate(valid_replicate_indices_for_this_fraction):
                        if i < len(accuracies_batch):
                             all_replicate_accuracies_this_fraction[original_idx] = accuracies_batch[i]
                    
                    mean_accuracies_for_this_layer_strat.append(np.nanmean(all_replicate_accuracies_this_fraction) if np.any(~np.isnan(all_replicate_accuracies_this_fraction)) else np.nan)
                    std_accuracies_for_this_layer_strat.append(np.nanstd(all_replicate_accuracies_this_fraction) if np.any(~np.isnan(all_replicate_accuracies_this_fraction)) else np.nan)
                else:
                    # All replicates failed for this fraction
                    mean_accuracies_for_this_layer_strat.append(np.nan)
                    std_accuracies_for_this_layer_strat.append(np.nan)
                
                # Cleanup: delete the copied networks for this fraction
                for net_to_del in replicates_pruned_for_this_fraction:
                    del net_to_del
                if debug_mode and replicates_pruned_for_this_fraction: 
                    logger.debug(f"[{strategy}/{layer_to_isolate_name_str}/Frac {frac_val:.2f}] Deleted {len(replicates_pruned_for_this_fraction)} network copies.")

            results["accuracies_isolated"][strategy][layer_to_isolate_name_str] = mean_accuracies_for_this_layer_strat
            results["stds_isolated"][strategy][layer_to_isolate_name_str] = std_accuracies_for_this_layer_strat
            if show_progress: layer_iterator.set_postfix_str(f"{layer_to_isolate_name_str} done. Last mean acc: {mean_accuracies_for_this_layer_strat[-1]:.2f}%" if mean_accuracies_for_this_layer_strat else f"{layer_to_isolate_name_str} done.")
    
    # Finalize aggregated pruning_details (calculate averages)
    for strategy in strategies_to_run:
        for frac_idx in range(len(dropout_fractions) - 1):
            if frac_idx in results["pruning_details"][strategy][0]:
                for layer_name_key_finalize in results["pruning_details"][strategy][0][frac_idx]:
                    agg_data = results["pruning_details"][strategy][0][frac_idx][layer_name_key_finalize]
                    num_reps_contrib = agg_data.pop("replicates_contributing", 0)
                    if num_reps_contrib > 0:
                        final_num_dropped = agg_data.pop("num_dropped_sum_reps",0) / num_reps_contrib
                        final_dropped_scores_sum = agg_data.pop("dropped_scores_sum_total_reps", 0.0) / num_reps_contrib
                        final_total_nodes = agg_data.pop("total_nodes_in_layer_sum_reps", 0) / num_reps_contrib
                        
                        results["pruning_details"][strategy][0][frac_idx][layer_name_key_finalize] = {
                            "num_dropped": final_num_dropped,
                            "dropped_scores_sum": final_dropped_scores_sum,
                            "total_nodes_in_layer": final_total_nodes
                        }
                    else: # No replicates contributed, fill with skippable data
                         results["pruning_details"][strategy][0][frac_idx][layer_name_key_finalize] = {
                            "num_dropped": 0,
                            "dropped_scores_sum": 0.0,
                            "total_nodes_in_layer": 0, # Or get from pre_pruning_stats if more robust
                            "skipped": True
                        }
            # For other layers not isolated in this frac_idx (which is all others for layer_isolated)
            # their num_dropped should be 0. The plotting function should handle this by seeing num_dropped = 0.
            # The current structure will only have the isolated layer's details for each frac_idx.
            # plot_mean_rq_of_pruned_nodes expects data for all layers for each frac_idx.
            # We need to ensure all layers are present in the dict for each frac_idx.
            if frac_idx in results["pruning_details"][strategy][0]:
                all_layers_for_frac = results["pruning_details"][strategy][0][frac_idx]
                for layer_name_fill in first_net_alignment_names:
                    if layer_name_fill not in all_layers_for_frac:
                        # This layer was not the isolated one for any replicate at this frac_idx (which is expected)
                        # Add placeholder data for non-isolated layers
                        # Get total_nodes from pre_pruning_stats if available
                        total_nodes_val = 0
                        if pre_pruning_stats_accumulator.get(layer_name_fill):
                           # This is not ideal, as pre_pruning_stats_accumulator might not be fully reflective of a single rep's total nodes
                           # A better approach might be to access it via all_network_metrics_precomputed[0]["scores"][layer_name_fill].shape[0]
                           # For simplicity here, let's use what's available or default to 0
                           first_rep_scores = all_network_metrics_precomputed[0]["scores"]
                           if layer_name_fill in first_rep_scores and first_rep_scores[layer_name_fill] is not None:
                               total_nodes_val = first_rep_scores[layer_name_fill].shape[0]

                        all_layers_for_frac[layer_name_fill] = {
                            "num_dropped": 0,
                            "dropped_scores_sum": 0.0,
                            "total_nodes_in_layer": total_nodes_val,
                            "skipped": False # Not skipped in the sense of error, just not targeted
                        }

    # Restore PyTorch threads if changed
    if 'previous_torch_threads' in locals():
        try:
            torch.set_num_threads(previous_torch_threads)
            logger.info(f"Restored PyTorch num_threads to {previous_torch_threads}.")
        except Exception as e:
            logger.warning(f"Could not restore PyTorch num_threads: {e}")

    return results


def run_cascading_layer_pruning_experiment(
    networks: List[nn.Module],
    dataset,
    dropout_fractions: List[float],
    metric_instance: AlignmentMetric,
    device="cuda",
    dropout_mode: str = "scaled",
    show_progress: bool = True,
    debug_mode: bool = False,
    exclude_classification_layer_config: bool = True,
    num_batches_for_pre_scoring: Optional[int] = 5, 
    force_cpu_for_large_metric_ops: bool = True,
    configured_cnn_mode: Optional[str] = "unfold",
    configured_cnn_rq_op: Optional[str] = "mean"
) -> Dict:
    """
    Run cascading layer pruning experiment.
    Placeholder for now. True cascading requires dynamic score re-evaluation.
    This initial version will implement a "Simple Cascade": 
    prune layers sequentially based on original scores.
    """
    logger.info("Starting (Placeholder) Cascading Layer Pruning Experiment Manager")
    device = _normalize_device(device)
    
    strategies_to_run_cascade = ["high_rq", "low_rq", "random"]

    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies": {st: [] for st in strategies_to_run_cascade}, 
        "stds": {st: [] for st in strategies_to_run_cascade},
        "pruning_details": {st: {0: {}} for st in strategies_to_run_cascade}, 
        "pre_pruning_layer_stats": {}
    }

    if not networks:
        logger.warning("No networks provided to run_cascading_layer_pruning_experiment.")
        return results

    # --- 1. Pre-compute scores and indices for ALL networks ONCE (similar to progressive_dropout) ---
    all_networks_scores_by_layer_list = []
    all_networks_ascend_indices_list = []
    all_networks_descend_indices_list = []
    all_networks_random_indices_list = []
    pre_pruning_stats_accumulator = {}

    logger.info(f"Cascading Pruning: Pre-computing scores & indices for {len(networks)} network replicates.")
    metric_config_for_cascading = [{
        "name": metric_instance.name, 
        "scale_by_norm": metric_instance.scale_by_norm,
        "force_cpu_for_large_metric_ops": force_cpu_for_large_metric_ops,
        "configured_cnn_mode": configured_cnn_mode,
        "configured_cnn_rq_op": configured_cnn_rq_op
    }]

    for net_idx, net_rep in enumerate(tqdm(networks, desc="Cascading: Preparing Network Metrics", disable=not show_progress)):
        _ensure_model_on_device(net_rep, device)
        net_rep.eval()
        
        scores_dict_of_dict = compute_all_node_scores(
            model=net_rep, 
            metric_configs=metric_config_for_cascading, 
            device=device, 
            data_loader=dataset.test_loader,
            num_batches=num_batches_for_pre_scoring,
            debug_mode=debug_mode
        )
        
        current_net_scores = {} 
        if scores_dict_of_dict:
            # Extract scores for the primary metric_instance.name
            for layer_name_key in scores_dict_of_dict: 
                metric_data_for_layer = scores_dict_of_dict[layer_name_key]
                if metric_instance.name in metric_data_for_layer: 
                    score_value = metric_data_for_layer[metric_instance.name]
                    if score_value is not None and isinstance(score_value, torch.Tensor) and score_value.numel() > 0 and not torch.isnan(score_value).all():
                        current_net_scores[layer_name_key] = score_value
        all_networks_scores_by_layer_list.append(current_net_scores)

        asc_indices, desc_indices, rand_indices = {}, {}, {}
        for l_name, scores_tensor in current_net_scores.items():
            if scores_tensor is not None: # Ensure scores_tensor is not None before processing
                count = scores_tensor.shape[0]
                asc_indices[l_name] = torch.argsort(scores_tensor, descending=False)
                desc_indices[l_name] = torch.argsort(scores_tensor, descending=True)
                all_layer_node_indices = list(range(count))
                random.shuffle(all_layer_node_indices)
                rand_indices[l_name] = torch.tensor(all_layer_node_indices, device=device, dtype=torch.long)
                
                if l_name not in pre_pruning_stats_accumulator:
                    pre_pruning_stats_accumulator[l_name] = {"means": [], "stds": []}
                pre_pruning_stats_accumulator[l_name]["means"].append(torch.mean(scores_tensor).item())
                pre_pruning_stats_accumulator[l_name]["stds"].append(torch.std(scores_tensor).item())
            else:
                logger.warning(f"Cascading Pruning: Scores tensor for layer '{l_name}' is None for net {net_idx}. Cannot generate sorted indices.")

        all_networks_ascend_indices_list.append(asc_indices)
        all_networks_descend_indices_list.append(desc_indices)
        all_networks_random_indices_list.append(rand_indices)

    for l_name_stat, stats_lists in pre_pruning_stats_accumulator.items():
        results["pre_pruning_layer_stats"][l_name_stat] = {
            "avg_mean_rq": np.mean(stats_lists["means"]) if stats_lists["means"] else np.nan,
            "avg_std_rq": np.mean(stats_lists["stds"]) if stats_lists["stds"] else np.nan
        }
    # --- End of Pre-computation ---

    # Main loop over strategies to apply within the cascade
    for cascade_internal_strategy in strategies_to_run_cascade:
        logger.info(f"Cascading Pruning: Running with internal strategy: {cascade_internal_strategy}")
        strategy_accuracies_all_fractions = []
        strategy_stds_all_fractions = []

        # Handle baseline (fraction 0.0) - evaluate original networks
        if 0.0 in dropout_fractions:
            if not hasattr(run_cascading_layer_pruning_experiment, '_shared_baseline_accuracies'):
                logger.info("Cascading Pruning: Evaluating baseline for original networks (once).")
                _, baseline_reps_accs = evaluate_networks_ensemble(networks, dataset.test_loader, device, show_batch_progress=False)
                run_cascading_layer_pruning_experiment._shared_baseline_accuracies = baseline_reps_accs
            baseline_replicate_accuracies = run_cascading_layer_pruning_experiment._shared_baseline_accuracies
            strategy_accuracies_all_fractions.append(np.mean(baseline_replicate_accuracies) if baseline_replicate_accuracies else np.nan)
            strategy_stds_all_fractions.append(np.std(baseline_replicate_accuracies) if baseline_replicate_accuracies else np.nan)

        fractions_to_prune = [f for f in dropout_fractions if f > 0.0]
        if not fractions_to_prune and 0.0 not in dropout_fractions: # Only if 0.0 was not even in original list
            logger.info(f"Cascading Pruning (Strat: {cascade_internal_strategy}): No dropout fractions to process.")
            # Fill this strategy's results with NaNs if no fractions at all
            results["accuracies"][cascade_internal_strategy] = [np.nan] * len(dropout_fractions)
            results["stds"][cascade_internal_strategy] = [np.nan] * len(dropout_fractions)
            continue # Next strategy
        elif not fractions_to_prune and 0.0 in dropout_fractions: # Only baseline was run
            logger.info(f"Cascading Pruning (Strat: {cascade_internal_strategy}): Only baseline fraction (0.0) processed.")
            # Ensure results for this strategy reflect only the baseline if it was the only fraction
            while len(strategy_accuracies_all_fractions) < len(dropout_fractions):
                strategy_accuracies_all_fractions.append(np.nan)
                strategy_stds_all_fractions.append(np.nan)
            results["accuracies"][cascade_internal_strategy] = strategy_accuracies_all_fractions
            results["stds"][cascade_internal_strategy] = strategy_stds_all_fractions
            continue # Next strategy

        frac_pbar_desc = f"True Cascading (Strat: {cascade_internal_strategy}) Fractions"
        frac_pbar = tqdm(fractions_to_prune, desc=frac_pbar_desc, disable=not show_progress, leave=True)
        
        for frac_idx_enum, frac_val in enumerate(frac_pbar):
            current_frac_pruning_details_for_avg = {}
            accuracies_this_fraction_all_replicates = []

            for net_idx, original_network_ref in enumerate(networks):
                if show_progress and isinstance(frac_pbar, tqdm):
                    frac_pbar.set_postfix_str(f"NetRep {net_idx+1}/{len(networks)}")

                net_copy = copy.deepcopy(original_network_ref)
                _ensure_model_on_device(net_copy, device)
                net_copy.eval()
                
                layers_to_prune_in_copy = net_copy.alignment_layers
                layer_names_in_copy = net_copy.alignment_names
                classification_layer_name = layer_names_in_copy[-1] if layer_names_in_copy else None
                rep_frac_pruning_details = {}

                if debug_mode: logger.debug(f"TrueCascading: Net {net_idx}, Frac {frac_val:.2f}, Strat {cascade_internal_strategy} - Starting layer cascade")

                for layer_idx_in_cascade, layer_mod_to_prune in enumerate(layers_to_prune_in_copy):
                    layer_name_actual = layer_names_in_copy[layer_idx_in_cascade]
                    task_id_str_rep_layer = f"Net {net_idx}, Layer {layer_name_actual}, Strat {cascade_internal_strategy}, Frac {frac_val:.2f}"

                    if exclude_classification_layer_config and layer_name_actual == classification_layer_name:
                        # ... (record 0-dropped details for CLS layer) ...
                        original_cls_score_tensor = all_networks_scores_by_layer_list[net_idx].get(layer_name_actual) # Use original scores for total nodes info
                        total_cls_nodes = original_cls_score_tensor.shape[0] if original_cls_score_tensor is not None else 0
                        rep_frac_pruning_details[layer_name_actual] = {
                            "num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_cls_nodes
                        }
                        continue

                    # --- TRUE CASCADE: Re-calculate scores for the current layer based on the current state of net_copy ---
                    if debug_mode: logger.debug(f"[{task_id_str_rep_layer}] Re-calculating scores for layer {layer_name_actual}")
                    current_dynamic_scores_dict_of_dict = compute_all_node_scores(
                        model=net_copy, # Pass the iteratively pruned network
                        metric_configs=metric_config_for_cascading, 
                        device=device, 
                        data_loader=dataset.test_loader, 
                        num_batches=num_batches_for_pre_scoring,
                        debug_mode=debug_mode # Can be very verbose
                    )
                    node_scores_for_this_layer_dynamically = None
                    if current_dynamic_scores_dict_of_dict and \
                       layer_name_actual in current_dynamic_scores_dict_of_dict and \
                       metric_instance.name in current_dynamic_scores_dict_of_dict[layer_name_actual]:
                        score_tensor_val_dyn = current_dynamic_scores_dict_of_dict[layer_name_actual][metric_instance.name]
                        if score_tensor_val_dyn is not None and isinstance(score_tensor_val_dyn, torch.Tensor) and score_tensor_val_dyn.numel() > 0:
                            node_scores_for_this_layer_dynamically = score_tensor_val_dyn
                    
                    if node_scores_for_this_layer_dynamically is None:
                        if debug_mode: logger.warning(f"[{task_id_str_rep_layer}] No dynamic scores found. Skipping pruning.")
                        total_nodes_val_sc = layer_mod_to_prune.weight.shape[0] if hasattr(layer_mod_to_prune, 'weight') and layer_mod_to_prune.weight is not None else 0
                        rep_frac_pruning_details[layer_name_actual] = {"num_dropped": 0, "dropped_scores_sum": 0.0, "total_nodes_in_layer": total_nodes_val_sc}
                        continue
                    # --- End TRUE CASCADE score calculation ---

                    out_dim_layer = layer_mod_to_prune.weight.shape[0]
                    n_drop_layer = int(round(frac_val * out_dim_layer))
                    num_dropped_this_layer = 0
                    sum_scores_dropped_this_layer = 0.0

                    if n_drop_layer > 0:
                        # Generate sorted indices based on these NEWLY computed dynamic scores
                        dynamic_asc_indices = torch.argsort(node_scores_for_this_layer_dynamically, descending=False)
                        dynamic_desc_indices = torch.argsort(node_scores_for_this_layer_dynamically, descending=True)
                        dynamic_rand_indices_list = list(range(node_scores_for_this_layer_dynamically.shape[0]))
                        random.shuffle(dynamic_rand_indices_list)
                        dynamic_rand_indices = torch.tensor(dynamic_rand_indices_list, device=device, dtype=torch.long)
                        
                        dynamic_indices_map = {
                            "high_rq": dynamic_desc_indices,
                            "low_rq": dynamic_asc_indices,
                            "random": dynamic_rand_indices
                        }
                        sorted_indices_for_this_layer_dynamically = dynamic_indices_map[cascade_internal_strategy]
                        
                        if sorted_indices_for_this_layer_dynamically.numel() > 0:
                            indices_to_drop_layer = sorted_indices_for_this_layer_dynamically[:n_drop_layer]
                            if cascade_internal_strategy != "random":
                                try:
                                    sum_scores_dropped_this_layer = node_scores_for_this_layer_dynamically[indices_to_drop_layer].sum().item()
                                except IndexError:
                                    logger.warning(f"[{task_id_str_rep_layer}] Index error summing dynamic scores for dropped nodes.")
                            
                            _apply_pruning_to_layer_module(layer_mod_to_prune, indices_to_drop_layer, dropout_mode, device, out_dim_for_scaling=out_dim_layer)
                            num_dropped_this_layer = len(indices_to_drop_layer)
                            if debug_mode: logger.debug(f"[{task_id_str_rep_layer}] Pruned {num_dropped_this_layer}/{out_dim_layer} nodes using dynamic scores.")
                        elif debug_mode:
                            logger.warning(f"[{task_id_str_rep_layer}] No dynamic sorted indices generated. No pruning.")
                    
                    rep_frac_pruning_details[layer_name_actual] = {
                        "num_dropped": num_dropped_this_layer,
                        "dropped_scores_sum": sum_scores_dropped_this_layer,
                        "total_nodes_in_layer": out_dim_layer
                    }
                
                # Evaluate the fully cascaded-pruned net_copy for this replicate and fraction
                current_accuracy = _evaluate_model_accuracy(net_copy, dataset.test_loader, device)
                accuracies_this_fraction_all_replicates.append(current_accuracy)

                # Aggregate pruning details for this replicate into current_frac_pruning_details_for_avg
                for layer_name_pd_rep, details_pd_rep in rep_frac_pruning_details.items():
                    if layer_name_pd_rep not in current_frac_pruning_details_for_avg:
                         # Initialize if somehow missed (e.g. a layer name not in initial first_net_alignment_names)
                        current_frac_pruning_details_for_avg[layer_name_pd_rep] = {
                            "num_dropped_sum_reps": 0, "dropped_scores_sum_total_reps": 0.0,
                            "total_nodes_in_layer_sum_reps": 0, "replicates_contributing": 0
                        }
                    current_frac_pruning_details_for_avg[layer_name_pd_rep]["num_dropped_sum_reps"] += details_pd_rep["num_dropped"]
                    current_frac_pruning_details_for_avg[layer_name_pd_rep]["dropped_scores_sum_total_reps"] += details_pd_rep["dropped_scores_sum"]
                    current_frac_pruning_details_for_avg[layer_name_pd_rep]["total_nodes_in_layer_sum_reps"] += details_pd_rep["total_nodes_in_layer"]
                    current_frac_pruning_details_for_avg[layer_name_pd_rep]["replicates_contributing"] += 1
            
            # Aggregate accuracies for this fraction
            if accuracies_this_fraction_all_replicates:
                strategy_accuracies_all_fractions.append(np.mean(accuracies_this_fraction_all_replicates))
                strategy_stds_all_fractions.append(np.std(accuracies_this_fraction_all_replicates))
            else:
                strategy_accuracies_all_fractions.append(np.nan)
                strategy_stds_all_fractions.append(np.nan)
            
            # Store/finalize aggregated pruning details for this fraction and strategy
            results["pruning_details"][cascade_internal_strategy][0][frac_idx_enum] = current_frac_pruning_details_for_avg

        results["accuracies"][cascade_internal_strategy] = strategy_accuracies_all_fractions
        results["stds"][cascade_internal_strategy] = strategy_stds_all_fractions
    
    logger.info("Cascading Layer Pruning Experiment Manager finished.")
    # Clear shared attribute after experiment run if desired, or manage its lifecycle if this func is called multiple times in one script run.
    if hasattr(run_cascading_layer_pruning_experiment, '_shared_baseline_accuracies'):
        delattr(run_cascading_layer_pruning_experiment, '_shared_baseline_accuracies')
    return results