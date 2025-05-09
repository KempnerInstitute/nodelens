"""
Dropout experiment management.

This module provides high-level functions for running dropout experiments
with multiple networks and strategies.
"""

import logging
import time
import copy
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random

from alignment.metrics import AlignmentMetric
from alignment.dropout import progressive_dropout, eigenvector_dropout, _normalize_device, _compute_metric_for_all_nodes, _create_mask_from_indices, _evaluate_model_accuracy, _ensure_model_on_device
from alignment.utils.evaluation import evaluate_networks, evaluate_on_loader

logger = logging.getLogger(__name__)


def run_progressive_dropout_experiment(
    networks: List[nn.Module],
    dataset,
    dropout_fractions: List[float],
    metric,
    device="cuda",
    pruning_mode: str = "global_joint",
    dropout_mode: str = "scaled",
    show_progress: bool = True,
    debug_mode: bool = False
) -> Dict:
    """
    Run progressive dropout experiment on multiple networks with multiple strategies.
    
    Args:
        networks: List of networks to evaluate
        dataset: Dataset object
        dropout_fractions: List of dropout fractions to test
        metric: Alignment metric to use
        device: Device to run on
        pruning_mode: Pruning mode to use
        dropout_mode: Dropout mode to use
        show_progress: Whether to show progress bars
        debug_mode: Whether to print additional debug information
        
    Returns:
        Dictionary with dropout experiment results
    """
    # Normalize device
    device = _normalize_device(device)
    
    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies": {"high_rq": [], "low_rq": [], "random": []},
        "losses": {"high_rq": [], "low_rq": [], "random": []},
        "stds": {"high_rq": [], "low_rq": [], "random": []}
    }
    
    try:
        strategies = ["high_rq", "low_rq", "random"]
        strategy_pbar = tqdm(strategies, desc="Pruning strategies", position=0) if show_progress else strategies
        
        for strategy in strategy_pbar:
            if show_progress:
                strategy_pbar.set_description(f"Strategy: {strategy}")
            
            strategy_networks = [copy.deepcopy(net) for net in networks]
            
            if debug_mode:
                logger.info(f"Running progressive_dropout with strategy={strategy}, pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")
            
            network_accuracies, network_losses = progressive_dropout(
                strategy_networks,
                dataset,
                dropout_fractions,
                metric,
                device,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                strategy=strategy,
                show_progress=show_progress,
                debug_mode=debug_mode
            )
            
            fraction_accs = [[] for _ in range(len(dropout_fractions))]
            fraction_losses = [[] for _ in range(len(dropout_fractions))]
            
            for net_idx in network_accuracies:
                for frac_idx, acc in enumerate(network_accuracies[net_idx]):
                    if frac_idx < len(fraction_accs):
                        fraction_accs[frac_idx].append(acc)
                for frac_idx, loss in enumerate(network_losses[net_idx]):
                    if frac_idx < len(fraction_losses):
                        fraction_losses[frac_idx].append(loss)
            
            for frac_idx in range(len(dropout_fractions)):
                if fraction_accs[frac_idx]:
                    mean_acc = np.mean(fraction_accs[frac_idx])
                    std_acc = np.std(fraction_accs[frac_idx])
                    mean_loss = np.mean(fraction_losses[frac_idx]) if fraction_losses[frac_idx] else 0.0
                    
                    results["accuracies"][strategy].append(mean_acc)
                    results["stds"][strategy].append(std_acc)
                    results["losses"][strategy].append(mean_loss)
                    
                    if debug_mode:
                        logger.info(f"  {strategy}, fraction={dropout_fractions[frac_idx]:.2f}: "
                                   f"acc={mean_acc:.2f}±{std_acc:.2f}%, loss={mean_loss:.2f}")
            
            if show_progress:
                last_acc = results["accuracies"][strategy][-1] if results["accuracies"][strategy] else 0
                strategy_pbar.set_postfix({"final_acc": f"{last_acc:.2f}%"})
        
    except Exception as e:
        logger.error(f"Error running progressive dropout: {str(e)}")
        import traceback
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
            import traceback
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
    exclude_classification_layer_config: bool = True
) -> Dict:
    logger.info("Starting Layer Isolated Dropout Experiment")
    device = _normalize_device(device)

    num_original_networks = len(original_networks)
    if num_original_networks == 0:
        logger.warning("No networks provided for layer isolated experiment.")
        return {}

    # Results structure: {strategy: {layer_idx: [avg_accuracies_for_fractions_over_replicates]}}
    # The main keys in the top-level dict will be strategies.
    # We also need to store std deviations.
    results = {
        "dropout_fractions": dropout_fractions,
        "accuracies_isolated": {},
        "stds_isolated": {}
    }

    # Pre-compute scores and sorted indices for EACH original network replicate ONCE
    all_network_metrics_precomputed = []
    for net_idx, net_rep in enumerate(original_networks):
        _ensure_model_on_device(net_rep, device)
        net_rep.eval()
        scores_this_rep = _compute_metric_for_all_nodes(net_rep, metric, device, dataset.test_loader, debug_mode=debug_mode, num_batches=5)
        
        asc_indices_this_rep = {}
        desc_indices_this_rep = {}
        rand_indices_this_rep = {}
        for l_idx_scores, score_tensor in scores_this_rep.items():
            count = score_tensor.shape[0]
            asc_indices_this_rep[l_idx_scores] = torch.argsort(score_tensor, descending=False)
            desc_indices_this_rep[l_idx_scores] = torch.argsort(score_tensor, descending=True)
            allidx_layer = list(range(count))
            random.shuffle(allidx_layer)
            rand_indices_this_rep[l_idx_scores] = torch.tensor(allidx_layer, device=device, dtype=torch.long)

        all_network_metrics_precomputed.append({
            "scores": scores_this_rep, 
            "asc_indices": asc_indices_this_rep, 
            "desc_indices": desc_indices_this_rep, 
            "rand_indices": rand_indices_this_rep
        })

    if not all_network_metrics_precomputed or not all_network_metrics_precomputed[0]["scores"]:
        logger.error("Failed to pre-compute metrics for any network.")
        return {}
    num_layers = len(all_network_metrics_precomputed[0]["scores"].keys())
    classification_layer_actual_idx = num_layers - 1

    strategies_to_run = ["high_rq", "low_rq", "random"]
    for strategy in strategies_to_run:
        results["accuracies_isolated"][strategy] = {l_idx: [] for l_idx in range(num_layers)}
        results["stds_isolated"][strategy] = {l_idx: [] for l_idx in range(num_layers)}

        for layer_to_isolate_idx in tqdm(range(num_layers), desc=f"Isolating Layers (Strat: {strategy})"):
            if exclude_classification_layer_config and layer_to_isolate_idx == classification_layer_actual_idx:
                baseline_accuracies_replicates = [_evaluate_model_accuracy(net, dataset.test_loader, device) for net in original_networks]
                avg_baseline_acc = np.mean(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan
                std_baseline_acc = np.std(baseline_accuracies_replicates) if baseline_accuracies_replicates else np.nan
                results["accuracies_isolated"][strategy][layer_to_isolate_idx] = [avg_baseline_acc] * len(dropout_fractions)
                results["stds_isolated"][strategy][layer_to_isolate_idx] = [std_baseline_acc] * len(dropout_fractions)
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
                    target_layer_module = net_copy.alignment_layers[layer_to_isolate_idx]
                    out_dim = target_layer_module.weight.data.shape[0]
                    n_drop = int(round(frac_val * out_dim))

                    if n_drop > 0 and layer_to_isolate_idx in metrics_for_this_rep["scores"]:
                        indices_map = {
                            "high_rq": metrics_for_this_rep["desc_indices"],
                            "low_rq": metrics_for_this_rep["asc_indices"],
                            "random": metrics_for_this_rep["rand_indices"]
                        }
                        sorted_indices_for_layer = indices_map[strategy].get(layer_to_isolate_idx)
                        
                        if sorted_indices_for_layer is not None:
                            indices_to_drop = sorted_indices_for_layer[:n_drop]
                            wdat = target_layer_module.weight.data
                            mask = _create_mask_from_indices(wdat.shape, indices_to_drop, device)
                            if dropout_mode == "scaled":
                                frac_d_layer = n_drop / float(out_dim) if out_dim > 0 else 0.0
                                scale = 1.0 / (1.0 - frac_d_layer) if frac_d_layer < 0.9999 else 10.0
                                target_layer_module.weight.data.mul_(mask).mul_(scale)
                                if target_layer_module.bias is not None:
                                    bias_mask = _create_mask_from_indices(target_layer_module.bias.data.shape, indices_to_drop, device)
                                    target_layer_module.bias.data.mul_(bias_mask).mul_(scale)
                            else:
                                target_layer_module.weight.data.mul_(mask)
                                if target_layer_module.bias is not None:
                                    bias_mask = _create_mask_from_indices(target_layer_module.bias.data.shape, indices_to_drop, device)
                                    target_layer_module.bias.data.mul_(bias_mask)
                        else:
                            if debug_mode: logger.warning(f"Layer {layer_to_isolate_idx} not found in sorted indices for strategy {strategy}, rep {net_rep_idx}")
                    
                    current_frac_accuracies_over_replicates.append(_evaluate_model_accuracy(net_copy, dataset.test_loader, device))
                
                accuracies_this_layer_all_fracs_avg_reps.append(np.mean(current_frac_accuracies_over_replicates) if current_frac_accuracies_over_replicates else np.nan)
                stds_this_layer_all_fracs_avg_reps.append(np.std(current_frac_accuracies_over_replicates) if current_frac_accuracies_over_replicates else np.nan)
            
            results["accuracies_isolated"][strategy][layer_to_isolate_idx] = accuracies_this_layer_all_fracs_avg_reps
            results["stds_isolated"][strategy][layer_to_isolate_idx] = stds_this_layer_all_fracs_avg_reps

    return results