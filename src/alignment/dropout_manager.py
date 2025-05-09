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

from alignment.dropout import progressive_dropout, eigenvector_dropout, _normalize_device
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