"""
Alignment metrics for neural network analysis.

This module provides metrics for measuring alignment between weight matrices 
and input activations, with various metrics to quantify the degree of alignment.
It supports node-wise scoring for pruning experiments and various utility
functions for alignment-based analysis.
"""

import torch
import math
import numpy as np
import sys
import os
import importlib.util
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import logging
from typing import Dict, Any, Callable, Optional, Tuple, Union, List, Protocol
from torch.utils.data import DataLoader
from alignment.utils.activation_utils import collect_layer_data # Ensure this is imported

logger = logging.getLogger(__name__)

# --- BROJA_2PID Import (Revised) ---
BROJA_2PID_MODULE = None
PID_AVAILABLE = False
try:
    # Attempt to import directly assuming it's vendored or installed
    from alignment.external import BROJA_2PID 
    BROJA_2PID_MODULE = BROJA_2PID
    PID_AVAILABLE = True
    logger.info("Successfully loaded BROJA_2PID module from alignment.external.")
except ImportError:
    logger.warning(
        "BROJA_2PID module not found in alignment.external. "
        "Please ensure BROJA_2PID.py (or package) is placed in src/alignment/external/. "
        "PID-based metrics will use dummy implementations."
    )
    # Define dummy implementation if import fails
    class DummyBROJA:
        @staticmethod
        def pid(p_rc_s: Dict) -> Dict[str, float]:
            logger.warning("Using dummy BROJA_2PID.pid implementation.")
            return {"SI": 0.0, "UIY": 0.0, "UIZ": 0.0, "CI": 0.0}
        BROJA_2PID_MODULE = DummyBROJA()
    PID_AVAILABLE = False
except Exception as e:
     logger.error(f"An unexpected error occurred while trying to load BROJA_2PID: {e}")
     # Define dummy implementation on other errors too
     class DummyBROJA:
         @staticmethod
         def pid(p_rc_s: Dict) -> Dict[str, float]:
             logger.warning("Using dummy BROJA_2PID.pid implementation due to loading error.")
             return {"SI": 0.0, "UIY": 0.0, "UIZ": 0.0, "CI": 0.0}
     BROJA_2PID_MODULE = DummyBROJA()
     PID_AVAILABLE = False

def get_broja_pid_module():
    global BROJA_2PID_MODULE
    # The module (real or dummy) is already assigned during initial import attempt.
    if BROJA_2PID_MODULE is None:
         # This state should ideally not be reached due to the try/except block above.
        logger.error("BROJA_2PID_MODULE is unexpectedly None. Returning a fallback dummy.")
        class FallbackDummyBROJA:
            @staticmethod
            def pid(p_rc_s: Dict) -> Dict[str, float]: return {"SI": 0.0, "UIY": 0.0, "UIZ": 0.0, "CI": 0.0}
        BROJA_2PID_MODULE = FallbackDummyBROJA()
    return BROJA_2PID_MODULE

# --- UTILITY FUNCTIONS ---
def covariance(X: torch.Tensor, force_cpu: bool = False) -> torch.Tensor:
    original_device = X.device
    if force_cpu and X.is_cuda:
        X = X.cpu()
        # logger.debug("Moved tensor to CPU for covariance calculation.")

    if X.ndim == 1: X = X.unsqueeze(0)
    if X.shape[0] < 2:
        # Return on the original device
        return torch.zeros((X.shape[1], X.shape[1]), device=original_device, dtype=X.dtype)
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov_matrix = torch.matmul(X_centered.T, X_centered) / (X.size(0) - 1)
    return cov_matrix.to(original_device) # Ensure result is on original device

def correlation(X: torch.Tensor, force_cpu: bool = False) -> torch.Tensor:
    original_device = X.device
    cov = covariance(X, force_cpu=force_cpu) # Pass flag down
    std_dev = torch.sqrt(torch.diag(cov) + 1e-10)
    outer_std_dev = torch.outer(std_dev, std_dev)
    corr_matrix = torch.where(outer_std_dev > 1e-10, cov / outer_std_dev, torch.zeros_like(cov))
    return corr_matrix.to(original_device)

# --- NEW METRIC: Node Redundancy (Input Feature Correlation) ---
@torch.no_grad()
def compute_node_redundancy(
    layer_inputs: torch.Tensor,
    verbose: bool = False,
    force_cpu_for_large_metric_ops: bool = False,
    **kwargs # Catch-all for unused specific kwargs
) -> torch.Tensor:
    """
    Compute redundancy between input features based on correlation of activations.
    This represents feature redundancy rather than per-output-node scores.
    Adapted from NodeRedundancyMetric in original metrics_utils.py.
    
    Args:
        layer_inputs: Input activations tensor [batch, features].
        verbose: Whether to show verbose logging.
        force_cpu_for_large_metric_ops: Whether to offload large computations to CPU.
        
    Returns:
        Tensor containing average absolute correlations per input feature [features].
    """
    zeros_output = lambda n: torch.zeros(n, device=layer_inputs.device, dtype=layer_inputs.dtype)
    
    if layer_inputs is None:
        logger.warning("compute_node_redundancy: layer_inputs is None. Returning empty tensor.")
        return zeros_output(1)
    
    if layer_inputs.ndim != 2 or layer_inputs.shape[0] < 2:
        logger.warning(f"compute_node_redundancy: inputs not 2D or <2 samples ({layer_inputs.shape}). Ret zeros.")
        num_feats = layer_inputs.shape[1] if layer_inputs.ndim == 2 else 1
        return zeros_output(num_feats)
    
    num_features = layer_inputs.shape[1]
    redundancy_scores = torch.zeros(num_features, device=layer_inputs.device)
    
    try:
        # Compute correlation matrix
        corr_matrix = correlation(layer_inputs, force_cpu=force_cpu_for_large_metric_ops)
        
        # Take absolute values of correlations (we care about strength, not direction)
        abs_corr = torch.abs(corr_matrix)
        
        # For each feature, compute average correlation with other features
        for i in range(num_features):
            # Exclude self-correlation (always 1.0)
            other_indices = [j for j in range(num_features) if j != i]
            if other_indices:  # Only if there are other features
                redundancy_scores[i] = torch.mean(abs_corr[i, other_indices])
    except Exception as e:
        logger.error(f"Error computing node redundancy: {e}", exc_info=verbose)
    
    return torch.nan_to_num(redundancy_scores)

# --- RAYLEIGH QUOTIENT ---
@torch.no_grad()
def compute_rayleigh_quotient(
    layer_inputs: torch.Tensor,
    layer_weights: torch.Tensor,
    relative: bool = True, verbose: bool = False, min_samples_for_cov: int = 2,
    force_cpu_for_large_metric_ops: bool = False # NEW kwarg
) -> torch.Tensor:
    num_out_features = layer_weights.shape[0]
    zeros_output = lambda: torch.zeros(num_out_features, device=layer_weights.device, dtype=layer_weights.dtype)

    if layer_inputs.ndim != 2:
        if layer_inputs.ndim > 2 and layer_inputs.shape[0] > 0:
            layer_inputs = layer_inputs.flatten(start_dim=1)
        else:
            logger.error(f"RQ: inputs not 2D ({layer_inputs.shape}). Ret zeros.")
            return zeros_output()
    if layer_weights.ndim != 2:
        if layer_weights.ndim > 2 and layer_weights.shape[0] > 0:
            layer_weights = layer_weights.reshape(layer_weights.shape[0], -1)
        else:
            logger.error(f"RQ: weights not 2D ({layer_weights.shape}). Ret zeros.")
            return zeros_output()
    if layer_inputs.shape[0] < min_samples_for_cov: 
        logger.warning(f"RQ: samples < {min_samples_for_cov}. Ret zeros.")
        return zeros_output()
    if layer_weights.shape[1] != layer_inputs.shape[1]:
        min_dim = min(layer_weights.shape[1], layer_inputs.shape[1])
        layer_weights = layer_weights[:, :min_dim]; layer_inputs = layer_inputs[:, :min_dim]
        if min_dim == 0: 
            logger.error("RQ: 0 common feature dim. Ret zeros.")
            return zeros_output()
    if layer_inputs.numel() == 0 or layer_weights.numel() == 0: 
        logger.warning("RQ: empty inputs/weights. Ret zeros.")
        return zeros_output()
    try:
        # Pass force_cpu_for_large_metric_ops to covariance
        C = covariance(layer_inputs.to(layer_weights.device), force_cpu=force_cpu_for_large_metric_ops)
        if torch.isnan(C).any() or torch.isinf(C).any(): 
            logger.warning("RQ: NaN/Inf in cov. Ret zeros.")
            return zeros_output()
        
        # Efficient computation for w^T C w for each w in W
        # W is [out_features, in_features], C is [in_features, in_features]
        # WC = W @ C  -> [out_features, in_features]
        # Element-wise multiply WC with W: WCW = WC * W -> [out_features, in_features]
        # Sum over in_features: numerator = WCW.sum(dim=1) -> [out_features]
        WC = torch.matmul(layer_weights, C)
        numerator = torch.sum(WC * layer_weights, dim=1)
        denominator = torch.sum(layer_weights * layer_weights, dim=1)
        
        rq_scores = torch.zeros_like(numerator, device=layer_weights.device) # Ensure scores are on original device
        # Move numerator to weights device for division and assignment
        numerator = numerator.to(rq_scores.device)

        mask = denominator > 1e-12
        rq_scores[mask] = numerator[mask] / denominator[mask]
        if relative:
            trace_C = torch.trace(C)
            if trace_C > 1e-12: rq_scores[mask] /= trace_C.to(rq_scores.device) # Ensure trace is on correct device
            else: rq_scores[mask] = 0.0
        return torch.nan_to_num(rq_scores, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logger.error(f"RQ calc error: {e}", exc_info=verbose); return zeros_output()

# --- MUTUAL INFORMATION METRICS ---
@torch.no_grad()
def mi_gaussian_approx(
    layer_outputs: torch.Tensor, target_outputs: Optional[torch.Tensor] = None, 
    order: int = 0, verbose: bool = False,
    force_cpu_for_large_metric_ops: bool = False # NEW kwarg
) -> torch.Tensor:
    zeros_output = lambda num_n: torch.zeros(num_n, device=layer_outputs.device, dtype=layer_outputs.dtype)
    if layer_outputs.ndim != 2 or layer_outputs.shape[0] < 2: 
        logger.warning(f"MI_gauss: outputs not 2D or <2 samples ({layer_outputs.shape if layer_outputs is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim == 2 else 1)
    
    batch_size, num_neurons = layer_outputs.shape
    mi_scores = torch.zeros(num_neurons, device=layer_outputs.device)

    ref_data = target_outputs
    if ref_data is None:
        if num_neurons <= 1: return zeros_output(num_neurons)
        # Use PC1 of layer_outputs as reference if no target_outputs
        try:
            cov_layer = covariance(layer_outputs, force_cpu=force_cpu_for_large_metric_ops)
            _, eigenvectors = torch.linalg.eigh(cov_layer.to(layer_outputs.device))
            ref_data = torch.matmul(layer_outputs, eigenvectors[:, -1].unsqueeze(1))
        except Exception as e:
            logger.warning(f"MI_gauss: PC1 calc failed: {e}. Ret zeros.")
            return zeros_output(num_neurons)
    
    if ref_data.ndim == 1: ref_data = ref_data.unsqueeze(1)
    if ref_data.shape[0] != batch_size: 
        logger.warning(f"MI_gauss: ref_data batch mismatch ({ref_data.shape[0]} vs {batch_size}). Ret zeros.")
        return zeros_output(num_neurons)

    for i in range(num_neurons):
        neuron_out_i = layer_outputs[:, i].unsqueeze(1)
        avg_rho_sq_sum = 0.0
        valid_refs = 0
        for k in range(ref_data.shape[1]):
            ref_k = ref_data[:, k].unsqueeze(1)
            combined = torch.cat((neuron_out_i, ref_k), dim=1)
            if combined.shape[0] < 2: continue
            cov_matrix = covariance(combined, force_cpu=force_cpu_for_large_metric_ops)
            var_neuron, var_ref = cov_matrix[0,0], cov_matrix[1,1]
            if var_neuron > 1e-12 and var_ref > 1e-12:
                rho_sq = (cov_matrix[0,1]**2) / (var_neuron * var_ref)
                avg_rho_sq_sum += torch.clamp(rho_sq, 0, 0.999999) # Clamp for log
                valid_refs +=1 
        if valid_refs > 0:
            mi_scores[i] = -0.5 * torch.log(1.0 - (avg_rho_sq_sum / valid_refs))
    return torch.nan_to_num(mi_scores)

@torch.no_grad()
def mi_direct_binning(
    layer_outputs: torch.Tensor, target_outputs: Optional[torch.Tensor] = None, 
    bins: int = 10, verbose: bool = False
) -> torch.Tensor:
    zeros_output = lambda num_n: torch.zeros(num_n, device=layer_outputs.device, dtype=layer_outputs.dtype)
    if layer_outputs.ndim != 2 or layer_outputs.shape[0] < 2: 
        logger.warning(f"MI_direct: outputs not 2D or <2 samples ({layer_outputs.shape if layer_outputs is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim == 2 else 1)

    batch_size, num_neurons = layer_outputs.shape
    mi_scores = torch.zeros(num_neurons, device=layer_outputs.device)
    source_np = layer_outputs.cpu().numpy()

    ref_np: Optional[np.ndarray] = None
    if target_outputs is not None:
        if target_outputs.ndim == 1: target_outputs = target_outputs.unsqueeze(1)
        if target_outputs.shape[0] != batch_size: 
            logger.warning(f"MI_direct: target batch mismatch ({target_outputs.shape[0]} vs {batch_size}). Ret zeros.")
            return zeros_output(num_neurons)
        ref_np = target_outputs.cpu().numpy()
    elif num_neurons > 1:
        # If no target, use mean of other neurons as reference (changes per neuron i)
        pass 
    else: # num_neurons <=1 and no target_outputs
        return zeros_output(num_neurons)

    for i in range(num_neurons):
        neuron_i_np = source_np[:, i]
        current_ref_np = ref_np
        if current_ref_np is None and num_neurons > 1: # Calculate ref based on others
            other_indices = [j for j in range(num_neurons) if j != i]
            if not other_indices: continue
            current_ref_np = np.mean(source_np[:, other_indices], axis=1, keepdims=True)
        if current_ref_np is None: continue

        min_i, max_i = np.min(neuron_i_np), np.max(neuron_i_np)
        bins_i_vals = np.linspace(min_i, max_i, bins + 1)
        digitized_i = np.digitize(neuron_i_np, bins_i_vals[:-1] if max_i > min_i + 1e-9 else [min_i]) -1
        digitized_i = np.clip(digitized_i, 0, bins - 1)
        
        avg_mi_for_neuron_i = 0.0
        valid_refs = 0
        for k in range(current_ref_np.shape[1]):
            ref_k_np = current_ref_np[:, k]
            min_k, max_k = np.min(ref_k_np), np.max(ref_k_np)
            bins_k_vals = np.linspace(min_k, max_k, bins + 1)
            digitized_k = np.digitize(ref_k_np, bins_k_vals[:-1] if max_k > min_k + 1e-9 else [min_k]) -1
            digitized_k = np.clip(digitized_k, 0, bins-1)

            joint_hist = np.zeros((bins, bins), dtype=float)
            for s_idx in range(batch_size): joint_hist[digitized_i[s_idx], digitized_k[s_idx]] += 1
            joint_p = joint_hist / batch_size
            p_i = np.sum(joint_p, axis=1); p_k = np.sum(joint_p, axis=0)
            
            mi_val = 0.0
            for b_i in range(bins): 
                for b_k in range(bins):
                    if joint_p[b_i,b_k]>1e-12 and p_i[b_i]>1e-12 and p_k[b_k]>1e-12:
                        mi_val += joint_p[b_i,b_k] * np.log2(joint_p[b_i,b_k]/(p_i[b_i]*p_k[b_k]))
            avg_mi_for_neuron_i += mi_val
            valid_refs +=1
        if valid_refs > 0: mi_scores[i] = avg_mi_for_neuron_i / valid_refs
        
    return torch.nan_to_num(mi_scores.to(layer_outputs.device))

# --- REDUNDANCY AND PID METRICS ---
@torch.no_grad()
def average_redundancy_gaussian(
    layer_inputs: torch.Tensor, layer_weights: torch.Tensor, verbose: bool = False,
    force_cpu_for_large_metric_ops: bool = False # NEW kwarg
) -> torch.Tensor:
    zeros_output = lambda num_n: torch.zeros(num_n, device=layer_weights.device, dtype=layer_weights.dtype)
    if layer_inputs.ndim!=2 or layer_inputs.shape[0]<2: 
        logger.warning(f"Red_gauss: inputs problem (shape: {layer_inputs.shape if layer_inputs is not None else 'None'}, ndim: {layer_inputs.ndim if layer_inputs is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_weights.shape[0] if layer_weights is not None else 1)
    if layer_weights.ndim!=2 or layer_weights.shape[0]<=1: 
        logger.warning(f"Red_gauss: weights problem (shape: {layer_weights.shape if layer_weights is not None else 'None'}, ndim: {layer_weights.ndim if layer_weights is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_weights.shape[0] if layer_weights is not None else 1)
    
    num_neurons = layer_weights.shape[0]
    avg_red_scores = torch.zeros(num_neurons, device=layer_weights.device)
    projected_outputs = torch.matmul(layer_inputs, layer_weights.T)
    if projected_outputs.shape[0] < 2: return avg_red_scores

    try: corr_matrix_projections = correlation(projected_outputs, force_cpu=force_cpu_for_large_metric_ops)
    except Exception as e: 
        logger.warning(f"Red_gauss: corr matrix failed: {e}. Ret zeros.")
        return avg_red_scores

    for i in range(num_neurons):
        sum_red_i, num_pairs = 0.0, 0
        for j in range(num_neurons):
            if i == j: continue
            rho_sq = torch.clamp(corr_matrix_projections[i,j]**2, 0, 0.999999)
            sum_red_i += -0.5 * torch.log(1.0 - rho_sq)
            num_pairs += 1
        if num_pairs > 0: avg_red_scores[i] = sum_red_i / num_pairs
    return torch.nan_to_num(avg_red_scores)

@torch.no_grad()
def average_pid_component(
    layer_inputs: torch.Tensor, layer_outputs: torch.Tensor, 
    pid_component_name: str, bins: int = 10, verbose: bool = False,
    force_cpu_for_large_metric_ops: bool = False # NEW kwarg
) -> torch.Tensor:
    zeros_output = lambda num_n: torch.zeros(num_n, device=layer_outputs.device, dtype=layer_outputs.dtype)
    
    # Use the PID_AVAILABLE flag for a quick check
    if not PID_AVAILABLE:
        logger.warning(f"PID_{pid_component_name}: BROJA not available. Ret zeros.")
        return zeros_output(layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim == 2 else 1)
    
    broja = get_broja_pid_module() # Get the module (could still be dummy if loaded but unusable?)

    if layer_inputs.ndim!=2 or layer_inputs.shape[0]<2: 
        logger.warning(f"PID: inputs problem (shape: {layer_inputs.shape if layer_inputs is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim == 2 else 1)
    if layer_outputs.ndim!=2 or layer_outputs.shape[0]<2 or layer_outputs.shape[1]<=1: 
        logger.warning(f"PID: outputs problem (shape: {layer_outputs.shape if layer_outputs is not None else 'None'}). Ret zeros.")
        return zeros_output(layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim == 2 else 1)

    batch_size, features_in = layer_inputs.shape
    _, num_neurons = layer_outputs.shape
    avg_pid_scores = torch.zeros(num_neurons, device=layer_outputs.device)
    layer_inputs_np, layer_outputs_np = layer_inputs.cpu().numpy(), layer_outputs.cpu().numpy()

    s_target_np = layer_inputs_np[:,0] # Default to first feature if PCA fails or not used
    if features_in > 1:
        try:
            n_pca_comp = min(batch_size-1, features_in, 5) # Keep PCA components low for stability
            if n_pca_comp >= 1:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1, whiten=False, svd_solver='auto')
                s_target_np = pca.fit_transform(layer_inputs_np).flatten()
            else: logger.warning(f"PID: Not enough samples/features for PCA on inputs ({features_in} features, {batch_size} samples). Using first feature.")
        except Exception as e_pca:
            logger.warning(f"PID: PCA on inputs failed: {e_pca}. Using first feature.")
            pass # s_target_np already defaults to first feature
    
    min_s, max_s = np.min(s_target_np), np.max(s_target_np)
    s_bins_vals = np.linspace(min_s, max_s, bins + 1)
    s_digitized = np.digitize(s_target_np, s_bins_vals[:-1] if max_s > min_s+1e-9 else [min_s]) -1
    s_digitized = np.clip(s_digitized, 0, bins - 1)

    for i in range(num_neurons):
        y1_np = layer_outputs_np[:, i]
        min_y1, max_y1 = np.min(y1_np), np.max(y1_np)
        y1_bins_vals = np.linspace(min_y1, max_y1, bins + 1)
        y1_digitized = np.digitize(y1_np, y1_bins_vals[:-1] if max_y1 > min_y1+1e-9 else [min_y1]) -1
        y1_digitized = np.clip(y1_digitized, 0, bins - 1)

        sum_pid_i, num_pairs = 0.0, 0
        for j in range(num_neurons):
            if i == j: continue
            y2_np = layer_outputs_np[:, j]
            min_y2, max_y2 = np.min(y2_np), np.max(y2_np)
            y2_bins_vals = np.linspace(min_y2, max_y2, bins + 1)
            y2_digitized = np.digitize(y2_np, y2_bins_vals[:-1] if max_y2 > min_y2+1e-9 else [min_y2]) -1
            y2_digitized = np.clip(y2_digitized, 0, bins-1)
            
            p_sy1y2_dict: Dict[Tuple[int,int,int], float] = {}
            for s_idx in range(batch_size):
                key = (s_digitized[s_idx], y1_digitized[s_idx], y2_digitized[s_idx])
                p_sy1y2_dict[key] = p_sy1y2_dict.get(key, 0.0) + 1.0
            if not p_sy1y2_dict: continue
            total_counts = sum(p_sy1y2_dict.values())
            if total_counts == 0 : continue
            broja_input = {k: v / total_counts for k,v in p_sy1y2_dict.items()}
            if not broja_input: continue

            try:
                pid_res = broja.pid(broja_input)
                comp_val_map = {"SI": "SI", "UI1": "UIY", "UI2": "UIZ", "CI": "CI"}
                sum_pid_i += pid_res.get(comp_val_map.get(pid_component_name, ""), 0.0)
                num_pairs +=1
            except Exception as e_pid:
                if verbose: logger.warning(f"PID: BROJA fail (neuron {i}, pair {j}, comp {pid_component_name}): {e_pid}. Skip.")
                continue
        if num_pairs > 0: avg_pid_scores[i] = sum_pid_i / num_pairs
    return torch.nan_to_num(avg_pid_scores.to(layer_outputs.device))

# --- METRIC PROTOCOL & WRAPPER --- 
class AlignmentMetric(Protocol):
    name: str
    scale_by_norm: bool

    def compute_per_node_scores(
        self,
        layer_inputs: Optional[torch.Tensor] = None, 
        layer_weights: Optional[torch.Tensor] = None, 
        layer_outputs: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        min_samples_for_cov: int = 2, 
        target_outputs: Optional[torch.Tensor] = None,
        bins: int = 10, 
        verbose: bool = False,
        **metric_specific_kwargs
    ) -> torch.Tensor:
        ...

class _AlignmentMetricImpl:
    def __init__(self, name: str, metric_fn: Callable, scale_by_norm: bool = False):
        self.name = name
        self._metric_fn = metric_fn
        self.scale_by_norm = scale_by_norm

    def compute_per_node_scores(
        self,
        layer_inputs: Optional[torch.Tensor] = None, 
        layer_weights: Optional[torch.Tensor] = None, 
        layer_outputs: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
        min_samples_for_cov: int = 2,
        target_outputs: Optional[torch.Tensor] = None,
        bins: int = 10,
        verbose: bool = False,
        **metric_specific_kwargs
    ) -> torch.Tensor:
        
        metric_name_lower = self.name.lower()
        if device is None:
            if layer_inputs is not None: device = layer_inputs.device
            elif layer_weights is not None: device = layer_weights.device
            elif layer_outputs is not None: device = layer_outputs.device
            else: raise ValueError("Cannot determine device for metric calculation.")
        eff_device = torch.device(device) if isinstance(device, str) else device
        
        # Log input tensors for debugging
        logger.debug(f"compute_per_node_scores({self.name}): Tensor availability - " +  # Ensure this is debug
                   f"inputs: {layer_inputs is not None}, weights: {layer_weights is not None}, " +
                   f"outputs: {layer_outputs is not None}")
        
        if layer_inputs is not None:
            logger.debug(f"compute_per_node_scores({self.name}): layer_inputs.shape = {layer_inputs.shape}, " +  # Ensure this is debug
                       f"dtype = {layer_inputs.dtype}, device = {layer_inputs.device}")
        if layer_weights is not None:
            logger.debug(f"compute_per_node_scores({self.name}): layer_weights.shape = {layer_weights.shape}, " +  # Ensure this is debug
                       f"dtype = {layer_weights.dtype}, device = {layer_weights.device}")
        if layer_outputs is not None:
            logger.debug(f"compute_per_node_scores({self.name}): layer_outputs.shape = {layer_outputs.shape}, " +  # Ensure this is debug
                       f"dtype = {layer_outputs.dtype}, device = {layer_outputs.device}")
        
        # Extract relevant kwargs for specific metric functions, pass others via **fn_kwargs
        fn_kwargs = {"verbose": verbose}
        force_cpu_flag = metric_specific_kwargs.get("force_cpu_for_large_metric_ops", False)
        # configured_cnn_mode = metric_specific_kwargs.get("configured_cnn_mode", "unfold") # Example if needed
        # configured_cnn_rq_op = metric_specific_kwargs.get("configured_cnn_rq_op", "mean") # Example if needed

        try:
            if "rayleigh_quotient" in metric_name_lower or "rq" in metric_name_lower:
                if layer_inputs is None or layer_weights is None: 
                    logger.error(f"{self.name} needs layer_inputs and layer_weights, but got inputs: {layer_inputs is not None}, weights: {layer_weights is not None}")
                    # Return an empty tensor with correct device/dtype instead of raising error
                    if layer_weights is not None:
                        return torch.zeros(layer_weights.shape[0], device=eff_device, dtype=layer_weights.dtype)
                    else:
                        return torch.zeros(1, device=eff_device, dtype=torch.float32)
                
                # Check tensor shapes and dimensions
                if layer_inputs.ndim != 2:
                    layer_inputs_reshaped = layer_inputs
                    if layer_inputs.ndim > 2 and layer_inputs.shape[0] > 0:
                        logger.debug(f"compute_per_node_scores({self.name}): Reshaping layer_inputs from {layer_inputs.shape} to [batch_size, features]") # Ensure this is debug
                        layer_inputs_reshaped = layer_inputs.flatten(start_dim=1)
                    else:
                        logger.error(f"compute_per_node_scores({self.name}): Cannot reshape layer_inputs with ndim {layer_inputs.ndim}")
                        return torch.zeros(layer_weights.shape[0], device=eff_device, dtype=layer_weights.dtype)
                else:
                    layer_inputs_reshaped = layer_inputs
                    
                if layer_weights.ndim != 2:
                    layer_weights_reshaped = layer_weights
                    if layer_weights.ndim > 2 and layer_weights.shape[0] > 0:
                        logger.debug(f"compute_per_node_scores({self.name}): Reshaping layer_weights from {layer_weights.shape} to [out_features, in_features]") # Ensure this is debug
                        layer_weights_reshaped = layer_weights.reshape(layer_weights.shape[0], -1)
                    else:
                        logger.error(f"compute_per_node_scores({self.name}): Cannot reshape layer_weights with ndim {layer_weights.ndim}")
                        return torch.zeros(layer_weights.shape[0], device=eff_device, dtype=layer_weights.dtype)
                else:
                    layer_weights_reshaped = layer_weights
                
                # Check for dimension mismatch and fix
                if layer_inputs_reshaped.shape[1] != layer_weights_reshaped.shape[1]:
                    min_dim = min(layer_weights_reshaped.shape[1], layer_inputs_reshaped.shape[1])
                    logger.warning(f"compute_per_node_scores({self.name}): Dimension mismatch - inputs: {layer_inputs_reshaped.shape[1]}, weights: {layer_weights_reshaped.shape[1]}. Using first {min_dim} dimensions.")
                    if min_dim == 0:
                        logger.error(f"compute_per_node_scores({self.name}): No common feature dimensions.")
                        return torch.zeros(layer_weights_reshaped.shape[0], device=eff_device, dtype=layer_weights_reshaped.dtype)
                    layer_weights_reshaped = layer_weights_reshaped[:, :min_dim]
                    layer_inputs_reshaped = layer_inputs_reshaped[:, :min_dim]
                
                fn_kwargs["relative"] = self.scale_by_norm
                fn_kwargs["min_samples_for_cov"] = min_samples_for_cov
                fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
                
                logger.debug(f"compute_per_node_scores({self.name}): Calling compute_rayleigh_quotient with shapes - inputs: {layer_inputs_reshaped.shape}, weights: {layer_weights_reshaped.shape}") # Ensure this is debug
                result = self._metric_fn(layer_inputs_reshaped.to(eff_device), layer_weights_reshaped.to(eff_device), **fn_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result
            
            elif "mi_gaussian" in metric_name_lower or "mi_g" in metric_name_lower:
                if layer_outputs is None: 
                    logger.error(f"{self.name} needs layer_outputs, but got outputs: {layer_outputs is not None}")
                    return torch.zeros(1, device=eff_device, dtype=torch.float32)
                
                if target_outputs is not None: fn_kwargs["target_outputs"] = target_outputs.to(eff_device)
                if "order" in metric_specific_kwargs: fn_kwargs["order"] = metric_specific_kwargs["order"]
                fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
                
                logger.debug(f"compute_per_node_scores({self.name}): Calling mi_gaussian_approx with outputs shape: {layer_outputs.shape}") # Ensure this is debug
                result = self._metric_fn(layer_outputs.to(eff_device), **fn_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result

            elif "mi_direct" in metric_name_lower or "mi_bin" in metric_name_lower:
                if layer_outputs is None: 
                    logger.error(f"{self.name} needs layer_outputs, but got outputs: {layer_outputs is not None}")
                    return torch.zeros(1, device=eff_device, dtype=torch.float32)
                
                if target_outputs is not None: fn_kwargs["target_outputs"] = target_outputs.to(eff_device)
                fn_kwargs["bins"] = bins
                # mi_direct_binning doesn't use covariance, so force_cpu_flag isn't directly passed to it.
                logger.debug(f"compute_per_node_scores({self.name}): Calling mi_direct_binning with outputs shape: {layer_outputs.shape}") # Ensure this is debug
                result = self._metric_fn(layer_outputs.to(eff_device), **fn_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result

            elif "redundancy_gaussian" in metric_name_lower or "red_g" in metric_name_lower:
                if layer_inputs is None or layer_weights is None: 
                    logger.error(f"{self.name} needs layer_inputs and layer_weights, but got inputs: {layer_inputs is not None}, weights: {layer_weights is not None}")
                    return torch.zeros(layer_weights.shape[0] if layer_weights is not None else 1, device=eff_device, 
                                     dtype=layer_weights.dtype if layer_weights is not None else torch.float32)
                
                fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
                logger.debug(f"compute_per_node_scores({self.name}): Calling average_redundancy_gaussian with shapes - inputs: {layer_inputs.shape}, weights: {layer_weights.shape}") # Ensure this is debug
                result = self._metric_fn(layer_inputs.to(eff_device), layer_weights.to(eff_device), **fn_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result
            
            # --- ADDED/UPDATED CASE for NodeRedundancy Metric ---
            elif "node_redundancy" in metric_name_lower:
                if layer_inputs is None:
                    logger.error(f"{self.name} needs layer_inputs, but got None.")
                    return torch.zeros(1, device=eff_device, dtype=torch.float32)
                
                fn_kwargs["force_cpu_for_large_metric_ops"] = metric_specific_kwargs.get("force_cpu_for_large_metric_ops", False)
                logger.debug(f"compute_per_node_scores({self.name}): Calling compute_node_redundancy with inputs shape: {layer_inputs.shape}")
                result = self._metric_fn(layer_inputs=layer_inputs.to(eff_device), **fn_kwargs, **metric_specific_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}")
                return result
                
            # --- ADDED/UPDATED CASE for Weight-based Metrics ---
            elif metric_name_lower in ["weight_cosine_similarity", "weight_dot_similarity", "weight_euclidean_distance"]:
                if layer_weights is None:
                    logger.error(f"{self.name} needs layer_weights, but it is None.")
                    return torch.empty(0, 0, device=eff_device, dtype=torch.float32) # Return 0x0 for matrix
                
                logger.debug(f"compute_per_node_scores({self.name}): Calling metric with layer_weights.shape = {layer_weights.shape}") # Ensure this is debug
                result = self._metric_fn(layer_weights=layer_weights.to(eff_device), **fn_kwargs, **metric_specific_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result
            # --- END ADDED/UPDATED CASE ---

            elif "pid_" in metric_name_lower:
                if not PID_AVAILABLE: 
                    logger.warning(f"PID metric {self.name} called; BROJA_2PID not available. Returning zeros.")
                    num_n = layer_outputs.shape[1] if layer_outputs is not None and layer_outputs.ndim > 1 else \
                            (layer_inputs.shape[1] if layer_inputs is not None and layer_inputs.ndim > 1 else 1)
                    return torch.zeros(num_n, device=eff_device)
                
                if layer_inputs is None or layer_outputs is None: 
                    logger.error(f"{self.name} needs layer_inputs and layer_outputs, but got inputs: {layer_inputs is not None}, outputs: {layer_outputs is not None}")
                    # Determine a sensible default size for the zeros tensor
                    default_size = 1
                    if layer_outputs is not None and layer_outputs.ndim > 1:
                        default_size = layer_outputs.shape[1]
                    elif layer_inputs is not None and layer_inputs.ndim > 1:
                         # This case might not be ideal if PID expects output node scores
                        pass # default_size remains 1 or could be layer_inputs.shape[1]
                    return torch.zeros(default_size, device=eff_device, 
                                     dtype=layer_outputs.dtype if layer_outputs is not None else (layer_inputs.dtype if layer_inputs is not None else torch.float32))
                
                fn_kwargs["bins"] = bins
                logger.debug(f"compute_per_node_scores({self.name}): Calling PID component metric with shapes - inputs: {layer_inputs.shape}, outputs: {layer_outputs.shape}") # Ensure this is debug
                result = self._metric_fn(layer_inputs.to(eff_device), layer_outputs.to(eff_device), **fn_kwargs, **metric_specific_kwargs) # Pass metric_specific_kwargs
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result

            elif metric_name_lower == "mi_proj_vs_mean_input":
                if layer_inputs is None or layer_weights is None:
                    logger.error(f"{self.name} needs layer_inputs and layer_weights.")
                    return torch.zeros(layer_weights.shape[0] if layer_weights is not None else 1, device=eff_device, dtype=torch.float32)
                logger.debug(f"compute_per_node_scores({self.name}): Calling with inputs: {layer_inputs.shape}, weights: {layer_weights.shape}") # Ensure this is debug
                result = self._metric_fn(layer_inputs=layer_inputs.to(eff_device), layer_weights=layer_weights.to(eff_device), **fn_kwargs, **metric_specific_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result

            elif metric_name_lower == "rq_alt_denom":
                if layer_inputs is None or layer_weights is None: 
                    logger.error(f"{self.name} needs layer_inputs and layer_weights.")
                    # Return an empty tensor with correct device/dtype instead of raising error
                    return torch.zeros(layer_weights.shape[0] if layer_weights is not None else 1, device=eff_device, dtype=layer_weights.dtype if layer_weights is not None else torch.float32)
                
                # Shape checks similar to the main RQ could be added here if desired for robustness
                # For now, assuming inputs are correctly shaped as per original metrics_utils.py usage.
                logger.debug(f"compute_per_node_scores({self.name}): Calling RQ_alt with inputs: {layer_inputs.shape}, weights: {layer_weights.shape}") # Ensure this is debug
                result = self._metric_fn(layer_inputs=layer_inputs.to(eff_device), layer_weights=layer_weights.to(eff_device), **fn_kwargs, **metric_specific_kwargs)
                logger.debug(f"compute_per_node_scores({self.name}): Result shape: {result.shape}") # Ensure this is debug
                return result

            else:
                logger.error(f"Dispatch for {self.name} in _AlignmentMetricImpl.compute_per_node_scores not implemented.")
                return torch.zeros(1, device=eff_device, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error in compute_per_node_scores for metric {self.name}: {e}", exc_info=True)
            if layer_weights is not None:
                return torch.zeros(layer_weights.shape[0], device=eff_device, dtype=layer_weights.dtype)
            elif layer_outputs is not None and layer_outputs.ndim > 1:
                return torch.zeros(layer_outputs.shape[1], device=eff_device, dtype=layer_outputs.dtype)
            else:
                return torch.zeros(1, device=eff_device, dtype=torch.float32)

# --- METRIC REGISTRY & DISPATCHER ---
# Raw metric functions for PID, to be wrapped by _AlignmentMetricImpl
def pid_si_metric_raw(layer_inputs: torch.Tensor, layer_outputs: torch.Tensor, **kwargs) -> torch.Tensor:
    return average_pid_component(layer_inputs, layer_outputs, "SI", **kwargs)
def pid_uiy_metric_raw(layer_inputs: torch.Tensor, layer_outputs: torch.Tensor, **kwargs) -> torch.Tensor:
    return average_pid_component(layer_inputs, layer_outputs, "UIY", **kwargs)
def pid_uiz_metric_raw(layer_inputs: torch.Tensor, layer_outputs: torch.Tensor, **kwargs) -> torch.Tensor:
    return average_pid_component(layer_inputs, layer_outputs, "UIZ", **kwargs)
def pid_ci_metric_raw(layer_inputs: torch.Tensor, layer_outputs: torch.Tensor, **kwargs) -> torch.Tensor:
    return average_pid_component(layer_inputs, layer_outputs, "CI", **kwargs)

# Initial registry with core metrics
ALIGNMENT_METRICS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "rayleigh_quotient": compute_rayleigh_quotient,
    "rq": compute_rayleigh_quotient,
    "mi_gaussian": mi_gaussian_approx,
    "mi_g": mi_gaussian_approx,
    "mi_direct": mi_direct_binning,
    "mi_bin": mi_direct_binning,
    "redundancy_gaussian": average_redundancy_gaussian,
    "red_g": average_redundancy_gaussian,
    "node_redundancy": compute_node_redundancy,  # Added Node Redundancy metric
    "pid_shared_info": pid_si_metric_raw,
    "pid_si": pid_si_metric_raw,
    "pid_unique_info_neuron": pid_uiy_metric_raw, 
    "pid_ui1": pid_uiy_metric_raw, 
    "pid_uiy": pid_uiy_metric_raw,
    "pid_unique_info_other": pid_uiz_metric_raw, 
    "pid_ui2": pid_uiz_metric_raw,
    "pid_uiz": pid_uiz_metric_raw,
    "pid_synergy_info": pid_ci_metric_raw,
    "pid_ci": pid_ci_metric_raw,
}

# RENAMED from get_metric_function
def get_metric(name: str, scale_by_norm: bool = False) -> AlignmentMetric: # Returns the Protocol type
    """Gets an AlignmentMetric instance (wrapper) from the registry."""
    metric_fn_callable = ALIGNMENT_METRICS_REGISTRY.get(name.lower())
    if metric_fn_callable is None:
        available = ", ".join(ALIGNMENT_METRICS_REGISTRY.keys())
        raise ValueError(f"Metric '{name}' not found in registry. Available: {available}")
    return _AlignmentMetricImpl(name=name, metric_fn=metric_fn_callable, scale_by_norm=scale_by_norm)


@torch.no_grad()
def compute_metrics_for_layers(
    model: nn.Module,
    collected_data: Dict[str, Dict[str, torch.Tensor]],
    metric_configs: List[Dict[str, Any]], 
    device: Union[str, torch.device],
) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    logger.info(f"compute_metrics_for_layers: === ENTERED FUNCTION ===. collected_data keys: {list(collected_data.keys()) if collected_data else 'EMPTY_OR_NONE'}. Num metric_configs: {len(metric_configs) if metric_configs else 0}")
    logger.debug(f"compute_metrics_for_layers: metric_configs = {metric_configs}") # CHANGED to debug
    
    results: Dict[str, Dict[str, Union[torch.Tensor, float]]] = {}
    model.to(device); model.eval()
    
    # Adjust module lookup based on whether model has base_model
    target_model_for_modules = model
    if hasattr(model, 'base_model') and isinstance(model.base_model, nn.Module):
        logger.debug("compute_metrics_for_layers: Using model.base_model to resolve module names.") # CHANGED to debug
        target_model_for_modules = model.base_model
    else:
        logger.debug("compute_metrics_for_layers: Using model directly to resolve module names.") # CHANGED to debug
        
    modules_dict = {name: mod for name, mod in target_model_for_modules.named_modules()}
    logger.debug(f"compute_metrics_for_layers: Available modules for lookup: {list(modules_dict.keys())}") # CHANGED to debug

    for layer_name, layer_data in collected_data.items():
        results[layer_name] = {}
        module = modules_dict.get(layer_name)
        
        logger.debug(f"compute_metrics_for_layers: Processing layer '{layer_name}'. Module found: {module is not None}. Layer_data keys: {list(layer_data.keys()) if layer_data else 'None'}") # CHANGED to debug
        
        if 'input' in layer_data:
            logger.debug(f"compute_metrics_for_layers: Layer '{layer_name}' input shape: {layer_data['input'].shape}") # CHANGED to debug
            # Check for NaN or Inf values in input
            if torch.isnan(layer_data['input']).any() or torch.isinf(layer_data['input']).any():
                logger.warning(f"compute_metrics_for_layers: Layer '{layer_name}' input contains NaN or Inf values!")
        else:
            logger.warning(f"compute_metrics_for_layers: Layer '{layer_name}' has no 'input' data!")
        
        if 'output' in layer_data:
            logger.debug(f"compute_metrics_for_layers: Layer '{layer_name}' output shape: {layer_data['output'].shape}") # CHANGED to debug
        
        if not module: 
            logger.warning(f"compute_metrics_for_layers: Module '{layer_name}' (from collected_data) not found in resolved model_modules for weight/type lookup.")
            continue
        
        if hasattr(module, 'weight') and module.weight is not None:
            logger.debug(f"compute_metrics_for_layers: Layer '{layer_name}' weight shape: {module.weight.shape}") # CHANGED to debug
            # Check for NaN or Inf values in weights
            if torch.isnan(module.weight).any() or torch.isinf(module.weight).any():
                logger.warning(f"compute_metrics_for_layers: Layer '{layer_name}' weights contain NaN or Inf values!")
        else:
            logger.warning(f"compute_metrics_for_layers: Layer '{layer_name}' has no weight attribute or it's None.")
            continue  # Skip this layer if no weights are available
        
        l_inputs = layer_data.get("input")
        l_outputs = layer_data.get("output")
        
        # Verify dimensions match for RQ computation
        weights_shape = module.weight.shape
        if l_inputs is not None and weights_shape[1] != l_inputs.shape[1]:
            logger.warning(f"compute_metrics_for_layers: Dimension mismatch for '{layer_name}': weights_shape[1]={weights_shape[1]}, l_inputs.shape[1]={l_inputs.shape[1]}")
            if weights_shape[1] < l_inputs.shape[1]:
                logger.info(f"compute_metrics_for_layers: Truncating input features for '{layer_name}' from {l_inputs.shape[1]} to {weights_shape[1]}")
                l_inputs = l_inputs[:, :weights_shape[1]]
            else:
                logger.warning(f"compute_metrics_for_layers: Input features for '{layer_name}' are fewer than weight features. This may cause issues.")

        if not metric_configs: 
            logger.warning(f"compute_metrics_for_layers: metric_configs is empty for layer '{layer_name}'. No metrics will be computed.")
            continue

        for m_config_idx, m_config in enumerate(metric_configs):
            logger.debug(f"compute_metrics_for_layers: INNER LOOP iteration {m_config_idx}, m_config: {m_config}") # CHANGED to debug
            metric_name = m_config.get("name") 
            if not metric_name:
                logger.warning(f"compute_metrics_for_layers: Metric config found without a 'name' for layer '{layer_name}'. Skipping config: {m_config}")
                continue
            
            logger.debug(f"compute_metrics_for_layers: Attempting to compute metric '{metric_name}' for layer '{layer_name}'.") # CHANGED to debug
            scale_by_norm = m_config.get("scale_by_norm", False)
            specific_kwargs_for_metric = {k:v for k,v in m_config.items() if k not in ["name", "scale_by_norm"]}
            
            results[layer_name][metric_name] = torch.tensor(float('nan')) # Initialize with NaN

            try:
                logger.debug(f"compute_metrics_for_layers: Getting metric instance for '{metric_name}' with scale_by_norm={scale_by_norm}") # CHANGED to debug
                metric_instance = get_metric(name=metric_name, scale_by_norm=scale_by_norm)
                current_layer_weights = module.weight.detach() if hasattr(module, 'weight') and module.weight is not None else None
                
                if metric_name.upper() == "RQ":
                    logger.debug(f"compute_metrics_for_layers (RQ Pre-Call): Layer '{layer_name}', l_inputs is None: {l_inputs is None}, current_layer_weights is None: {current_layer_weights is None}") # CHANGED to debug
                    if isinstance(l_inputs, torch.Tensor):
                        logger.debug(f"compute_metrics_for_layers (RQ Pre-Call): Layer '{layer_name}', l_inputs.shape: {l_inputs.shape}") # CHANGED to debug
                    if isinstance(current_layer_weights, torch.Tensor):
                        logger.debug(f"compute_metrics_for_layers (RQ Pre-Call): Layer '{layer_name}', current_layer_weights.shape: {current_layer_weights.shape}") # CHANGED to debug

                logger.debug(f"compute_metrics_for_layers: Calling compute_per_node_scores with kwargs: {specific_kwargs_for_metric}") # CHANGED to debug
                metric_val = metric_instance.compute_per_node_scores(
                    layer_inputs=l_inputs, 
                    layer_weights=current_layer_weights, 
                    layer_outputs=l_outputs,
                    device=device, 
                    **specific_kwargs_for_metric
                )
                logger.debug(f"compute_metrics_for_layers: Result from compute_per_node_scores: {metric_val}") # CHANGED to debug
                
                if metric_val is not None:
                    logger.debug(f"compute_metrics_for_layers: Metric value for '{metric_name}' has shape: {metric_val.shape}, will store it in results") # CHANGED to debug
                    results[layer_name][metric_name] = metric_val.item() if metric_val.numel()==1 else metric_val.cpu()
                else:
                    logger.warning(f"compute_metrics_for_layers: Metric value for '{metric_name}' is None")
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}' for layer '{layer_name}': {e}", exc_info=True)
    logger.info(f"compute_metrics_for_layers: === EXITING FUNCTION ===. Results: {results}")
    return results 

# --- NEW: Pairwise Metric Calculation ---

@torch.no_grad()
def compute_pairwise_metric(
    data1: torch.Tensor,
    data2: torch.Tensor,
    metric_name: str,
    target_data: Optional[torch.Tensor] = None, # e.g., for PID's S variable
    verbose: bool = False,
    bins: int = 10 # For binning metrics
) -> Union[torch.Tensor, float]:
    """
    Compute a metric between two sets of data (activations, weights, etc.).

    Args:
        data1: First tensor [batch, features1] or [features1]
        data2: Second tensor [batch, features2] or [features2]
        metric_name: Name of the metric (e.g., 'mi_gaussian', 'correlation', 'cosine_similarity', 'pid_si').
        target_data: Optional third tensor [batch, features_target] for metrics like PID.
        verbose: Enable verbose logging.
        bins: Number of bins for discrete metrics.

    Returns:
        Metric value (float or Tensor depending on metric and input shapes).
    """
    metric_name_lower = metric_name.lower()
    device = data1.device

    # Ensure data is at least 2D [batch, features]
    if data1.ndim == 1: data1 = data1.unsqueeze(0)
    if data2.ndim == 1: data2 = data2.unsqueeze(0)
    if target_data is not None and target_data.ndim == 1: target_data = target_data.unsqueeze(0)
    
    batch_size = data1.shape[0]
    if data2.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch between data1 ({batch_size}) and data2 ({data2.shape[0]})")
    if target_data is not None and target_data.shape[0] != batch_size:
        raise ValueError(f"Batch size mismatch between data and target_data ({target_data.shape[0]})")

    # --- Metric Dispatch --- 

    try:
        if metric_name_lower in ["mi_gaussian", "mi_g", "redundancy_gaussian", "red_g"]:
            # Calculate MI using Gaussian approx between data1 and data2
            # Assuming data1 and data2 represent activities of two nodes/groups
            # Concatenate features for covariance calculation
            if data1.shape[1] > 1 or data2.shape[1] > 1:
                logger.warning(f"Pairwise {metric_name} typically compares single features. Averaging over features.")
                # Compute MI for each feature pair and average? Or compute for mean activity?
                # Let's compute for the mean activity for simplicity now.
                data1_mean = data1.mean(dim=1, keepdim=True)
                data2_mean = data2.mean(dim=1, keepdim=True)
            else:
                 data1_mean = data1
                 data2_mean = data2
            
            combined = torch.cat((data1_mean, data2_mean), dim=1)
            if combined.shape[0] < 2: return 0.0
            cov_matrix = covariance(combined, force_cpu=force_cpu_for_large_metric_ops)
            var1, var2 = cov_matrix[0, 0], cov_matrix[1, 1]
            if var1 > 1e-12 and var2 > 1e-12:
                rho_sq = (cov_matrix[0, 1]**2) / (var1 * var2)
                mi_val = -0.5 * torch.log(1.0 - torch.clamp(rho_sq, 0, 0.999999))
                return torch.nan_to_num(mi_val).item()
            else:
                return 0.0

        elif metric_name_lower in ["mi_direct", "mi_bin"]:
            # Calculate MI using binning between data1 and data2 (mean activities)
            if data1.shape[1] > 1 or data2.shape[1] > 1:
                 logger.warning(f"Pairwise {metric_name} typically compares single features. Using mean activity.")
                 data1_np = data1.mean(dim=1).cpu().numpy()
                 data2_np = data2.mean(dim=1).cpu().numpy()
            else:
                 data1_np = data1.squeeze(1).cpu().numpy()
                 data2_np = data2.squeeze(1).cpu().numpy()

            min1, max1 = np.min(data1_np), np.max(data1_np)
            bins1_vals = np.linspace(min1, max1, bins + 1)
            digitized1 = np.digitize(data1_np, bins1_vals[:-1] if max1 > min1 + 1e-9 else [min1]) - 1
            digitized1 = np.clip(digitized1, 0, bins - 1)

            min2, max2 = np.min(data2_np), np.max(data2_np)
            bins2_vals = np.linspace(min2, max2, bins + 1)
            digitized2 = np.digitize(data2_np, bins2_vals[:-1] if max2 > min2 + 1e-9 else [min2]) - 1
            digitized2 = np.clip(digitized2, 0, bins - 1)

            joint_hist = np.zeros((bins, bins), dtype=float)
            for s_idx in range(batch_size): joint_hist[digitized1[s_idx], digitized2[s_idx]] += 1
            joint_p = joint_hist / batch_size
            p1 = np.sum(joint_p, axis=1); p2 = np.sum(joint_p, axis=0)
            
            mi_val = 0.0
            for b1 in range(bins): 
                for b2 in range(bins):
                    if joint_p[b1, b2] > 1e-12 and p1[b1] > 1e-12 and p2[b2] > 1e-12:
                        mi_val += joint_p[b1, b2] * np.log2(joint_p[b1, b2] / (p1[b1] * p2[b2]))
            return mi_val

        elif metric_name_lower == "correlation":
            if data1.shape[1] > 1 or data2.shape[1] > 1:
                 logger.warning(f"Pairwise {metric_name} typically compares single features. Using mean activity.")
                 data1_mean = data1.mean(dim=1, keepdim=True)
                 data2_mean = data2.mean(dim=1, keepdim=True)
            else:
                 data1_mean = data1
                 data2_mean = data2
            
            combined = torch.cat((data1_mean, data2_mean), dim=1)
            if combined.shape[0] < 2: return 0.0
            corr_matrix = correlation(combined, force_cpu=force_cpu_for_large_metric_ops)
            return corr_matrix[0, 1].item()

        elif metric_name_lower == "cosine_similarity":
            # Interpret data1 and data2 as vectors (e.g., weight vectors or mean activation vectors)
            vec1 = data1.mean(dim=0) # Average over batch if multiple samples given
            vec2 = data2.mean(dim=0)
            return F.cosine_similarity(vec1, vec2, dim=0).item()

        elif "pid_" in metric_name_lower:
            if not PID_AVAILABLE:
                logger.warning(f"PID metric '{metric_name}' requested, but BROJA_2PID not available. Returning 0.")
                return 0.0
            if target_data is None:
                raise ValueError("PID metrics require target_data (S variable) for pairwise calculation.")
            
            broja = get_broja_pid_module()
            
            # Use mean activities if multiple features
            if data1.shape[1] > 1: data1 = data1.mean(dim=1, keepdim=True)
            if data2.shape[1] > 1: data2 = data2.mean(dim=1, keepdim=True)
            if target_data.shape[1] > 1: target_data = target_data.mean(dim=1, keepdim=True)

            y1_np = data1.squeeze(1).cpu().numpy()
            y2_np = data2.squeeze(1).cpu().numpy()
            s_np = target_data.squeeze(1).cpu().numpy()

            # Digitize y1, y2, s (similar to average_pid_component)
            min_y1, max_y1 = np.min(y1_np), np.max(y1_np)
            y1_bins_vals = np.linspace(min_y1, max_y1, bins + 1)
            y1_digitized = np.digitize(y1_np, y1_bins_vals[:-1] if max_y1 > min_y1+1e-9 else [min_y1]) - 1
            y1_digitized = np.clip(y1_digitized, 0, bins - 1)

            min_y2, max_y2 = np.min(y2_np), np.max(y2_np)
            y2_bins_vals = np.linspace(min_y2, max_y2, bins + 1)
            y2_digitized = np.digitize(y2_np, y2_bins_vals[:-1] if max_y2 > min_y2+1e-9 else [min_y2]) - 1
            y2_digitized = np.clip(y2_digitized, 0, bins - 1)

            min_s, max_s = np.min(s_np), np.max(s_np)
            s_bins_vals = np.linspace(min_s, max_s, bins + 1)
            s_digitized = np.digitize(s_np, s_bins_vals[:-1] if max_s > min_s+1e-9 else [min_s]) - 1
            s_digitized = np.clip(s_digitized, 0, bins - 1)
            
            # Compute joint distribution p(S, Y1, Y2)
            p_sy1y2_dict: Dict[Tuple[int,int,int], float] = {}
            for s_idx in range(batch_size):
                key = (s_digitized[s_idx], y1_digitized[s_idx], y2_digitized[s_idx])
                p_sy1y2_dict[key] = p_sy1y2_dict.get(key, 0.0) + 1.0
            
            if not p_sy1y2_dict: return 0.0
            total_counts = sum(p_sy1y2_dict.values())
            if total_counts == 0: return 0.0
            broja_input = {k: v / total_counts for k, v in p_sy1y2_dict.items()}
            if not broja_input: return 0.0

            pid_res = broja.pid(broja_input)
            # Map requested metric name to BROJA output key
            comp_val_map = {
                "pid_si": "SI", "pid_shared_info": "SI",
                "pid_ui1": "UIY", "pid_unique_info_1": "UIY",
                "pid_ui2": "UIZ", "pid_unique_info_2": "UIZ",
                "pid_ci": "CI", "pid_synergy_info": "CI"
            }
            return pid_res.get(comp_val_map.get(metric_name_lower, ""), 0.0)

        else:
            raise ValueError(f"Unsupported pairwise metric: {metric_name_lower}")
            
    except Exception as e:
        logger.error(f"Error computing pairwise metric '{metric_name_lower}': {e}", exc_info=verbose)
        return 0.0 # Return default value on error 

# --- NEW: compute_all_node_scores --- 
def compute_all_node_scores(
    model: nn.Module,
    metric_configs: List[Dict[str, Any]], 
    device: Union[str, torch.device],
    data_loader: DataLoader, 
    num_batches: Optional[int] = 1, 
    debug_mode: bool = False 
) -> Dict[str, Dict[str, torch.Tensor]]: 
    """
    Computes node scores for specified metrics for all relevant layers in a model.
    This version uses the centralized `activation_utils.collect_layer_data`.
    """
    eff_device = torch.device(device) if isinstance(device, str) else device
    model.to(eff_device)
    model.eval()

    # Determine target layers for activation collection
    target_layers_for_hooks = []
    # If model is AlignmentNetwork, use its defined alignment_names
    # This requires AlignmentNetwork to be duck-typed or imported carefully to avoid circularity
    # Assuming model is AlignmentNetwork or DDP-wrapped AlignmentNetwork, as called from dropout_manager
    actual_model_for_names = model
    if hasattr(model, 'module'): # DDP wrapped
        actual_model_for_names = model.module
    
    if hasattr(actual_model_for_names, 'alignment_names') and actual_model_for_names.alignment_names:
        target_layers_for_hooks = actual_model_for_names.alignment_names
        logger.debug(f"compute_all_node_scores: Using alignment_names for hooks: {target_layers_for_hooks}")
    else:
        # Fallback for generic nn.Module: hook layers with weights (e.g., Linear, Conv)
        # This might need adjustment based on which layers are truly relevant for scoring.
        for name, module_item in model.named_modules():
            if isinstance(module_item, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                target_layers_for_hooks.append(name)
        if not target_layers_for_hooks:
            logger.warning("compute_all_node_scores: No target layers identified for generic model. Returning empty scores.")
            return {}
        logger.debug(f"compute_all_node_scores: Using weight-having layers for hooks: {target_layers_for_hooks}")

    if not target_layers_for_hooks:
        logger.warning("compute_all_node_scores: No layers to hook. Cannot compute scores.")
        return {}

    # Determine if inputs/outputs are needed based on metric_configs
    # Simplified: assume RQ/Redundancy/PID need inputs, MI/PID need outputs. Collect both for safety.
    # A more granular check could be done by inspecting metric_fn signatures or specific metric needs.
    collect_inputs_flag = False
    collect_outputs_flag = False
    for m_conf in metric_configs:
        m_name_lower = m_conf.get("name", "").lower()
        if "rayleigh_quotient" in m_name_lower or "rq" in m_name_lower or "redundancy" in m_name_lower or "pid_" in m_name_lower:
            collect_inputs_flag = True
        if "mi_" in m_name_lower or "pid_" in m_name_lower:
            collect_outputs_flag = True
    
    if not collect_inputs_flag and not collect_outputs_flag:
        # If no metrics seem to require inputs or outputs, default to collecting outputs as MI_G might be a default.
        # Or, if only weight-based metrics were hypothetically supported, this might be fine.
        # For now, let's ensure at least outputs are collected if nothing specific is triggered.
        collect_outputs_flag = True 
        logger.debug("compute_all_node_scores: No specific input/output needs detected from metrics, defaulting to collect_outputs=True.")


    # Use the centralized collect_layer_data function
    # `model` here is the AlignmentNetwork (or DDP wrapped one), collect_layer_data handles base_model logic.
    collected_activations = collect_layer_data(
        model=model, # Pass the potentially DDP-wrapped AlignmentNetwork
        dataloader=data_loader,
        target_layers=target_layers_for_hooks,
        num_batches=num_batches if num_batches is not None else 1,
        device=eff_device,
        collect_inputs=collect_inputs_flag,
        collect_outputs=collect_outputs_flag,
        flatten_spatial=True # Default flattening, individual metrics might need to unflatten if necessary
    )

    if not collected_activations:
        logger.warning("compute_all_node_scores: Activation collection returned no data.")
        return {}

    # Ensure 'verbose' is set in metric_configs based on debug_mode for compute_metrics_for_layers
    updated_metric_configs = []
    for m_conf in metric_configs:
        conf_copy = m_conf.copy()
        conf_copy.setdefault("verbose", debug_mode)
        updated_metric_configs.append(conf_copy)

    # Call compute_metrics_for_layers with the collected data
    layer_metric_scores = compute_metrics_for_layers(
        model=model, 
        collected_data=collected_activations, 
        metric_configs=updated_metric_configs, # Pass updated configs
        device=eff_device
    )
    return layer_metric_scores

# The old internal hook logic in compute_all_node_scores (make_hook_fn, etc.) is removed by this refactoring.

# --- NEW METRIC: Weight Similarity ---
@torch.no_grad()
def compute_weight_similarity(
    layer_weights: torch.Tensor, 
    metric_type: str = "cosine", 
    verbose: bool = False, # Added for signature consistency, though not used here
    **kwargs # To catch any other unused specific kwargs passed by the metric system
) -> torch.Tensor:
    """
    Measure similarity between weight vectors of a layer.
    Note: This metric operates only on weights and does not use layer_inputs or layer_outputs.
          The result is a pairwise similarity matrix, not per-node scores in the typical sense.

    Args:
        layer_weights: Weight tensor of the layer (out_features, in_features).
        metric_type: Similarity metric to use ("cosine", "dot", "euclidean").
        verbose: Unused, for signature consistency.
        **kwargs: Unused, for signature consistency.

    Returns:
        Tensor containing pairwise similarity values [out_features, out_features].
    """
    if layer_weights is None or layer_weights.ndim != 2:
        logger.warning("compute_weight_similarity: layer_weights is None or not 2D. Returning empty tensor.")
        return torch.empty(0, device=layer_weights.device if layer_weights is not None else 'cpu')

    if metric_type == "cosine":
        # Normalize weights for cosine similarity
        w_norm = layer_weights / (torch.norm(layer_weights, dim=1, keepdim=True) + 1e-12) # Added epsilon for stability
        similarity = torch.mm(w_norm, w_norm.t())
    elif metric_type == "dot":
        similarity = torch.mm(layer_weights, layer_weights.t())
    elif metric_type == "euclidean":
        # cdist computes pairwise distances. For similarity, we might want 1/(1+dist) or exp(-dist), 
        # or simply return the distance matrix. For now, returning distance.
        similarity = torch.cdist(layer_weights, layer_weights)
    else:
        raise ValueError(f"Unknown weight similarity metric_type: {metric_type}. Supported: cosine, dot, euclidean")
            
    return similarity

# Update ALIGNMENT_METRICS_REGISTRY
# Option 1: Register one function and pass metric_type via kwargs (preferred)
# ALIGNMENT_METRICS_REGISTRY["weight_similarity"] = compute_weight_similarity

# Option 2: Create specific wrappers if metric_type needs to be part of the name (more verbose in registry)
# This is chosen to align with how PID components are handled (distinct names for distinct operations)

def compute_weight_cosine_similarity(layer_weights: torch.Tensor, **kwargs) -> torch.Tensor:
    return compute_weight_similarity(layer_weights, metric_type="cosine", **kwargs)

def compute_weight_dot_similarity(layer_weights: torch.Tensor, **kwargs) -> torch.Tensor:
    return compute_weight_similarity(layer_weights, metric_type="dot", **kwargs)

def compute_weight_euclidean_distance(layer_weights: torch.Tensor, **kwargs) -> torch.Tensor:
    return compute_weight_similarity(layer_weights, metric_type="euclidean", **kwargs)

ALIGNMENT_METRICS_REGISTRY.update({
    "weight_cosine_similarity": compute_weight_cosine_similarity,
    "weight_dot_similarity": compute_weight_dot_similarity,
    "weight_euclidean_distance": compute_weight_euclidean_distance,
    # Add other metric aliases or new metrics here
})

# --- NEW METRIC: MI (Projected Input vs Mean Input) ---
@torch.no_grad()
def compute_mi_proj_vs_mean_input(
    layer_inputs: torch.Tensor,
    layer_weights: torch.Tensor,
    bins: int = 30, 
    eps: float = 1e-9, 
    force_cpu_for_large_metric_ops: bool = True, 
    verbose: bool = False,
    **kwargs # Catch-all for unused specific kwargs
) -> torch.Tensor:
    """
    Measure Mutual Information between each neuron's projection of normalized inputs 
    and the mean of all normalized input features.
    Adapted from MIMetric in metrics_utils.py.
    
    Args:
        layer_inputs: Input activations tensor (batch, features).
        layer_weights: Weight tensor of the layer (out_features, in_features).
        bins: Number of bins for histogram.
        eps: Small value to prevent division by zero.
        force_cpu_for_large_metric_ops: Whether to force CPU for large metric operations.
        verbose: Unused, for signature consistency.

    Returns:
        Tensor containing MI values per output neuron [out_features].
    """
    if layer_inputs is None or layer_weights is None:
        logger.warning("compute_mi_proj_vs_mean_input: layer_inputs or layer_weights is None. Returning empty tensor.")
        return torch.empty(0, device=layer_weights.device if layer_weights is not None else (layer_inputs.device if layer_inputs is not None else 'cpu'))

    if layer_inputs.ndim != 2 or layer_weights.ndim != 2:
        logger.warning("compute_mi_proj_vs_mean_input: layer_inputs and layer_weights must be 2D. Returning empty tensor.")
        return torch.empty(0, device=layer_weights.device)
    
    if layer_inputs.shape[1] != layer_weights.shape[1]:
        logger.warning("compute_mi_proj_vs_mean_input: Mismatch in features between inputs and weights. Returning empty tensor.")
        return torch.empty(0, device=layer_weights.device)
    
    # Handle empty tensors
    if layer_inputs.numel() == 0 or layer_weights.numel() == 0 or layer_inputs.shape[0] == 0:
        logger.warning("compute_mi_proj_vs_mean_input: Empty input tensor detected. Returning empty tensor.")
        return torch.empty(0, device=layer_weights.device)

    original_input_device = layer_inputs.device
    mi_scores_device = layer_weights.device

    perform_on_cpu = False
    if force_cpu_for_large_metric_ops and layer_inputs.is_cuda and \
       ((layer_inputs.shape[0] > 200_000 and layer_inputs.shape[1] > 50) or (layer_inputs.numel() > 10_000_000)):
        if verbose: logger.info(f"MI_ProjVsMean: Large input tensor ({layer_inputs.shape}) on CUDA. Offloading to CPU.")
        perform_on_cpu = True

    current_inputs = layer_inputs.cpu() if perform_on_cpu else layer_inputs
    current_weights = layer_weights.cpu() if perform_on_cpu else layer_weights
    
    min_in, max_in = current_inputs.min(), current_inputs.max()
    X_normalized = (current_inputs - min_in) / (max_in - min_in + eps) if (max_in - min_in).abs() > eps else torch.zeros_like(current_inputs)
    
    norm_W = torch.norm(current_weights, dim=1, keepdim=True)
    W_norm = current_weights / (norm_W + eps)
    
    projections = torch.matmul(X_normalized, W_norm.t()) # Shape: [batch_size, out_features]
    mi_scores = torch.zeros(layer_weights.shape[0], device=mi_scores_device)
    y_variable_for_mi = current_inputs.mean(dim=1) # Shape: [batch_size]

    for i in range(projections.shape[1]):  # Iterate over output neurons
        x_node_projection = projections[:, i].contiguous() # Already on CPU if perform_on_cpu is True
        
        if x_node_projection.numel() == 0 or y_variable_for_mi.numel() == 0:
            continue 
            
        min_val_x, max_val_x = x_node_projection.min().item(), x_node_projection.max().item()
        min_val_y, max_val_y = y_variable_for_mi.min().item(), y_variable_for_mi.max().item()

        range_x_tuple = (min_val_x, max_val_x) if max_val_x > min_val_x + eps else None
        range_y_tuple = (min_val_y, max_val_y) if max_val_y > min_val_y + eps else None

        if x_node_projection.shape[0] != y_variable_for_mi.shape[0] or range_x_tuple is None or range_y_tuple is None:
            if verbose: logger.debug(f"MI_ProjVsMean node {i}: Skipping due to shape mismatch or zero range.")
            continue
        
        # Ensure data for np.histogram2d is on CPU and flattened numpy array
        x_np = x_node_projection.cpu().numpy().flatten()
        y_np = y_variable_for_mi.cpu().numpy().flatten()

        try:
            hist_xy_np, _, _ = np.histogram2d(x_np, y_np, bins=bins, range=[list(range_x_tuple), list(range_y_tuple)], density=True)
            hist_xy = torch.from_numpy(hist_xy_np).float().to(current_inputs.device) # Move to processing device (CPU or original)

            # For hist_x and hist_y, use torch.histogram for consistency if possible, or ensure CPU for np
            hist_x_torch, _ = torch.histogram(x_node_projection.to(current_inputs.device), bins=bins, range=range_x_tuple, density=True)
            hist_y_torch, _ = torch.histogram(y_variable_for_mi.to(current_inputs.device), bins=bins, range=range_y_tuple, density=True)

        except Exception as e_hist:
            if verbose: logger.warning(f"MI_ProjVsMean node {i}: Error in histogramming: {e_hist}")
            continue
            
        bin_width_x = (range_x_tuple[1] - range_x_tuple[0]) / bins
        bin_width_y = (range_y_tuple[1] - range_y_tuple[0]) / bins
            
        px = hist_x_torch * bin_width_x 
        py = hist_y_torch * bin_width_y 
        pxy = hist_xy * bin_width_x * bin_width_y

        px = torch.clamp(px, min=eps); py = torch.clamp(py, min=eps); pxy = torch.clamp(pxy, min=eps)
            
        h_x = -torch.sum(px[px>0] * torch.log2(px[px>0]))
        h_y = -torch.sum(py[py>0] * torch.log2(py[py>0]))
        h_xy = -torch.sum(pxy[pxy>0] * torch.log2(pxy[pxy>0]))
            
        mi_val = h_x + h_y - h_xy
        mi_scores[i] = torch.clamp(mi_val, min=0.0)

    return mi_scores.to(mi_scores_device)

# --- NEW METRIC: Alternative Rayleigh Quotient ---
@torch.no_grad()
def compute_rq_alternative_denominator(
    layer_inputs: torch.Tensor, 
    layer_weights: torch.Tensor, 
    relative: bool = True, 
    epsilon: float = 1e-8, 
    verbose: bool = False, # Added for signature consistency
    **kwargs # Catch-all for unused specific kwargs
) -> torch.Tensor:
    """
    Alternative Representation Quality (RQ) metric based on metrics_utils.RQMetric.
    Uses a different denominator and relative scaling compared to compute_rayleigh_quotient.
    Denominator involves norm(weights @ cov).
    Relative scaling is by sqrt(d_in).
    """
    if layer_inputs is None or layer_weights is None:
        logger.warning("compute_rq_alternative_denominator: layer_inputs or layer_weights is None.")
        return torch.empty(0, device=layer_weights.device if layer_weights is not None else (layer_inputs.device if layer_inputs is not None else 'cpu'))

    if layer_inputs.ndim != 2 or layer_weights.ndim != 2:
        # This specific implementation might be more flexible, but for consistency with compute_rayleigh_quotient:
        logger.warning("compute_rq_alternative_denominator: inputs/weights not 2D. This metric might expect specific shapes.")
        # Attempt to flatten if possible, similar to primary RQ, but behavior might differ from original metrics_utils version
        if layer_inputs.ndim > 2 and layer_inputs.shape[0] > 0: layer_inputs = layer_inputs.flatten(start_dim=1)
        else: return torch.zeros(layer_weights.shape[0], device=layer_weights.device, dtype=layer_weights.dtype)
        if layer_weights.ndim > 2 and layer_weights.shape[0] > 0: layer_weights = layer_weights.reshape(layer_weights.shape[0], -1)
        else: return torch.zeros(layer_weights.shape[0], device=layer_weights.device, dtype=layer_weights.dtype)
    
    if layer_inputs.shape[0] < 2: # Need at least 2 samples for covariance
        logger.warning("compute_rq_alternative_denominator: Not enough samples for covariance.")
        return torch.zeros(layer_weights.shape[0], device=layer_weights.device, dtype=layer_weights.dtype)

    if layer_inputs.shape[1] != layer_weights.shape[1]:
        # Handle feature mismatch by truncation, similar to primary RQ, though original might not have done this.
        logger.warning(f"Dimension mismatch for RQ_alt: weights_shape[1]={layer_weights.shape[1]}, l_inputs.shape[1]={layer_inputs.shape[1]}. Truncating.")
        min_dim = min(layer_weights.shape[1], layer_inputs.shape[1])
        if min_dim == 0: return torch.zeros(layer_weights.shape[0], device=layer_weights.device, dtype=layer_weights.dtype)
        layer_weights = layer_weights[:, :min_dim]
        layer_inputs = layer_inputs[:, :min_dim]

    # Center the inputs
    X = layer_inputs - layer_inputs.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    # Note: metrics_utils.RQMetric did not use the force_cpu flag for covariance.
    # Adding it here for consistency if it were to be passed via kwargs.
    force_cpu = kwargs.get("force_cpu_for_large_metric_ops", False)
    cov = covariance(X, force_cpu=force_cpu) # Using the covariance util from metrics.py
    cov = cov + torch.eye(cov.size(0), device=cov.device) * epsilon # Add small value to diagonal for stability
    
    # Compute the RQ values
    weights_at_cov = torch.matmul(layer_weights, cov)
    numerator = torch.sum(layer_weights * weights_at_cov, dim=1)
    denominator = (torch.norm(layer_weights, dim=1).pow(2)) * (torch.norm(weights_at_cov, dim=1) + epsilon)
    
    rq = numerator / (denominator + epsilon) # Fixed: use epsilon instead of eps
    
    if relative:
        d_in = layer_weights.size(1) # Number of input features
        if d_in > 0:
            rq = rq * torch.sqrt(torch.tensor(d_in, device=rq.device, dtype=rq.dtype))
        else:
            rq.fill_(0.0)
            
    return torch.nan_to_num(rq)

# ... (rest of metrics.py, including _AlignmentMetricImpl adjustments if needed for these new metrics)

# In _AlignmentMetricImpl.compute_per_node_scores, we need to handle these weight-only metrics.
# The current dispatch logic primarily expects layer_inputs or layer_outputs.
# We might need a new category or a check if only layer_weights are needed.

# Consider adjusting the dispatch in _AlignmentMetricImpl:
# Inside _AlignmentMetricImpl.compute_per_node_scores:
# ...
# elif "weight_" in metric_name_lower: # Or a more specific check
#    if layer_weights is None:
#        logger.error(f"{self.name} needs layer_weights.")
#        return torch.zeros(1, device=eff_device, dtype=torch.float32) # Or appropriate shape
#    logger.info(f"compute_per_node_scores({self.name}): Calling metric with layer_weights.shape = {layer_weights.shape}")
#    # metric_specific_kwargs might contain 'metric_type' for the generic compute_weight_similarity
#    # If using specific registered functions like compute_weight_cosine_similarity, they handle metric_type internally.
#    result = self._metric_fn(layer_weights=layer_weights.to(eff_device), **fn_kwargs, **metric_specific_kwargs)
#    logger.info(f"compute_per_node_scores({self.name}): Result shape: {result.shape}")
#    return result

# ...

# Update the metrics registry with the newly defined metric functions
# This needs to be at the very end of the file to avoid circular references
ALIGNMENT_METRICS_REGISTRY.update({
    "weight_cosine_similarity": compute_weight_cosine_similarity,
    "weight_dot_similarity": compute_weight_dot_similarity,
    "weight_euclidean_distance": compute_weight_euclidean_distance,
    "rq_alt_denom": compute_rq_alternative_denominator,
    "mi_proj_vs_mean_input": compute_mi_proj_vs_mean_input,
})