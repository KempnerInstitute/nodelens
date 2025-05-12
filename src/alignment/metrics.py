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
            # logger.error(f"RQ: inputs not 2D ({layer_inputs.shape}). Ret zeros."); 
            return zeros_output()
    if layer_weights.ndim != 2:
        if layer_weights.ndim > 2 and layer_weights.shape[0] > 0:
            layer_weights = layer_weights.reshape(layer_weights.shape[0], -1)
        else:
            # logger.error(f"RQ: weights not 2D ({layer_weights.shape}). Ret zeros."); 
            return zeros_output()
    if layer_inputs.shape[0] < min_samples_for_cov: 
        # logger.warning(f"RQ: samples < {min_samples_for_cov}. Ret zeros."); 
        return zeros_output()
    if layer_weights.shape[1] != layer_inputs.shape[1]:
        min_dim = min(layer_weights.shape[1], layer_inputs.shape[1])
        layer_weights = layer_weights[:, :min_dim]; layer_inputs = layer_inputs[:, :min_dim]
        if min_dim == 0: 
            # logger.error("RQ: 0 common feature dim. Ret zeros."); 
            return zeros_output()
    if layer_inputs.numel() == 0 or layer_weights.numel() == 0: 
        # logger.warning("RQ: empty inputs/weights. Ret zeros."); 
        return zeros_output()
    try:
        # Pass force_cpu_for_large_metric_ops to covariance
        C = covariance(layer_inputs.to(layer_weights.device), force_cpu=force_cpu_for_large_metric_ops)
        if torch.isnan(C).any() or torch.isinf(C).any(): 
            # logger.warning("RQ: NaN/Inf in cov. Ret zeros."); 
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
        # logger.warning(f"MI_gauss: outputs not 2D or <2 samples. Ret zeros."); 
        return zeros_output(layer_outputs.shape[1] if layer_outputs.ndim == 2 else 1)
    
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
            # logger.warning(f"MI_gauss: PC1 calc failed: {e}. Ret zeros."); 
            return zeros_output(num_neurons)
    
    if ref_data.ndim == 1: ref_data = ref_data.unsqueeze(1)
    if ref_data.shape[0] != batch_size: 
        # logger.warning(f"MI_gauss: ref_data batch mismatch. Ret zeros."); 
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
        # logger.warning(f"MI_direct: outputs not 2D or <2 samples. Ret zeros."); 
        return zeros_output(layer_outputs.shape[1] if layer_outputs.ndim == 2 else 1)

    batch_size, num_neurons = layer_outputs.shape
    mi_scores = torch.zeros(num_neurons, device=layer_outputs.device)
    source_np = layer_outputs.cpu().numpy()

    ref_np: Optional[np.ndarray] = None
    if target_outputs is not None:
        if target_outputs.ndim == 1: target_outputs = target_outputs.unsqueeze(1)
        if target_outputs.shape[0] != batch_size: 
            # logger.warning(f"MI_direct: target batch mismatch. Ret zeros."); 
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
        # logger.warning("Red_gauss: inputs problem. Ret zeros."); 
        return zeros_output(layer_weights.shape[0])
    if layer_weights.ndim!=2 or layer_weights.shape[0]<=1: 
        # logger.warning("Red_gauss: weights problem. Ret zeros."); 
        return zeros_output(layer_weights.shape[0])
    
    num_neurons = layer_weights.shape[0]
    avg_red_scores = torch.zeros(num_neurons, device=layer_weights.device)
    projected_outputs = torch.matmul(layer_inputs, layer_weights.T)
    if projected_outputs.shape[0] < 2: return avg_red_scores

    try: corr_matrix_projections = correlation(projected_outputs, force_cpu=force_cpu_for_large_metric_ops)
    except Exception as e: 
        # logger.warning(f"Red_gauss: corr matrix failed: {e}. Ret zeros."); 
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
        # logger.warning(f"PID_{pid_component_name}: BROJA not available. Ret zeros."); 
        return zeros_output(layer_outputs.shape[1])
    
    broja = get_broja_pid_module() # Get the module (could still be dummy if loaded but unusable?)

    if layer_inputs.ndim!=2 or layer_inputs.shape[0]<2: 
        # logger.warning("PID: inputs problem. Ret zeros."); 
        return zeros_output(layer_outputs.shape[1])
    if layer_outputs.ndim!=2 or layer_outputs.shape[0]<2 or layer_outputs.shape[1]<=1: 
        # logger.warning("PID: outputs problem. Ret zeros."); 
        return zeros_output(layer_outputs.shape[1])

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
            # else: logger.warning("PID: Not enough samples/features for PCA on inputs. Using first feature.")
        except Exception as e_pca:
            # logger.warning(f"PID: PCA on inputs failed: {e_pca}. Using first feature.");
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
                # if verbose: logger.warning(f"PID: BROJA fail ({i},{j}): {e_pid}. Skip.")
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
        
        # Extract relevant kwargs for specific metric functions, pass others via **fn_kwargs
        # This helps avoid passing unsupported args to underlying functions.
        fn_kwargs = {"verbose": verbose}
        force_cpu_flag = metric_specific_kwargs.get("force_cpu_for_large_metric_ops", False)
        # configured_cnn_mode = metric_specific_kwargs.get("configured_cnn_mode", "unfold") # Example if needed
        # configured_cnn_rq_op = metric_specific_kwargs.get("configured_cnn_rq_op", "mean") # Example if needed

        if "rayleigh_quotient" in metric_name_lower or "rq" in metric_name_lower:
            if layer_inputs is None or layer_weights is None: raise ValueError(f"{self.name} needs layer_inputs and layer_weights.")
            fn_kwargs["relative"] = self.scale_by_norm
            fn_kwargs["min_samples_for_cov"] = min_samples_for_cov
            fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
            return self._metric_fn(layer_inputs.to(eff_device), layer_weights.to(eff_device), **fn_kwargs)
        
        elif "mi_gaussian" in metric_name_lower or "mi_g" in metric_name_lower:
            if layer_outputs is None: raise ValueError(f"{self.name} needs layer_outputs.")
            if target_outputs is not None: fn_kwargs["target_outputs"] = target_outputs.to(eff_device)
            if "order" in metric_specific_kwargs: fn_kwargs["order"] = metric_specific_kwargs["order"]
            fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
            return self._metric_fn(layer_outputs.to(eff_device), **fn_kwargs)

        elif "mi_direct" in metric_name_lower or "mi_bin" in metric_name_lower:
            if layer_outputs is None: raise ValueError(f"{self.name} needs layer_outputs.")
            if target_outputs is not None: fn_kwargs["target_outputs"] = target_outputs.to(eff_device)
            fn_kwargs["bins"] = bins
            # mi_direct_binning doesn't use covariance, so force_cpu_flag isn't directly passed to it.
            return self._metric_fn(layer_outputs.to(eff_device), **fn_kwargs)

        elif "redundancy_gaussian" in metric_name_lower or "red_g" in metric_name_lower:
            if layer_inputs is None or layer_weights is None: raise ValueError(f"{self.name} needs layer_inputs and layer_weights.")
            fn_kwargs["force_cpu_for_large_metric_ops"] = force_cpu_flag # Pass it on
            return self._metric_fn(layer_inputs.to(eff_device), layer_weights.to(eff_device), **fn_kwargs)
        
        elif "pid_" in metric_name_lower:
            if not PID_AVAILABLE: 
                logger.warning(f"PID metric {self.name} called; BROJA_2PID not available. Returning zeros.")
                num_n = layer_outputs.shape[1] if layer_outputs is not None else (layer_inputs.shape[1] if layer_inputs is not None else 1)
                return torch.zeros(num_n, device=eff_device)
            if layer_inputs is None or layer_outputs is None: raise ValueError(f"{self.name} needs layer_inputs and layer_outputs.")
            fn_kwargs["bins"] = bins
            return self._metric_fn(layer_inputs.to(eff_device), layer_outputs.to(eff_device), **fn_kwargs)
        else:
            raise NotImplementedError(f"Dispatch for {self.name} in _AlignmentMetricImpl.compute_per_node_scores not implemented.")

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

ALIGNMENT_METRICS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "rayleigh_quotient": compute_rayleigh_quotient,
    "rq": compute_rayleigh_quotient,
    "mi_gaussian": mi_gaussian_approx,
    "mi_g": mi_gaussian_approx,
    "mi_direct": mi_direct_binning,
    "mi_bin": mi_direct_binning,
    "redundancy_gaussian": average_redundancy_gaussian,
    "red_g": average_redundancy_gaussian,
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
    results: Dict[str, Dict[str, Union[torch.Tensor, float]]] = {}
    model.to(device); model.eval()
    modules_dict = {name: mod for name, mod in model.named_modules()}

    for layer_name, layer_data in collected_data.items():
        results[layer_name] = {}
        module = modules_dict.get(layer_name)
        if not module: continue
        
        l_inputs = layer_data.get("input"); l_outputs = layer_data.get("output")

        for m_config in metric_configs:
            metric_name = m_config["name"]
            scale_by_norm = m_config.get("scale_by_norm", False)
            specific_kwargs_for_metric = {k:v for k,v in m_config.items() if k not in ["name", "scale_by_norm"]}

            try:
                # Get the AlignmentMetric wrapper instance
                metric_instance = get_metric(name=metric_name, scale_by_norm=scale_by_norm)
                
                # Call its compute_per_node_scores method
                metric_val = metric_instance.compute_per_node_scores(
                    layer_inputs=l_inputs, 
                    layer_weights=module.weight.detach() if hasattr(module, 'weight') and module.weight is not None else None, 
                    layer_outputs=l_outputs,
                    device=device, 
                    **specific_kwargs_for_metric
                )
                if metric_val is not None:
                    results[layer_name][metric_name] = metric_val.item() if metric_val.numel()==1 else metric_val.cpu()
                else: results[layer_name][metric_name] = torch.tensor(float('nan'))
            except Exception as e:
                verbose_logging = specific_kwargs_for_metric.get("verbose", False)
                logger.error(f"Error computing metric '{metric_name}' for layer '{layer_name}': {e}", exc_info=verbose_logging)
                results[layer_name][metric_name] = torch.tensor(float('nan'))
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
    metric_configs: List[Dict[str, Any]], # e.g. [{"name": "RQ", "scale_by_norm": False, "min_samples_for_cov": 2}]
    device: Union[str, torch.device],
    data_loader: DataLoader, # DataLoader from which to get batches
    num_batches: Optional[int] = 1, # Number of batches to process to collect activations
    # The following are passed down to compute_metrics_for_layers or individual metrics via metric_configs if needed
    # force_cpu_for_large_metric_ops: bool = True, # This should be a kwarg within a metric_config if metric-specific
    # configured_cnn_mode: Optional[str] = "unfold", # This is now part of ModelConfig for AlignmentNetwork wrapper
    # configured_cnn_rq_op: Optional[str] = "mean", # This is also part of AlignmentConfig
    debug_mode: bool = False # For verbose logging if any metric supports it
) -> Dict[str, Dict[str, torch.Tensor]]: # Returns {layer_name: {metric_name: scores_tensor}}
    """
    Computes node scores for specified metrics for all relevant layers in a model.
    This involves running data through the model to collect activations.
    """
    eff_device = torch.device(device) if isinstance(device, str) else device
    model.to(eff_device)
    model.eval()

    # Identify layers to hook (e.g., all layers with weights, or specific ones if model is AlignmentNetwork)
    # For a generic nn.Module, we might hook all modules that have parameters or are common layer types.
    # If model is AlignmentNetwork, it has self.alignment_names and self.alignment_layers.
    layers_to_hook = []
    if hasattr(model, 'alignment_layers') and hasattr(model, 'alignment_names'): # Check if it's an AlignmentNetwork
        # Use the pre-defined alignment layers from AlignmentNetwork
        for name, layer in zip(model.alignment_names, model.alignment_layers):
            layers_to_hook.append((name, layer))
    else:
        # For generic nn.Module, hook identifiable layers (e.g., Conv, Linear)
        # This basic heuristic might need refinement for complex custom models.
        for name, module_item in model.named_modules():
            if isinstance(module_item, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layers_to_hook.append((name, module_item))
    
    if not layers_to_hook:
        logger.warning("compute_all_node_scores: No suitable layers found to hook in the model.")
        return {}

    # Setup hooks to capture inputs and outputs
    captured_activations: Dict[str, Dict[str, List[torch.Tensor]]] = {name: {"input": [], "output": []} for name, _ in layers_to_hook}
    hooks = []

    def make_hook_fn(layer_name_key: str, input_or_output: str):
        def hook_fn(module, m_input, m_output):
            # For inputs, m_input is a tuple. We usually care about the first element.
            data_to_store = m_input[0] if input_or_output == "input" else m_output
            if isinstance(data_to_store, torch.Tensor):
                # Detach and move to CPU to avoid accumulating on GPU memory during forward passes
                # This is important if num_batches is large.
                # However, for metric computation, they might need to be on `eff_device` later.
                # Let's keep on original device for now, assuming num_batches is small for this typical use case.
                # If OOM occurs, consider .cpu() here and .to(eff_device) in compute_metrics_for_layers.
                captured_activations[layer_name_key][input_or_output].append(data_to_store.detach())
            elif isinstance(data_to_store, tuple): # Some layers might return tuples
                 if data_to_store and isinstance(data_to_store[0], torch.Tensor):
                    captured_activations[layer_name_key][input_or_output].append(data_to_store[0].detach())
        return hook_fn

    for layer_name, layer_module in layers_to_hook:
        hooks.append(layer_module.register_forward_hook(make_hook_fn(layer_name, "input")))
        hooks.append(layer_module.register_forward_hook(make_hook_fn(layer_name, "output")))

    # Pass data through the model to trigger hooks
    batches_processed = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            # Assuming batch_data is (inputs, targets) or just inputs
            if isinstance(batch_data, (list, tuple)):
                inputs, _ = batch_data[0].to(eff_device), batch_data[1] # Only move inputs to device
            else:
                inputs = batch_data.to(eff_device)
            
            try:
                _ = model(inputs)
                batches_processed += 1
            except Exception as e:
                logger.error(f"Error during model forward pass in compute_all_node_scores for batch {batch_idx}: {e}", exc_info=debug_mode)
                # Potentially break or continue based on severity
                break 

    # Remove hooks
    for h in hooks: h.remove()

    if batches_processed == 0:
        logger.warning("compute_all_node_scores: No batches were processed. Cannot compute metrics.")
        return {}

    # Consolidate captured data: Dict[str, Dict[str, torch.Tensor]]
    # Each list of batch tensors should be concatenated.
    collected_data_for_metric_computation: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name, io_data in captured_activations.items():
        collected_data_for_metric_computation[layer_name] = {}
        if io_data["input"]: 
            try: collected_data_for_metric_computation[layer_name]["input"] = torch.cat(io_data["input"]) 
            except RuntimeError as e: logger.error(f"Error concatenating inputs for {layer_name}: {e}")
        if io_data["output"]: 
            try: collected_data_for_metric_computation[layer_name]["output"] = torch.cat(io_data["output"])
            except RuntimeError as e: logger.error(f"Error concatenating outputs for {layer_name}: {e}")
    
    # Call compute_metrics_for_layers with the collected data
    # Pass specific metric_configs which may include kwargs for individual metrics
    layer_metric_scores = compute_metrics_for_layers(
        model=model, 
        collected_data=collected_data_for_metric_computation, 
        metric_configs=metric_configs, # This now includes specific kwargs per metric
        device=eff_device
        # force_cpu_for_large_metric_ops, configured_cnn_mode, configured_cnn_rq_op 
        # are now expected to be part of metric_configs if a specific metric needs them,
        # or handled by the metric implementation itself (e.g. cnn_mode by AlignmentNetwork wrapper).
    )
    return layer_metric_scores 

@torch.no_grad()
def compute_metrics_for_layers(
    model: nn.Module,
    collected_data: Dict[str, Dict[str, torch.Tensor]], 
    metric_configs: List[Dict[str, Any]], 
    device: Union[str, torch.device],
) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    results: Dict[str, Dict[str, Union[torch.Tensor, float]]] = {}
    model.to(device); model.eval()
    modules_dict = {name: mod for name, mod in model.named_modules()}

    for layer_name, layer_data in collected_data.items():
        results[layer_name] = {}
        module = modules_dict.get(layer_name)
        if not module: continue
        
        l_inputs = layer_data.get("input"); l_outputs = layer_data.get("output")

        for m_config in metric_configs:
            metric_name = m_config["name"]
            scale_by_norm = m_config.get("scale_by_norm", False)
            specific_kwargs_for_metric = {k:v for k,v in m_config.items() if k not in ["name", "scale_by_norm"]}

            try:
                # Get the AlignmentMetric wrapper instance
                metric_instance = get_metric(name=metric_name, scale_by_norm=scale_by_norm)
                
                # Call its compute_per_node_scores method
                metric_val = metric_instance.compute_per_node_scores(
                    layer_inputs=l_inputs, 
                    layer_weights=module.weight.detach() if hasattr(module, 'weight') and module.weight is not None else None, 
                    layer_outputs=l_outputs,
                    device=device, 
                    **specific_kwargs_for_metric
                )
                if metric_val is not None:
                    results[layer_name][metric_name] = metric_val.item() if metric_val.numel()==1 else metric_val.cpu()
                else: results[layer_name][metric_name] = torch.tensor(float('nan'))
            except Exception as e:
                verbose_logging = specific_kwargs_for_metric.get("verbose", False)
                logger.error(f"Error computing metric '{metric_name}' for layer '{layer_name}': {e}", exc_info=verbose_logging)
                results[layer_name][metric_name] = torch.tensor(float('nan'))
    return results 