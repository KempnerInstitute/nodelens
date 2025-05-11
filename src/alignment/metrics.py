"""
Alignment metrics for neural network analysis.

This module provides metrics for measuring alignment between weight matrices 
and input activations, with various metrics to quantify the degree of alignment.
It supports node-wise scoring for pruning experiments and various utility
functions for alignment-based analysis.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import traceback

# Import core implementations from utils.metrics_utils
from alignment.utils.metrics_utils import (
    AlignmentMetricBase,
    RQMetric,
    MIMetric,
    WeightSimilarityMetric,
    NodeRedundancyMetric,
    AlignmentMetricsFactory as AlignmentMetrics,
    alignment
)

# If these utils are used by the moved function, ensure they are accessible
from alignment.utils.model_utils import _normalize_device, _ensure_model_on_device, _flatten_layer_weights_for_node, process_cnn_weights

logger = logging.getLogger(__name__)

class RankAlignmentMetric(AlignmentMetricBase):
    """
    Rank-based alignment metric.
    
    This metric assesses alignment based on the correlation of
    principal components of activations and weight matrices.
    """
    
    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure rank alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            
        Returns:
            Tensor containing rank alignment values
        """
        # Ensure inputs have at least 2 dimensions
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
            
        # Move weights to same device as inputs
        weights = weights.to(inputs.device)
        
        # Center the inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute SVD of inputs
        try:
            U, S, V = torch.linalg.svd(X, full_matrices=False)
            
            # Compute alignment between singular vectors and weight vectors
            alignment_scores = []
            for i in range(min(10, V.size(0))):  # Use top 10 singular vectors
                singular_vector = V[i, :]
                
                # Compute alignment with each weight vector
                for j in range(weights.size(0)):
                    weight_vector = weights[j, :]
                    
                    # Compute cosine similarity
                    cos_sim = torch.dot(singular_vector, weight_vector) / (
                        torch.norm(singular_vector) * torch.norm(weight_vector) + 1e-8
                    )
                    alignment_scores.append(cos_sim.abs().item())
            
            # Return mean alignment
            return torch.tensor(alignment_scores).mean()
            
        except Exception:
            # Fallback if SVD fails
            return torch.tensor(0.5, device=inputs.device)
    
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric."""
        return "RankAlignment"


class NullSpaceAlignmentMetric(AlignmentMetricBase):
    """
    Null space alignment metric.
    
    This metric assesses alignment based on the projection of
    weights onto the null space of the input covariance matrix.
    """
    
    @staticmethod
    def measure(inputs: torch.Tensor, weights: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Measure null space alignment between inputs and weights.
        
        Args:
            inputs: Input activations tensor
            weights: Weight tensor
            
        Returns:
            Tensor containing null space alignment values
        """
        # Ensure inputs have at least 2 dimensions
        if inputs.dim() < 2:
            inputs = inputs.unsqueeze(0)
            
        # Move weights to same device as inputs
        weights = weights.to(inputs.device)
        
        # Center the inputs
        X = inputs - inputs.mean(dim=0, keepdim=True)
        
        # Compute eigendecomposition of input covariance
        try:
            cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Calculate threshold for null space
            threshold = torch.max(eigenvalues) * 1e-3
            
            # Identify null space eigenvectors
            null_indices = (eigenvalues < threshold).nonzero().flatten()
            
            if null_indices.size(0) == 0:
                # No clear null space
                return torch.tensor(0.0, device=inputs.device)
                
            null_eigenvectors = eigenvectors[:, null_indices]
            
            # Compute projection of weights onto null space
            projection_scores = []
            for i in range(weights.size(0)):
                weight_vector = weights[i, :]
                
                # Project weight vector onto null space
                projection = torch.matmul(
                    torch.matmul(null_eigenvectors.t(), weight_vector),
                    null_eigenvectors
                )
                
                # Compute fraction of weight vector in null space
                projection_norm = torch.norm(projection)
                weight_norm = torch.norm(weight_vector)
                
                projection_scores.append((projection_norm / weight_norm).item())
                
            return torch.tensor(projection_scores).mean()
                
        except Exception:
            # Fallback if eigendecomposition fails
            return torch.tensor(0.0, device=inputs.device)
    
    @classmethod
    def get_name(cls) -> str:
        """Return the name of the metric."""
        return "NullSpaceAlignment"


# Register additional metrics with the registry
AlignmentMetrics.register("rank", RankAlignmentMetric)
AlignmentMetrics.register("null_space", NullSpaceAlignmentMetric)


class AlignmentMetric:
    """
    Interface for computing alignment metrics on neural networks.
    
    This class provides a standardized interface for alignment metrics,
    including support for node-wise scoring for pruning experiments.
    It serves as a higher-level wrapper around the lower-level AlignmentMetrics
    factory and individual metric implementations.
    """
    
    def __init__(self, name: str = "RQ", scale_by_norm: bool = False, force_cpu_for_large_metric_ops: bool = True):
        """
        Initialize alignment metric.
        
        Args:
            name: Name of the alignment metric to use
            scale_by_norm: Whether to scale the covariance or final measure by norm
            force_cpu_for_large_metric_ops: If True, offload known large tensor ops in metrics to CPU.
        """
        self.name = name
        self.scale_by_norm = scale_by_norm
        self.force_cpu_for_large_metric_ops = force_cpu_for_large_metric_ops
        
        # Validate that the metric exists
        self._validate_metric_name()
        
    def _validate_metric_name(self):
        """Validate that the metric name is valid."""
        normalized_name = self.name.lower()
        
        valid_metrics = [
            "rq", "mi", "rank", "null_space", 
            "weight_similarity", "redundancy", "class_separation"
        ]
        
        if normalized_name not in valid_metrics:
            raise ValueError(f"Unknown alignment metric: {self.name}. "
                           f"Valid options are: {', '.join(valid_metrics)}")
        
    def measure(self, 
                activations: Union[List[torch.Tensor], torch.Tensor], 
                targets: Optional[torch.Tensor] = None, 
                num_classes: int = 10) -> Union[List[float], torch.Tensor]:
        """
        Measure alignment between activations and targets.
        
        This method handles both:
        1. List of activations from different layers (returns a list of scores)
        2. Single activation tensor (returns a single tensor of scores)
        
        Args:
            activations: Layer activations (list for multiple layers or tensor for single layer)
            targets: Target labels (optional, used for class-based metrics)
            num_classes: Number of classes in the dataset
            
        Returns:
            List of alignment values per layer or tensor of values for a single layer
        """
        if isinstance(activations, list):
            # Multiple layers - compute for each and return list
            results = []
            
            for layer_activations in activations:
                # Handle class-specific metrics
                if self.name.lower() == "class_separation" and targets is not None:
                    result = self._compute_class_separation(
                        layer_activations, targets, num_classes
                    )
                else:
                    # For metrics that don't use targets, we pass a dummy weight matrix
                    # TODO: Implement proper handling for all metrics
                    if layer_activations.dim() > 2:
                        layer_activations = layer_activations.reshape(layer_activations.size(0), -1)
                    
                    # Use a dummy identity matrix as weights
                    dummy_weights = torch.eye(
                        layer_activations.size(1), 
                        device=layer_activations.device
                    )
                    
                    # Compute using the lower-level API
                    result = AlignmentMetrics.measure(
                        layer_activations, dummy_weights, 
                        method=self.name, force_cpu_for_large_metric_ops=self.force_cpu_for_large_metric_ops
                    ).mean().item()
                
                results.append(result)
                
            return results
        else:
            # Single layer - return tensor of scores
            return self.compute_per_node_scores(activations, None)
    
    def _compute_class_separation(self, 
                                activations: torch.Tensor, 
                                targets: torch.Tensor, 
                                num_classes: int) -> float:
        """
        Compute separation between class activations.
        
        Args:
            activations: Layer activations
            targets: Target labels
            num_classes: Number of classes in the dataset
            
        Returns:
            Class separation score
        """
        if activations.dim() > 2:
            activations = activations.reshape(activations.size(0), -1)
        
        # Compute class means
        class_means = []
        for c in range(num_classes):
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_mean = activations[class_mask].mean(dim=0)
                class_means.append(class_mean)
        
        if len(class_means) > 1:
            # Compute mean distance between class centers
            class_means = torch.stack(class_means)
            dists = torch.cdist(class_means, class_means)
            # Mask the diagonal
            mask = torch.ones_like(dists) - torch.eye(dists.size(0), device=dists.device)
            return (dists * mask).sum().item() / (mask.sum().item() + 1e-8)
        else:
            return 0.0
    
    def compute_per_node_scores(
        self,
        layer_input: torch.Tensor,
        layer_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        is_conv_layer: bool = False,
        cnn_mode_for_metric: Optional[str] = "unfold",
        cnn_rq_aggregation_op: Optional[str] = "mean"
    ) -> torch.Tensor:
        """
        Compute a per-node alignment score for each node's weight vector w_i.
        
        Args:
            layer_input: 2D tensor of shape (N, input_dim) with the data
            layer_weights: 2D tensor (#nodes, input_dim), or None to use identity
            device: PyTorch device to ensure everything is on the correct device
            is_conv_layer: True if the current layer_input/layer_weights are for a convolutional layer.
            cnn_mode_for_metric: Specifies how CNN activations/weights should be interpreted by the metric (e.g., "filter_patch_summary" for RQ).
            cnn_rq_aggregation_op: If using summary for CNN RQ, specifies op ('mean', 'max').
            
        Returns:
            A 1D tensor (#nodes,) of alignment scores
        """
        if device is None:
            device = layer_input.device if isinstance(layer_input, torch.Tensor) else torch.device('cpu')
        
        # Move input to device if needed
        if isinstance(layer_input, torch.Tensor):
            layer_input = layer_input.to(device)
        
        # Basic checks
        if layer_input.dim() != 2 and not (is_conv_layer and cnn_mode_for_metric == "filter_patch_summary"):
            # For filter_patch_summary, layer_input might be (Patches, Features_per_patch)
            # For other cases, it expects (N_samples, D_features)
            if layer_input.dim() > 2:
                layer_input = layer_input.reshape(layer_input.size(0), -1)
            elif layer_input.dim() == 1 and layer_input.size(0) == layer_weights.size(1): # Single sample case
                layer_input = layer_input.unsqueeze(0)
        
        # If weights are not provided, use identity matrix
        if layer_weights is None:
            # For RQ and similar metrics, weights must have compatible shape
            layer_weights = torch.eye(layer_input.size(1), device=device)
        else:
            # Move weights to device
            layer_weights = layer_weights.to(device)
        
        # Dispatch to appropriate method based on metric name
        if self.name.lower() == "rq":
            return self._compute_rq(layer_input, layer_weights, device, 
                                    is_conv_layer=is_conv_layer, 
                                    cnn_mode=cnn_mode_for_metric, 
                                    cnn_rq_aggregation_op=cnn_rq_aggregation_op)
        elif self.name.lower().startswith("mi"):
            return self._compute_mi(layer_input, layer_weights, device)
        elif self.name.lower() == "rank":
            metric_cls_instance = RankAlignmentMetric()
            return metric_cls_instance.measure(layer_input, layer_weights)
        elif self.name.lower() == "null_space":
            metric_cls_instance = NullSpaceAlignmentMetric()
            return metric_cls_instance.measure(layer_input, layer_weights)
        else:
            # Default to RQ as a fallback, passing default CNN flags
            return self._compute_rq(layer_input, layer_weights, device, 
                                    is_conv_layer=False, cnn_mode="unfold", cnn_rq_aggregation_op="mean")
    
    def _compute_rq(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device,
        is_conv_layer: bool = False,
        cnn_mode: Optional[str] = "unfold",
        cnn_rq_aggregation_op: Optional[str] = "mean"
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient per node.
        
        Args:
            X: Input data (N, input_dim)
            W: Weight matrix (num_nodes, input_dim)
            device: Compute device
            is_conv_layer: True if the current layer_input/layer_weights are for a convolutional layer.
            cnn_mode: Specifies how CNN activations/weights should be interpreted by the metric (e.g., "filter_patch_summary" for RQ).
            cnn_rq_aggregation_op: If using summary for CNN RQ, specifies op ('mean', 'max').
            
        Returns:
            Per-node RQ scores (num_nodes,)
        """
        N = X.size(0)
        if N < 2:
            return torch.zeros(W.size(0), device=device)

        if is_conv_layer and cnn_mode == "filter_patch_summary":
            # X is (num_total_patches_for_batch, filter_input_dims)
            # W is (out_channels, filter_input_dims)
            # logger.info(f"RQ using filter_patch_summary for Conv. X: {X.shape}, W: {W.shape}, op: {cnn_rq_aggregation_op}")
            num_filters = W.shape[0]
            filter_scores = torch.zeros(num_filters, device=device)
            eps = 1e-12

            # Use self.force_cpu_for_large_metric_ops for its internal heuristic if needed
            perform_on_cpu_patchwise = False
            if self.force_cpu_for_large_metric_ops and X.is_cuda and X.numel() > 10_000_000: 
                perform_on_cpu_patchwise = True
            
            X_op = X.cpu() if perform_on_cpu_patchwise else X
            W_op = W.cpu() if perform_on_cpu_patchwise else W

            for j in range(num_filters):
                w_j = W_op[j, :] 
                w_j_norm_sq = (w_j * w_j).sum()
                if w_j_norm_sq < eps:
                    filter_scores[j] = 0.0
                    continue

                # Projections of all patches onto this filter: p_k^T w_j  (or w_j^T p_k)
                # X_op is (N_patches, D_filter_in), w_j is (D_filter_in)
                # Result proj_val_for_filter_j is (N_patches)
                proj_val_for_filter_j = X_op @ w_j.T 
                
                # Squared cosine similarity: (w_j^T p_k)^2 / (||w_j||^2 ||p_k||^2)
                # Equivalent to (proj_val_for_filter_j)^2 / (w_j_norm_sq * ||p_k||^2)
                patch_norms_sq = (X_op * X_op).sum(dim=1)
                
                # Avoid division by zero for patch_norms_sq
                valid_patches_mask = patch_norms_sq > eps
                if not valid_patches_mask.any():
                    filter_scores[j] = 0.0
                    continue

                # Select only valid patches
                valid_proj_val = proj_val_for_filter_j[valid_patches_mask]
                valid_patch_norms_sq = patch_norms_sq[valid_patches_mask]

                patch_level_rqs = (valid_proj_val**2) / (w_j_norm_sq * valid_patch_norms_sq + eps)
                
                if patch_level_rqs.numel() == 0:
                    filter_scores[j] = 0.0
                    continue

                if cnn_rq_aggregation_op == "mean":
                    agg_score = patch_level_rqs.mean()
                elif cnn_rq_aggregation_op == "max":
                    agg_score = patch_level_rqs.max()
                elif cnn_rq_aggregation_op == "var":
                    agg_score = patch_level_rqs.var()
                elif cnn_rq_aggregation_op == "sum":
                    agg_score = patch_level_rqs.sum()
                else: # Default to mean
                    agg_score = patch_level_rqs.mean()
                filter_scores[j] = agg_score
            
            return filter_scores.to(device) # Ensure result is on original device

        # Existing logic for Linear layers or cnn_mode='unfold' (global covariance on unfolded patches)
        perform_on_cpu_cov = False
        if self.force_cpu_for_large_metric_ops and X.is_cuda and ( (X.shape[0] > 500_000 and X.shape[1] > 100) or (X.numel() > 20_000_000) ):
            logger.info(f"RQ: Large input tensor X ({X.shape}) on CUDA with force_cpu=True. Offloading centering and cov to CPU.")
            perform_on_cpu_cov = True

        if perform_on_cpu_cov:
            # ... (CPU offload logic for covariance as before) ...
            X_cpu = X.cpu(); W_cpu = W.cpu()
            mean_X_cpu = X_cpu.mean(dim=0, keepdim=True); X_centered_cpu = X_cpu - mean_X_cpu
            cov_x = (X_centered_cpu.t() @ X_centered_cpu) / (N - 1)
            # cov_x = cov_x.to(device) # Keep cov_x on CPU if W_op is CPU
            W_op_cov = W_cpu
        else:
            mean_X = X.mean(dim=0, keepdim=True); X_centered = X - mean_X 
            cov_x = (X_centered.t() @ X_centered) / (N - 1)
            W_op_cov = W

        if self.scale_by_norm:
            norm_val = cov_x.norm(p=2)
            if norm_val > 1e-12: cov_x = cov_x / norm_val
        
        # Ensure cov_x is on the same device as W_op_cov for matmul
        cov_x = cov_x.to(W_op_cov.device)

        w_norm_sq_cov = (W_op_cov * W_op_cov).sum(dim=1)  
        wCov_cov = W_op_cov @ cov_x               
        numerator_cov = (wCov_cov * W_op_cov).sum(dim=1)  
        eps_cov = 1e-12
        rq_scores = numerator_cov / (w_norm_sq_cov + eps_cov)
        
        if perform_on_cpu_cov:
            rq_scores = rq_scores.to(device)
        return rq_scores

    def _compute_mi(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute mutual information-based scores.
        
        Args:
            X: Input data (N, input_dim)
            W: Weight matrix (num_nodes, input_dim)
            device: Compute device
            
        Returns:
            Per-node MI scores (num_nodes,)
        """
        # Delegate to factory, but factory.measure would need the flag too if MI has large ops.
        # For now, MIMetric.measure itself needs to be refactored to use this flag.
        # This means `AlignmentMetrics.measure` needs to pass it, or MIMetric gets it from its own init.
        # Let's assume for now that MIMetric will be refactored similarly if direct CPU offload needed.
        # The call to AlignmentMetrics.measure needs to be able to pass this flag down.
        # This is getting complex. A simpler way for now: _compute_mi will call the MIMetric static method
        # and pass the flag it receives from compute_per_node_scores.
        # compute_per_node_scores will get it from self.force_cpu_for_large_metric_ops
        return AlignmentMetrics.measure(X, W, method="MI", force_cpu_for_large_metric_ops=self.force_cpu_for_large_metric_ops)


# Function to get an alignment metric instance
def get_metric(name: str, scale_by_norm: bool = False, force_cpu_for_large_metric_ops: bool = True) -> AlignmentMetric:
    """
    Get alignment metric by name.
    
    Args:
        name: Name of the metric ('RQ', 'MI', 'rank', 'null_space', etc.)
        scale_by_norm: Whether to scale the covariance or final measure by norm
        force_cpu_for_large_metric_ops: Global flag for CPU offload in metrics.
        
    Returns:
        AlignmentMetric instance
    """
    return AlignmentMetric(name=name, scale_by_norm=scale_by_norm, force_cpu_for_large_metric_ops=force_cpu_for_large_metric_ops)


# Register a new metric with the registry
def register_metric(name: str, metric_class: AlignmentMetricBase) -> None:
    """
    Register a new alignment metric.
    
    Args:
        name: Name to register the metric under
        metric_class: Class implementing the AlignmentMetricBase interface
    """
    AlignmentMetrics.register(name, metric_class)


# Legacy function for backward compatibility
def alignment(inputs: torch.Tensor, weights: torch.Tensor, 
             method: str = "RQ", relative: bool = True, **kwargs) -> torch.Tensor:
    """
    Compute alignment between inputs and weights.
    
    Args:
        inputs: Input activations tensor
        weights: Weight tensor
        method: Alignment method
        relative: Whether to use relative alignment (for RQ)
        
    Returns:
        Tensor containing alignment values
    """
    kwargs["relative"] = relative
    return AlignmentMetrics.measure(inputs, weights, method=method, **kwargs) 


# Function to be moved from dropout.py - NOW THE CANONICAL VERSION
def compute_all_node_scores(
    model: nn.Module,
    metric_configs: List[Dict[str, Any]], 
    device: torch.device,
    data_loader: DataLoader,
    num_batches: Optional[int] = 5,
    debug_mode: bool = False,
    configured_cnn_mode: Optional[str] = "unfold", 
    configured_cnn_rq_op: Optional[str] = "mean",
    force_cpu_for_large_metric_ops: bool = True
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Computes per-node scores for specified alignment layers for multiple metrics.
    Activations are processed per batch for each metric to manage memory, 
    and then scores are averaged across batches.
    """
    if not hasattr(model, "alignment_layers") or not hasattr(model, "alignment_names"):
        raise ValueError("Model must define .alignment_layers and .alignment_names attributes (typically an AlignmentNetwork).")
    if not isinstance(metric_configs, list) or not all(isinstance(mc, dict) and "name" in mc for mc in metric_configs):
        raise ValueError("metric_configs must be a list of dictionaries, each with a 'name' key.")
    if not metric_configs: 
        logger.warning("compute_all_node_scores called with empty metric_configs. Returning empty dict.")
        return {}

    normalized_device = _normalize_device(device)
    model.eval()
    _ensure_model_on_device(model, normalized_device)

    if debug_mode:
        logger.info(f"Computing node scores for {len(metric_configs)} metrics, using {num_batches} batches, for model with {len(model.alignment_layers)} alignment layers on device {normalized_device}")
        for i, (layer_mod, layer_name) in enumerate(zip(model.alignment_layers, model.alignment_names)):
            if hasattr(layer_mod, 'weight') and layer_mod.weight is not None:
                 logger.info(f"Layer {i}: {layer_name} - {type(layer_mod).__name__} - Shape: {layer_mod.weight.shape}")
            else:
                 logger.info(f"Layer {i}: {layer_name} - {type(layer_mod).__name__} - No weight attribute or weight is None")

    if not hasattr(model, "hidden"):
        model.hidden = {}
    else: 
        model.hidden.clear()

    # Determine effective number of batches to process
    effective_num_batches = num_batches
    if effective_num_batches is None or effective_num_batches < 0: # Treat None or <0 as all batches
        effective_num_batches = len(data_loader)
        if debug_mode:
            logger.info(f"num_batches is None or <0, using all {effective_num_batches} batches from data_loader.")
    elif effective_num_batches == 0:
        logger.warning("num_batches is 0, no batches will be processed for activation hooking.")
        # Populate with zeros based on expected structure if no batches ran but layers exist
        # (This logic is duplicated below, can be refactored if needed)
        all_scores_per_layer_all_metrics_empty: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx_empty, layer_mod_scores_empty in enumerate(model.alignment_layers):
            metrics_for_this_layer_empty = {}
            node_count_empty = layer_mod_scores_empty.weight.shape[0] if hasattr(layer_mod_scores_empty, 'weight') and layer_mod_scores_empty.weight is not None else 0
            for m_config_empty in metric_configs:
                metrics_for_this_layer_empty[m_config_empty["name"]] = torch.zeros(node_count_empty, device=normalized_device)
            all_scores_per_layer_all_metrics_empty[layer_idx_empty] = metrics_for_this_layer_empty
        return all_scores_per_layer_all_metrics_empty

    processed_batch_count = 0
    hooks = []
    try:
        alignment_layer_modules = model.alignment_layers # For type checking in hook
        def get_activation_hook(layer_idx_for_hook):
            def hook(module, layer_input, layer_output):
                x = layer_input[0].detach()
                layer_name_for_storage = model.alignment_names[layer_idx_for_hook]
                if not model.hidden.get(layer_name_for_storage):
                    model.hidden[layer_name_for_storage] = []
                model.hidden[layer_name_for_storage].append(x) # Store raw input tensor for this batch
                if debug_mode and len(model.hidden[layer_name_for_storage]) == 1 and processed_batch_count < 1: # Log shape only for first batch fully processed by hooks for this layer
                    logger.info(f"Layer {layer_name_for_storage} (hook for batch {processed_batch_count+1}) input shape: {x.shape}")
            return hook

        for i, layer_mod_hook in enumerate(model.alignment_layers):
            hooks.append(layer_mod_hook.register_forward_hook(get_activation_hook(i)))

        # MODIFIED: Use effective_num_batches for tqdm total and loop break condition
        batch_iter = tqdm(data_loader, desc="Processing batches for activation hooking", total=effective_num_batches, disable=not debug_mode, leave=False)
        for inputs, _targets in batch_iter:
            if processed_batch_count >= effective_num_batches:
                break # Ensure we don't process more than intended
            inputs = inputs.to(normalized_device)
            for layer_name_clear in model.alignment_names:
                 model.hidden[layer_name_clear] = [] 
            model(inputs) 
            processed_batch_count += 1
            # No need for a second check of processed_batch_count >= effective_num_batches here if tqdm handles total correctly
    finally:
        for h in hooks:
            h.remove()
        hooks.clear()

    all_scores_per_layer_all_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
    if processed_batch_count == 0:
        logger.warning("No batches were actually processed for activation hooking (effective_num_batches might have been 0 or loader empty). Returning empty scores.")
        # ... (populate with zeros as above) ...
        for layer_idx_empty, layer_mod_scores_empty in enumerate(model.alignment_layers):
            metrics_for_this_layer_empty = {}
            node_count_empty = layer_mod_scores_empty.weight.shape[0] if hasattr(layer_mod_scores_empty, 'weight') and layer_mod_scores_empty.weight is not None else 0
            for m_config_empty in metric_configs:
                metrics_for_this_layer_empty[m_config_empty["name"]] = torch.zeros(node_count_empty, device=normalized_device)
            all_scores_per_layer_all_metrics[layer_idx_empty] = metrics_for_this_layer_empty
        return all_scores_per_layer_all_metrics

    layer_iter = enumerate(model.alignment_layers)
    if debug_mode:
        layer_iter = tqdm(list(layer_iter), desc="Computing layer scores (batch-wise avg) for multiple metrics")

    for layer_idx, layer_mod_scores in layer_iter:
        layer_name_scores = model.alignment_names[layer_idx]
        metrics_for_this_layer: Dict[str, torch.Tensor] = {}
        current_layer_module_for_processing = model.alignment_layers[layer_idx]
        is_conv_type_processing = isinstance(current_layer_module_for_processing, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))

        if layer_name_scores not in model.hidden or not model.hidden[layer_name_scores]:
            # ... (handle missing data as before) ...
            node_count = layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0
            for m_config in metric_configs:
                metrics_for_this_layer[m_config["name"]] = torch.zeros(node_count, device=normalized_device)
            all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
            if layer_name_scores in model.hidden: model.hidden[layer_name_scores] = None 
            continue
        
        w_flat, layer_metadata = process_cnn_weights(model, layer_idx, pruning_strategy="structure-aware")
        if w_flat is None:
            logger.error(f"Failed to get weights for layer {layer_name_scores}. Skipping metric computation for this layer.")
            node_count = layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0
            for m_config in metric_configs:
                metrics_for_this_layer[m_config["name"]] = torch.zeros(node_count, device=normalized_device)
            all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
            if layer_name_scores in model.hidden: model.hidden[layer_name_scores] = None
            continue

        activation_batches_for_layer = model.hidden[layer_name_scores] # List of tensors, each from a batch

        for m_config in metric_configs:
            metric_name = m_config["name"]
            scale_by_norm_for_metric = m_config.get("scale_by_norm", False)
            current_metric_instance = get_metric(
                name=metric_name, 
                scale_by_norm=scale_by_norm_for_metric,
                force_cpu_for_large_metric_ops=force_cpu_for_large_metric_ops
            )
            
            per_batch_node_scores_for_metric = []

            for batch_idx_act, act_tensor_one_batch in enumerate(activation_batches_for_layer):
                X_processed_one_batch: Optional[torch.Tensor] = None
                if is_conv_type_processing:
                    # ... (unfolding logic as implemented in previous step, ensure kernel_s etc. are defined)
                    kernel_s = current_layer_module_for_processing.kernel_size
                    stride_s = current_layer_module_for_processing.stride
                    padding_s = current_layer_module_for_processing.padding
                    dilation_s = current_layer_module_for_processing.dilation
                    # act_tensor_one_batch should be NCHW or NCL from the hook
                    if isinstance(current_layer_module_for_processing, (nn.Conv2d, nn.ConvTranspose2d)) and act_tensor_one_batch.dim() == 4:
                        try:
                            unfolded_act = F.unfold(act_tensor_one_batch, kernel_size=kernel_s, dilation=dilation_s, padding=padding_s, stride=stride_s)
                            unfolded_act = unfolded_act.transpose(1, 2).contiguous().view(-1, unfolded_act.size(1))
                            X_processed_one_batch = unfolded_act
                        except Exception as e_unfold:
                            logger.error(f"Batch {batch_idx_act}: Error unfolding for Conv2D layer {layer_name_scores}: {e_unfold}. Input: {act_tensor_one_batch.shape}")
                    elif isinstance(current_layer_module_for_processing, (nn.Conv1d, nn.ConvTranspose1d)) and act_tensor_one_batch.dim() == 3:
                        try:
                            # Ensure kernel_s, dilation_s etc. are 1-element tuples for 1D conv if they are ints
                            k_s = (kernel_s[0],1) if isinstance(kernel_s, tuple) else (kernel_s,1)
                            d_s = (dilation_s[0],1) if isinstance(dilation_s, tuple) else (dilation_s,1)
                            p_s = (padding_s[0],0) if isinstance(padding_s, tuple) else (padding_s,0)
                            st_s = (stride_s[0],1) if isinstance(stride_s, tuple) else (stride_s,1)
                            unfolded_act = F.unfold(act_tensor_one_batch.unsqueeze(3), kernel_size=k_s, dilation=d_s, padding=p_s, stride=st_s)
                            unfolded_act = unfolded_act.transpose(1, 2).contiguous().view(-1, unfolded_act.size(1))
                            X_processed_one_batch = unfolded_act
                        except Exception as e_unfold:
                            logger.error(f"Batch {batch_idx_act}: Error unfolding for Conv1D layer {layer_name_scores}: {e_unfold}. Input: {act_tensor_one_batch.shape}")
                    else:
                        logger.warning(f"Batch {batch_idx_act}: Skipping unfold for {layer_name_scores} (type {type(current_layer_module_for_processing).__name__}, dim {act_tensor_one_batch.dim()}). Using flattened.")
                        X_processed_one_batch = act_tensor_one_batch.reshape(act_tensor_one_batch.size(0), -1)
                else: # Linear layer
                    X_processed_one_batch = act_tensor_one_batch.reshape(act_tensor_one_batch.size(0), -1)

                if X_processed_one_batch is not None and X_processed_one_batch.numel() > 0:
                    try:
                        node_scores_this_batch = current_metric_instance.compute_per_node_scores(
                            X_processed_one_batch, 
                            w_flat, 
                            device=normalized_device, 
                            is_conv_layer=is_conv_type_processing, 
                            cnn_mode_for_metric=configured_cnn_mode,
                            cnn_rq_aggregation_op=configured_cnn_rq_op
                        )
                        if node_scores_this_batch is not None:
                             per_batch_node_scores_for_metric.append(node_scores_this_batch)
                    except Exception as e_comp_batch:
                        logger.error(f"Batch {batch_idx_act}: Error computing scores for metric '{metric_name}' on layer {layer_name_scores}: {e_comp_batch}", exc_info=debug_mode)
                # else: logger.warning if X_processed_one_batch is None or empty for a batch
            
            # Average scores across batches for this metric and layer
            if per_batch_node_scores_for_metric:
                final_scores_for_metric = torch.stack(per_batch_node_scores_for_metric).mean(dim=0)
                if debug_mode:
                    logger.info(f"  Metric '{metric_name}': Layer {layer_name_scores} avg score stats: min={torch.min(final_scores_for_metric).item():.4f}, max={torch.max(final_scores_for_metric).item():.4f}, mean={torch.mean(final_scores_for_metric).item():.4f}, std={torch.std(final_scores_for_metric).item():.4f}")
                metrics_for_this_layer[metric_name] = final_scores_for_metric.detach()
            else:
                logger.warning(f"No batch scores computed for metric '{metric_name}' on layer {layer_name_scores}. Setting to zeros.")
                node_count_fallback = w_flat.shape[0] if w_flat is not None else (layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0)
                metrics_for_this_layer[metric_name] = torch.zeros(node_count_fallback, device=normalized_device)
        
        all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
        model.hidden[layer_name_scores] = None 
    
    model.hidden.clear() 
    return all_scores_per_layer_all_metrics 