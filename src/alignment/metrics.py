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
    
    def __init__(self, name: str = "RQ", scale_by_norm: bool = False):
        """
        Initialize alignment metric.
        
        Args:
            name: Name of the alignment metric to use
            scale_by_norm: Whether to scale the covariance or final measure by norm
        """
        self.name = name
        self.scale_by_norm = scale_by_norm
        
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
                        method=self.name
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
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Compute a per-node alignment score for each node's weight vector w_i.
        
        Args:
            layer_input: 2D tensor of shape (N, input_dim) with the data
            layer_weights: 2D tensor (#nodes, input_dim), or None to use identity
            device: PyTorch device to ensure everything is on the correct device
            
        Returns:
            A 1D tensor (#nodes,) of alignment scores
        """
        if device is None:
            device = layer_input.device if isinstance(layer_input, torch.Tensor) else torch.device('cpu')
        
        # Move input to device if needed
        if isinstance(layer_input, torch.Tensor):
            layer_input = layer_input.to(device)
        
        # Basic checks
        if layer_input.dim() != 2:
            # For CNN, flatten the input
            orig_shape = layer_input.shape
            layer_input = layer_input.reshape(orig_shape[0], -1)
        
        # If weights are not provided, use identity matrix
        if layer_weights is None:
            # For RQ and similar metrics, weights must have compatible shape
            layer_weights = torch.eye(layer_input.size(1), device=device)
        else:
            # Move weights to device
            layer_weights = layer_weights.to(device)
        
        # Dispatch to appropriate method based on metric name
        if self.name.lower() == "rq":
            return self._compute_rq(layer_input, layer_weights, device)
        elif self.name.lower().startswith("mi"):
            return self._compute_mi(layer_input, layer_weights, device)
        elif self.name.lower() == "rank":
            metric = RankAlignmentMetric()
            return metric.measure(layer_input, layer_weights)
        elif self.name.lower() == "null_space":
            metric = NullSpaceAlignmentMetric()
            return metric.measure(layer_input, layer_weights)
        else:
            # Default to RQ as a fallback
            return self._compute_rq(layer_input, layer_weights, device)
    
    def _compute_rq(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient per node.
        
        Args:
            X: Input data (N, input_dim)
            W: Weight matrix (num_nodes, input_dim)
            device: Compute device
            
        Returns:
            Per-node RQ scores (num_nodes,)
        """
        # Center the input
        N = X.size(0)
        if N < 2:
            # fallback if dataset is extremely small
            return torch.zeros(W.size(0), device=device)

        X_centered = X - X.mean(dim=0, keepdim=True)
        # sample covariance => (input_dim x input_dim)
        cov_x = (X_centered.t() @ X_centered) / (N - 1)

        # Possibly scale by norm
        if self.scale_by_norm:
            norm_val = cov_x.norm(p=2)
            if norm_val > 1e-12:
                cov_x = cov_x / norm_val

        # W shape (#nodes, input_dim)
        # Let's do a row-by-row RQ:
        w_norm_sq = (W * W).sum(dim=1)  # (#nodes,)
        wCov = W @ cov_x               # (#nodes, input_dim)
        numerator = (wCov * W).sum(dim=1)  # dot with W row wise => (#nodes,)

        eps = 1e-12
        rq_scores = numerator / (w_norm_sq + eps)
        
        # Scale by sqrt(d) if using relative RQ
        if False:  # TODO: Add a relative parameter
            d = W.size(1)
            rq_scores = rq_scores * np.sqrt(d)
            
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
        # For now, we delegate to the AlignmentMetrics class
        return AlignmentMetrics.measure(X, W, method="MI")


# Function to get an alignment metric instance
def get_metric(name: str, scale_by_norm: bool = False) -> AlignmentMetric:
    """
    Get alignment metric by name.
    
    Args:
        name: Name of the metric ('RQ', 'MI', 'rank', 'null_space', etc.)
        scale_by_norm: Whether to scale the covariance or final measure by norm
        
    Returns:
        AlignmentMetric instance
    """
    return AlignmentMetric(name=name, scale_by_norm=scale_by_norm)


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
    num_batches: int = 5,
    debug_mode: bool = False,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Computes per-node scores for specified alignment layers for multiple metrics.
    This function orchestrates activation hooking and per-layer metric computation.
    It no longer has logic to exclude classification layers from score computation itself;
    that responsibility lies with the calling application (e.g., pruning).

    Args:
        model: The model (expected to have .alignment_layers and .alignment_names attributes, 
               typically an AlignmentNetwork instance).
        metric_configs: List of metric configurations (dicts). Each dict should have at least a "name"
                        key, and can optionally have "scale_by_norm" (defaults to False).
        device: The torch.device to run computations on.
        data_loader: DataLoader for providing input data to the model for activation hooking.
        num_batches: Number of batches from data_loader to use for collecting activations.
        debug_mode: If True, enables verbose logging.

    Returns:
        A dictionary mapping layer_idx to another dictionary, which maps metric_name 
        to a 1D tensor of per-node scores for that layer and metric.
    """
    if not hasattr(model, "alignment_layers") or not hasattr(model, "alignment_names"):
        raise ValueError("Model must define .alignment_layers and .alignment_names attributes (typically an AlignmentNetwork).")
    # MODIFIED: Validation for metric_configs
    if not isinstance(metric_configs, list) or not all(isinstance(mc, dict) and "name" in mc for mc in metric_configs):
        raise ValueError("metric_configs must be a list of dictionaries, each with a 'name' key.")
    if not metric_configs: # No metrics to compute
        logger.warning("compute_all_node_scores called with empty metric_configs. Returning empty dict.")
        return {}

    normalized_device = _normalize_device(device)
    model.eval()
    _ensure_model_on_device(model, normalized_device)

    if debug_mode:
        logger.info(f"Computing node scores for model with {len(model.alignment_layers)} alignment layers on device {normalized_device}")
        for i, (layer_mod, layer_name) in enumerate(zip(model.alignment_layers, model.alignment_names)):
            if hasattr(layer_mod, 'weight') and layer_mod.weight is not None:
                 logger.info(f"Layer {i}: {layer_name} - {type(layer_mod).__name__} - Shape: {layer_mod.weight.shape}")
            else:
                 logger.info(f"Layer {i}: {layer_name} - {type(layer_mod).__name__} - No weight attribute or weight is None")

    if not hasattr(model, "hidden"):
        model.hidden = {}
    else: # Clear any stale hidden states from previous calls or other uses
        model.hidden.clear()

    batch_count = 0
    hooks = []
    try:
        # Store original layer modules for type checking later
        alignment_layer_modules = model.alignment_layers

        def get_activation_hook(layer_idx_for_hook): # Pass layer_idx to know the layer type
            def hook(module, layer_input, layer_output):
                x = layer_input[0].detach() # Get input to the module
                current_layer_module = alignment_layer_modules[layer_idx_for_hook]
                is_conv_type = isinstance(current_layer_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
                
                # For non-Conv layers, flatten to 2D. For Conv layers, keep original shape for later unfolding.
                if not is_conv_type and x.dim() > 2:
                    x = x.reshape(x.size(0), -1) # Flatten N, C, H, W -> N, C*H*W or N, L, C -> N, L*C
                elif is_conv_type and x.dim() <=2 and x.dim() > 0: # e.g. if a conv layer gets a flattened input by mistake from previous layer
                    # This case is tricky, ideally input to conv is already 4D/5D.
                    # If it's 2D (N, Features), and we know original C,H,W, we could reshape.
                    # For now, if it's already 2D for a conv layer, we might log a warning or pass it as is, 
                    # assuming it's an edge case or a model structure where this is intended.
                    pass # Keep as is, unfolding might fail or work depending on layout
                
                layer_name_for_storage = model.alignment_names[layer_idx_for_hook]
                current_val = model.hidden.get(layer_name_for_storage)
                if not isinstance(current_val, list):
                    if current_val is not None and debug_mode:
                        logger.warning(f"Hook for layer '{layer_name_for_storage}': model.hidden was {type(current_val)}, re-initializing to list.")
                    model.hidden[layer_name_for_storage] = []
                
                try:
                    model.hidden[layer_name_for_storage].append(x) # Store detached x
                except AttributeError as e_hook_append:
                    # ... (enhanced error handling as before) ...
                    logger.error(
                        f"CRITICAL HOOK AttributeError for layer '{layer_name_for_storage}': model.hidden['{layer_name_for_storage}'] is type {type(model.hidden.get(layer_name_for_storage))}. Error: {e_hook_append}"
                    )
                    logger.error(f"Model hidden dict right before error: {model.hidden}")
                    if model.hidden.get(layer_name_for_storage) is None: 
                        model.hidden[layer_name_for_storage] = []
                        model.hidden[layer_name_for_storage].append(x)
                    else:
                        raise 

                if debug_mode and len(model.hidden[layer_name_for_storage]) == 1:
                    logger.info(f"Layer {layer_name_for_storage} (type: {type(current_layer_module).__name__}) input shape: {x.shape}, stored in hidden.")
            return hook

        for i, layer_mod_hook in enumerate(model.alignment_layers):
            hooks.append(layer_mod_hook.register_forward_hook(get_activation_hook(i))) # Pass layer index i

        batch_iter = tqdm(data_loader, desc="Processing batches for metrics", disable=not debug_mode, leave=False)
        for inputs, _targets in batch_iter:
            inputs = inputs.to(normalized_device)
            model(inputs) # This triggers hooks, populating model.hidden
            batch_count += 1
            if batch_count >= num_batches:
                break
    finally:
        for h in hooks:
            h.remove()
        hooks.clear() # Clear the list of hooks

    # MODIFIED: Outer dictionary stores results per layer
    all_scores_per_layer_all_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
    layer_iter = enumerate(model.alignment_layers)
    if debug_mode:
        layer_iter = tqdm(list(layer_iter), desc="Computing layer scores from activations for multiple metrics")

    for layer_idx, layer_mod_scores in layer_iter:
        layer_name_scores = model.alignment_names[layer_idx]
        metrics_for_this_layer: Dict[str, torch.Tensor] = {}
        current_layer_module_for_processing = model.alignment_layers[layer_idx] # Get the actual layer module

        if layer_name_scores not in model.hidden or not model.hidden[layer_name_scores]:
            if debug_mode:
                logger.warning(f"No/empty hooking data for layer '{layer_name_scores}'. Scores set to zero for all metrics.")
            # Populate with zero tensors for all requested metrics for this layer
            node_count = layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0
            for m_config in metric_configs:
                metrics_for_this_layer[m_config["name"]] = torch.zeros(node_count, device=normalized_device)
            all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
            if layer_name_scores in model.hidden: model.hidden[layer_name_scores] = None
            continue
        
        # Get weights (already good via process_cnn_weights)
        w_flat, layer_metadata = process_cnn_weights(model, layer_idx, pruning_strategy="structure-aware")
        
        # MODIFIED: Process activations based on layer type
        activation_list = model.hidden[layer_name_scores]
        is_conv_type_processing = isinstance(current_layer_module_for_processing, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                                                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
        
        X_acts_final_for_metric: Optional[torch.Tensor] = None

        if is_conv_type_processing:
            processed_batch_activations = []
            # Get unfold params from the layer itself
            # These need to be present on the layer module (e.g. layer_mod_scores)
            # Make sure kernel_size, stride etc. are correctly obtained
            kernel_s = current_layer_module_for_processing.kernel_size
            stride_s = current_layer_module_for_processing.stride
            padding_s = current_layer_module_for_processing.padding
            dilation_s = current_layer_module_for_processing.dilation

            for act_tensor_batch in activation_list:
                # act_tensor_batch should be in its original 4D/5D form here if hook was changed
                if act_tensor_batch.dim() <= 2 and act_tensor_batch.numel() > 0 : # If it got flattened somehow, try to unflatten if possible (needs original shape info - hard)
                    logger.warning(f"Conv layer {layer_name_scores} received 2D input for unfolding. This might lead to incorrect metrics if original shape isn\'t restored.")
                    # This path is problematic without original shape info. For now, we proceed, but it's a known issue.
                
                # Unfold only works for specific dims (e.g. 4D for Conv2D -> unfold)
                # Ensure act_tensor_batch has the right dimensions for unfold based on layer type
                # Example for Conv2d, input is (N, C, H, W)
                if isinstance(current_layer_module_for_processing, (nn.Conv2d, nn.ConvTranspose2d)) and act_tensor_batch.dim() == 4:
                    try:
                        unfolded_act = F.unfold(act_tensor_batch, kernel_size=kernel_s, dilation=dilation_s, padding=padding_s, stride=stride_s)
                        # unfolded_act shape: (N, C_in*kernel_h*kernel_w, L_out), where L_out is num_patches
                        # We need (N*L_out, C_in*kernel_h*kernel_w) to match flattened filter (out_channels, C_in*kernel_h*kernel_w)
                        unfolded_act = unfolded_act.transpose(1, 2).contiguous() # (N, L_out, C_in*KH*KW)
                        unfolded_act = unfolded_act.view(-1, unfolded_act.size(2)) # (N*L_out, C_in*KH*KW)
                        processed_batch_activations.append(unfolded_act)
                    except Exception as e_unfold:
                        logger.error(f"Error unfolding activations for Conv layer {layer_name_scores}: {e_unfold}. Input shape: {act_tensor_batch.shape}. Skipping batch.")
                        continue # Skip this batch if unfolding fails
                elif isinstance(current_layer_module_for_processing, (nn.Conv1d, nn.ConvTranspose1d)) and act_tensor_batch.dim() == 3:
                     # Similar logic for Conv1d: input (N,C,L), unfold, then (N*L_patches, C*KernelL)
                    try:
                        unfolded_act = F.unfold(act_tensor_batch.unsqueeze(3), kernel_size=(kernel_s[0],1), dilation=(dilation_s[0],1), padding=(padding_s[0],0), stride=(stride_s[0],1)) # Use dummy H dim
                        unfolded_act = unfolded_act.transpose(1, 2).contiguous()
                        unfolded_act = unfolded_act.view(-1, unfolded_act.size(2))
                        processed_batch_activations.append(unfolded_act)
                    except Exception as e_unfold:
                        logger.error(f"Error unfolding activations for Conv1D layer {layer_name_scores}: {e_unfold}. Input shape: {act_tensor_batch.shape}. Skipping batch.")
                        continue
                # Add Conv3d if necessary
                else: # Not a Conv layer type that we have specific unfold logic for, or dim mismatch
                    logger.warning(f"Skipping unfold for layer {layer_name_scores} (type {type(current_layer_module_for_processing).__name__}, input dim {act_tensor_batch.dim()}). Using flattened activations.")
                    processed_batch_activations.append(act_tensor_batch.reshape(act_tensor_batch.size(0), -1))
            
            if processed_batch_activations:
                X_acts_final_for_metric = torch.cat(processed_batch_activations, dim=0)
            else:
                logger.warning(f"No activations could be processed/unfolded for Conv layer {layer_name_scores}. Metric scores will be zeros.")
        else: # Linear or other non-Conv layers
            # Flatten each tensor in the list and then concatenate
            X_acts_final_for_metric = torch.cat([act_b.reshape(act_b.size(0), -1) for act_b in activation_list], dim=0)

        if X_acts_final_for_metric is None or X_acts_final_for_metric.numel() == 0:
            logger.warning(f"Final activations X_acts_final_for_metric is None or empty for layer {layer_name_scores}. Setting metric scores to zero.")
            # Populate with zero tensors for all requested metrics for this layer
            node_count = w_flat.shape[0] if w_flat is not None else (layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0)
            for m_config in metric_configs:
                metrics_for_this_layer[m_config["name"]] = torch.zeros(node_count, device=normalized_device)
            all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
            model.hidden[layer_name_scores] = None 
            continue # Move to next layer
        
        if debug_mode:
            logger.info(f"Layer {layer_name_scores}: Final Processed Activations X_acts_final_for_metric shape {X_acts_final_for_metric.shape}, Weights w_flat shape {w_flat.shape if w_flat is not None else 'None'}")
        
        for m_config in metric_configs:
            metric_name = m_config["name"]
            scale_by_norm_for_metric = m_config.get("scale_by_norm", False)
            current_metric_instance = get_metric(name=metric_name, scale_by_norm=scale_by_norm_for_metric)
            
            try:
                # Ensure w_flat is not None before passing
                if w_flat is None:
                    raise ValueError("w_flat is None, cannot compute scores.")
                node_scores = current_metric_instance.compute_per_node_scores(X_acts_final_for_metric, w_flat, device=normalized_device)
                if debug_mode and node_scores is not None and node_scores.numel() > 0:
                    logger.info(f"  Metric '{metric_name}': Layer {layer_name_scores} score stats: min={torch.min(node_scores).item():.4f}, max={torch.max(node_scores).item():.4f}, mean={torch.mean(node_scores).item():.4f}, std={torch.std(node_scores).item():.4f}")
                metrics_for_this_layer[metric_name] = node_scores.detach() if node_scores is not None else torch.zeros(w_flat.shape[0] if w_flat is not None else 0, device=normalized_device)
            except Exception as e_comp:
                logger.error(f"Error computing scores for metric '{metric_name}' on layer {layer_name_scores}: {e_comp}", exc_info=debug_mode)
                node_count_fallback = layer_mod_scores.weight.shape[0] if hasattr(layer_mod_scores, 'weight') and layer_mod_scores.weight is not None else 0
                fallback_zeros_shape = w_flat.shape[0] if w_flat is not None and hasattr(w_flat, 'shape') else node_count_fallback
                metrics_for_this_layer[metric_name] = torch.zeros(fallback_zeros_shape, device=normalized_device)
        
        all_scores_per_layer_all_metrics[layer_idx] = metrics_for_this_layer
        model.hidden[layer_name_scores] = None 
    
    model.hidden.clear() 
    return all_scores_per_layer_all_metrics 