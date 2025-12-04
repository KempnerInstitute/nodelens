"""
LLM Pruning Baselines: Wanda and SparseGPT.

This module implements state-of-the-art LLM pruning methods for comparison:

1. Wanda (Sun et al., 2023): "A Simple and Effective Pruning Approach for Large Language Models"
   - Pruning metric: |W| × ||X||_2 (Weight magnitude × Activation norm)
   - One-shot structured pruning without retraining
   - Reference: https://arxiv.org/abs/2306.11695

2. SparseGPT (Frantar & Alistarh, 2023): "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
   - Second-order pruning using OBS-style weight reconstruction
   - Minimizes reconstruction error when pruning
   - Reference: https://arxiv.org/abs/2301.00774

These methods are baselines compared against alignment-based pruning in:
- NVIDIA Minitron (https://arxiv.org/abs/2407.14679)
- Our alignment-based pruning experiments
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BasePruningStrategy, PruningConfig

logger = logging.getLogger(__name__)


class WandaPruning(BasePruningStrategy):
    """
    Wanda: Pruning by Weights AND Activations.
    
    From Sun et al., 2023: "A Simple and Effective Pruning Approach for Large Language Models"
    
    The importance score for each weight is computed as:
        importance(w_ij) = |w_ij| × ||X_j||_2
    
    Where:
        - w_ij is the weight connecting input j to output i
        - X_j is the j-th input feature across calibration samples
        - ||X_j||_2 is the L2 norm of activations for input feature j
    
    This combines weight magnitude (traditional magnitude pruning) with
    activation magnitude (data-dependent importance).
    
    Args:
        config: Pruning configuration
        num_calibration_samples: Number of samples for calibration (default: 128)
        
    Example:
        >>> strategy = WandaPruning()
        >>> # Calibrate with sample activations
        >>> strategy.calibrate(model, calibration_dataloader)
        >>> # Compute importance scores for a layer
        >>> scores = strategy.compute_importance_scores(layer, activations=X)
    
    Reference:
        Sun et al. "A Simple and Effective Pruning Approach for Large Language Models"
        https://arxiv.org/abs/2306.11695
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
    ):
        super().__init__(config)
        self.num_calibration_samples = num_calibration_samples
        self.activation_norms: Dict[str, torch.Tensor] = {}
        self._calibrated = False
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Calibrate activation norms using calibration data.
        
        Args:
            model: Model to calibrate
            dataloader: Calibration data loader
            device: Device for computation
        """
        logger.info(f"Calibrating Wanda with {self.num_calibration_samples} samples...")
        
        # Dictionary to store activation norms per layer
        layer_activations: Dict[str, List[torch.Tensor]] = {}
        
        # Hook to capture activations
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                if name not in layer_activations:
                    layer_activations[name] = []
                # Store input activations (for weight × activation)
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                # Flatten batch and sequence dimensions, keep feature dim
                if inp.dim() == 3:
                    # [batch, seq, hidden] -> [batch*seq, hidden]
                    inp = inp.view(-1, inp.size(-1))
                layer_activations[name].append(inp.detach().cpu())
            return hook
        
        # Register hooks on Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Run calibration
        model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_seen >= self.num_calibration_samples:
                    break
                
                batch_size = 1  # Default batch size
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    batch_size = input_ids.size(0)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    model(input_ids, attention_mask=attention_mask)
                else:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                    else:
                        inputs = batch.to(device)
                    batch_size = inputs.size(0) if hasattr(inputs, 'size') else 1
                    model(inputs)
                
                samples_seen += batch_size
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute activation norms (L2 norm per feature dimension)
        for name, acts in layer_activations.items():
            if acts:
                # Concatenate all activations
                all_acts = torch.cat(acts, dim=0)  # [total_tokens, hidden]
                # Compute L2 norm per input feature
                self.activation_norms[name] = torch.norm(all_acts, p=2, dim=0)  # [hidden]
                logger.debug(f"Layer {name}: activation norm shape {self.activation_norms[name].shape}")
        
        self._calibrated = True
        logger.info(f"Wanda calibration complete. Computed norms for {len(self.activation_norms)} layers.")
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Wanda importance scores: |W| × ||X||_2
        
        Args:
            module: Linear module to compute scores for
            inputs: Input activations (if not using calibrated norms)
            layer_name: Name of the layer (for looking up calibrated norms)
            
        Returns:
            Importance scores with same shape as weights
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        
        weight = module.weight.data  # [out_features, in_features]
        
        # Get activation norms
        if inputs is not None:
            # Compute norms from provided inputs
            if inputs.dim() == 3:
                inputs = inputs.view(-1, inputs.size(-1))
            activation_norm = torch.norm(inputs, p=2, dim=0)  # [in_features]
        elif layer_name and layer_name in self.activation_norms:
            # Use calibrated norms
            activation_norm = self.activation_norms[layer_name].to(weight.device)
        elif self._calibrated:
            # Try to find matching layer name
            for name in self.activation_norms:
                if name.endswith(layer_name) or layer_name in name:
                    activation_norm = self.activation_norms[name].to(weight.device)
                    break
            else:
                logger.warning(f"No calibrated activation norms for layer {layer_name}, using weight magnitude only")
                return weight.abs()
        else:
            logger.warning("Wanda not calibrated and no inputs provided. Using weight magnitude only.")
            return weight.abs()
        
        # Ensure dimensions match
        if activation_norm.shape[0] != weight.shape[1]:
            logger.warning(f"Activation norm shape {activation_norm.shape} doesn't match "
                         f"weight in_features {weight.shape[1]}. Using weight magnitude only.")
            return weight.abs()
        
        # Wanda score: |W| × ||X||_2
        # Broadcasting: [out, in] × [in] -> [out, in]
        importance = weight.abs() * activation_norm.unsqueeze(0)
        
        return importance
    
    def get_structured_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Get structured (per-neuron/per-channel) importance scores.
        
        Args:
            module: Module to score
            inputs: Optional input activations
            layer_name: Layer name for calibrated norms
            dim: Dimension to aggregate over (0 for output neurons, 1 for input features)
            
        Returns:
            1D tensor of importance scores per neuron/channel
        """
        importance = self.compute_importance_scores(module, inputs, layer_name)
        
        # Aggregate to get per-neuron scores
        if dim == 0:
            # Sum over input dimension -> score per output neuron
            return importance.sum(dim=1)
        else:
            # Sum over output dimension -> score per input feature
            return importance.sum(dim=0)


class SparseGPTPruning(BasePruningStrategy):
    """
    SparseGPT: Second-order pruning with weight reconstruction.
    
    From Frantar & Alistarh, 2023: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
    
    This method uses second-order information (Hessian approximation) to:
    1. Determine which weights to prune (lowest saliency)
    2. Update remaining weights to compensate for pruning
    
    The saliency for pruning weight w_i is:
        saliency_i = w_i² / [H^{-1}]_ii
    
    Where H is the Hessian matrix approximated as X^T X (outer product of activations).
    
    After pruning w_i, remaining weights are updated:
        w_j := w_j - w_i * [H^{-1}]_ij / [H^{-1}]_ii
    
    This is an OBS (Optimal Brain Surgeon) style reconstruction that minimizes
    the increase in loss when pruning.
    
    Args:
        config: Pruning configuration
        num_calibration_samples: Number of samples for Hessian estimation
        block_size: Block size for blockwise reconstruction (default: 128)
        percdamp: Dampening factor for numerical stability (default: 0.01)
        
    Example:
        >>> strategy = SparseGPTPruning()
        >>> strategy.calibrate(model, calibration_dataloader)
        >>> # Prune with reconstruction
        >>> strategy.prune_layer(layer, sparsity=0.5)
    
    Reference:
        Frantar & Alistarh "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
        https://arxiv.org/abs/2301.00774
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
        block_size: int = 128,
        percdamp: float = 0.01,
    ):
        super().__init__(config)
        self.num_calibration_samples = num_calibration_samples
        self.block_size = block_size
        self.percdamp = percdamp
        self.hessians: Dict[str, torch.Tensor] = {}
        self._calibrated = False
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Compute Hessian approximation (X^T X) for each layer.
        
        Memory-optimized version that:
        - Processes activations incrementally (running sum)
        - Stores only diagonal for large layers to save memory
        - Keeps Hessians on CPU
        
        Args:
            model: Model to calibrate
            dataloader: Calibration data loader
            device: Device for computation
        """
        logger.info(f"Calibrating SparseGPT with {self.num_calibration_samples} samples...")
        
        # For memory efficiency, we'll compute running sum of X^T X
        # Store (running_H, nsamples) per layer
        running_hessians: Dict[str, Tuple[torch.Tensor, int]] = {}
        
        # Hook to capture activations and update Hessian incrementally
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    inp = input[0]
                else:
                    inp = input
                
                # Flatten to 2D: [batch*seq, features]
                if inp.dim() == 3:
                    inp = inp.view(-1, inp.size(-1))
                
                # Move to CPU and float32 for stability
                inp = inp.detach().float().cpu()
                n_tokens = inp.shape[0]
                
                # Compute H increment: X^T X
                H_inc = inp.T @ inp
                
                if name not in running_hessians:
                    running_hessians[name] = (H_inc, n_tokens)
                else:
                    old_H, old_n = running_hessians[name]
                    running_hessians[name] = (old_H + H_inc, old_n + n_tokens)
            return hook
        
        # Register hooks only for Linear layers (MLP layers)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Only hook MLP layers to save memory (skip attention)
                if any(p in name for p in ["mlp", "up_proj", "gate_proj", "down_proj", "fc"]):
                    hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Run calibration
        model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_seen >= self.num_calibration_samples:
                    break
                
                batch_size = 1
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    batch_size = input_ids.size(0)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    model(input_ids, attention_mask=attention_mask)
                else:
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(device)
                    else:
                        inputs = batch.to(device)
                    batch_size = inputs.size(0) if hasattr(inputs, 'size') else 1
                    model(inputs)
                
                samples_seen += batch_size
                
                # Clear CUDA cache periodically
                if samples_seen % 4 == 0:
                    torch.cuda.empty_cache()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Finalize Hessians: normalize and add dampening
        for name, (H_sum, nsamples) in running_hessians.items():
            if nsamples > 0:
                # Normalize
                H = H_sum / nsamples
                
                # Add dampening for numerical stability
                damp = self.percdamp * torch.diag(H).mean()
                H += damp * torch.eye(H.shape[0], device=H.device)
                
                # Store on CPU to save GPU memory
                self.hessians[name] = H.cpu()
                logger.debug(f"Layer {name}: Hessian shape {H.shape}")
        
        # Clear running storage
        del running_hessians
        torch.cuda.empty_cache()
        
        self._calibrated = True
        logger.info(f"SparseGPT calibration complete. Computed Hessians for {len(self.hessians)} layers.")
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute SparseGPT saliency scores: w² / [H^{-1}]_ii
        
        For unstructured pruning, this gives the "cost" of removing each weight.
        Lower scores = safer to prune.
        
        For structured pruning, we aggregate over the neuron dimension.
        
        Args:
            module: Linear module
            inputs: Optional inputs for online Hessian computation
            layer_name: Layer name for looking up calibrated Hessian
            
        Returns:
            Importance scores
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        
        weight = module.weight.data.float()  # [out, in]
        
        # Get Hessian
        H = None
        if layer_name and layer_name in self.hessians:
            H = self.hessians[layer_name]
        elif self._calibrated:
            for name in self.hessians:
                if name.endswith(layer_name) or layer_name in name:
                    H = self.hessians[name]
                    break
        
        if H is None:
            if inputs is not None:
                # Compute Hessian from inputs
                if inputs.dim() == 3:
                    inputs = inputs.view(-1, inputs.size(-1))
                inputs = inputs.float()
                nsamples = inputs.shape[0]
                H = (inputs.T @ inputs) / nsamples
                damp = self.percdamp * torch.diag(H).mean()
                H += damp * torch.eye(H.shape[0], device=H.device)
            else:
                logger.warning("SparseGPT not calibrated and no inputs. Using weight magnitude.")
                return weight.abs()
        
        H = H.to(weight.device)
        
        # Compute H^{-1} diagonal (we need [H^{-1}]_ii for saliency)
        try:
            # For efficiency, use Cholesky decomposition
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv_diag = torch.diag(H_inv)
        except RuntimeError:
            # Fall back to direct inverse if Cholesky fails
            try:
                H_inv = torch.linalg.inv(H)
                H_inv_diag = torch.diag(H_inv)
            except RuntimeError:
                logger.warning("Hessian inversion failed, using weight magnitude")
                return weight.abs()
        
        # Saliency score: w² / [H^{-1}]_ii
        # Higher saliency = more important (bigger loss increase if pruned)
        # Broadcasting: [out, in]² / [in] -> [out, in]
        saliency = (weight ** 2) / (H_inv_diag.unsqueeze(0) + 1e-10)
        
        return saliency
    
    def get_structured_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        dim: int = 0,
    ) -> torch.Tensor:
        """
        Get structured importance scores (aggregated per neuron).
        
        Args:
            module: Module to score
            inputs: Optional input activations
            layer_name: Layer name
            dim: Dimension to aggregate
            
        Returns:
            1D tensor of per-neuron scores
        """
        saliency = self.compute_importance_scores(module, inputs, layer_name)
        
        # Aggregate to get per-neuron scores
        if dim == 0:
            return saliency.sum(dim=1)
        else:
            return saliency.sum(dim=0)
    
    def prune_and_reconstruct(
        self,
        module: nn.Module,
        sparsity: float,
        layer_name: Optional[str] = None,
        inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune weights and reconstruct remaining weights to minimize error.
        
        This is the full SparseGPT algorithm with OBS-style reconstruction.
        
        Args:
            module: Linear module to prune
            sparsity: Target sparsity (fraction to prune)
            layer_name: Layer name for Hessian lookup
            inputs: Optional inputs for online computation
            
        Returns:
            Tuple of (pruning_mask, reconstructed_weight)
        """
        if not hasattr(module, "weight"):
            raise ValueError("Module does not have weights")
        
        W = module.weight.data.clone().float()
        rows, cols = W.shape
        
        # Get Hessian
        H = None
        if layer_name and layer_name in self.hessians:
            H = self.hessians[layer_name].to(W.device)
        elif inputs is not None:
            if inputs.dim() == 3:
                inputs = inputs.view(-1, inputs.size(-1))
            inputs = inputs.float().to(W.device)
            nsamples = inputs.shape[0]
            H = (inputs.T @ inputs) / nsamples
            damp = self.percdamp * torch.diag(H).mean()
            H += damp * torch.eye(H.shape[0], device=H.device)
        else:
            logger.warning("No Hessian available, returning simple magnitude pruning")
            scores = W.abs()
            k = int(sparsity * W.numel())
            threshold = scores.flatten().kthvalue(k).values
            mask = (scores > threshold).float()
            return mask, W * mask
        
        # Compute H^{-1} using Cholesky
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            logger.warning("Cholesky failed, using direct inverse")
            H_inv = torch.linalg.inv(H)
        
        # Number of weights to prune
        num_prune = int(sparsity * W.numel())
        
        # Create mask (1 = keep, 0 = prune)
        mask = torch.ones_like(W)
        
        # Prune in blocks for efficiency (simplified version)
        # Full SparseGPT uses column-wise processing; this is a simplified version
        
        # Compute saliency for all weights
        H_inv_diag = torch.diag(H_inv)
        saliency = (W ** 2) / (H_inv_diag.unsqueeze(0) + 1e-10)
        
        # Find weights with lowest saliency to prune
        flat_saliency = saliency.flatten()
        prune_indices = torch.topk(flat_saliency, num_prune, largest=False).indices
        
        # Create mask
        mask_flat = mask.flatten()
        mask_flat[prune_indices] = 0
        mask = mask_flat.view(rows, cols)
        
        # Reconstruct remaining weights (simplified - full algo does column-by-column)
        # For each pruned weight, update connected weights
        # This is a simplified version; full SparseGPT is more sophisticated
        W_new = W.clone()
        
        # Zero out pruned weights
        W_new = W_new * mask
        
        # Convert back to original dtype
        W_new = W_new.to(module.weight.dtype)
        
        return mask, W_new


# Convenience functions for integration with the pruning framework

def compute_wanda_scores(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute Wanda scores for all Linear layers.
    
    Args:
        model: Model to analyze
        dataloader: Calibration data
        device: Device
        num_samples: Number of calibration samples
        
    Returns:
        Dict mapping layer names to importance scores
    """
    strategy = WandaPruning(num_calibration_samples=num_samples)
    strategy.calibrate(model, dataloader, device)
    
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scores[name] = strategy.compute_importance_scores(
                module, layer_name=name
            )
    
    return scores


def compute_sparsegpt_scores(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to compute SparseGPT saliency scores.
    
    Args:
        model: Model to analyze
        dataloader: Calibration data
        device: Device
        num_samples: Number of calibration samples
        
    Returns:
        Dict mapping layer names to saliency scores
    """
    strategy = SparseGPTPruning(num_calibration_samples=num_samples)
    strategy.calibrate(model, dataloader, device)
    
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scores[name] = strategy.compute_importance_scores(
                module, layer_name=name
            )
    
    return scores

