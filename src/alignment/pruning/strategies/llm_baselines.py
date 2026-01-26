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

        # IMPORTANT (faithful to canonical Wanda behavior + memory):
        # Official Wanda implementations accumulate a running statistic (per layer) instead of
        # storing all activations. The canonical update (see `external/wanda/layerwrapper.py`
        # in origin/iss117_acllm_v3) is equivalent to maintaining:
        #   scaler_row[j] = E_sample[ sum_t x_{t,j}^2 ]   (avg over samples; sum over tokens)
        # and then using sqrt(scaler_row) as the per-input-feature activation scale.
        #
        # This differs from "concatenate all activations then take torch.norm" only by a
        # layer-constant scaling (for fixed sequence length), but the running version is
        # much more memory-efficient and matches reference code structure.
        running: Dict[str, Tuple[torch.Tensor, int]] = {}  # name -> (scaler_row (CPU), nsamples)

        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                # Store input activations (for weight × activation)
                inp = input[0] if isinstance(input, tuple) else input

                # Normalize shapes to match reference Wanda behavior:
                # - If 2D, treat as a single sample (batch=1).
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)

                tmp = int(inp.shape[0])  # batch size (number of samples in this hook call)
                if tmp <= 0:
                    return

                # Flatten batch & sequence into tokens for the sum-of-squares statistic.
                # Typical LLM MLP inputs are [B, S, F].
                if inp.dim() == 3:
                    tokens = inp.reshape(-1, inp.shape[-1])  # [B*S, F]
                else:
                    # Fallback: treat last dim as features and everything else as "tokens".
                    tokens = inp.reshape(-1, inp.shape[-1])

                # sum_t x^2 for each input feature (over tokens and batch)
                sumsq = tokens.detach().to(dtype=torch.float32).pow(2).sum(dim=0).cpu()  # [F]

                if name not in running:
                    running[name] = (sumsq / tmp, tmp)
                else:
                    scaler_row, nsamples = running[name]
                    new_n = nsamples + tmp
                    # Running mean update (matches reference logic)
                    scaler_row = scaler_row * (nsamples / new_n) + (sumsq / new_n)
                    running[name] = (scaler_row, new_n)

            return hook

        # Register hooks (only for MLP/FFN projections by default to keep calibration lightweight).
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(p in name for p in ["mlp", "up_proj", "gate_proj", "down_proj", "fc"]):
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
        
        # Finalize activation norms:
        # activation_norm[j] = sqrt(scaler_row[j]) where scaler_row is the running avg of sumsq.
        self.activation_norms = {}
        for name, (scaler_row, nsamples) in running.items():
            if nsamples <= 0:
                continue
            # Guard against tiny numerical negatives.
            scaler_row = torch.clamp(scaler_row, min=0.0)
            self.activation_norms[name] = torch.sqrt(scaler_row)
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

    def prune_unstructured_inplace(
        self,
        module: nn.Module,
        sparsity: float,
        *,
        layer_name: Optional[str] = None,
        mode: str = "low",
        per_row: bool = True,
    ) -> torch.Tensor:
        """
        Apply Wanda-style *unstructured* pruning to a Linear module (in-place).

        This corresponds to the original Wanda setting: prune individual weights using
        the Wanda score |W| * ||X||_2 computed from calibration activations.

        Notes:
        - Many Wanda implementations prune *per output row* (per-neuron) to enforce uniform
          sparsity across rows. We expose this via `per_row` (default True).

        Args:
            module: nn.Linear-like module with `.weight`.
            sparsity: Fraction of weights to prune in this module (0..1).
            layer_name: Optional name used to look up calibrated activation norms.
            mode: 'low' prunes low Wanda-scores (standard); 'high' prunes high scores (stress test);
                  'random' prunes random weights.
            per_row: If True, prune the same fraction within each output row.

        Returns:
            Mask tensor with same shape as module.weight (1 = keep, 0 = prune).
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        if sparsity <= 0:
            return torch.ones_like(module.weight.data, dtype=torch.float32)

        scores = self.compute_importance_scores(module, layer_name=layer_name)
        scores = scores.detach()

        W = module.weight.data
        device = W.device
        scores = scores.to(device)

        rows, cols = scores.shape
        mask = torch.ones((rows, cols), dtype=torch.bool, device=device)

        if per_row:
            k = int(sparsity * cols)
            if k <= 0:
                return mask.float()

            if mode == "random":
                # Random selection per row
                rand = torch.rand((rows, cols), device=device)
                sort_res = torch.sort(rand, dim=1, stable=True)
                idx = sort_res[1][:, :k]
            elif mode == "low":
                # Paper-faithful: stable row-wise sort, then prune lowest fraction.
                sort_res = torch.sort(scores, dim=1, stable=True)
                idx = sort_res[1][:, :k]
            else:  # mode == "high"
                # Stable row-wise sort, prune highest fraction.
                sort_res = torch.sort(scores, dim=1, stable=True)
                idx = sort_res[1][:, -k:]

            row_idx = torch.arange(rows, device=device).unsqueeze(1).expand_as(idx)
            mask[row_idx, idx] = False
        else:
            # Global unstructured selection within the matrix
            flat = scores.flatten()
            k = int(sparsity * flat.numel())
            if k <= 0:
                return mask.float()
            if mode == "random":
                idx = torch.randperm(flat.numel(), device=device)[:k]
            elif mode == "low":
                _, idx = torch.topk(flat, k, largest=False)
            else:
                _, idx = torch.topk(flat, k, largest=True)
            mask_flat = mask.flatten()
            mask_flat[idx] = False
            mask = mask_flat.view_as(mask)

        # Apply in-place
        with torch.no_grad():
            module.weight.data.mul_(mask.float())

        return mask.float()


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
        Prune weights and reconstruct remaining weights (SparseGPT-style).

        This implements the key SparseGPT mechanism: prune low-saliency weights and
        propagate the induced error using a second-order (Hessian/covariance) approximation.

        Important:
        - This operates in the *unstructured* setting (weight-level pruning).
        - Some workflows also provide a separate *channel-adapted* SparseGPT baseline which
          uses the diagonal saliency as a scoring signal for structured channel pruning. This
          method is the canonical unstructured (weight-level) variant.
        """
        if not hasattr(module, "weight"):
            raise ValueError("Module does not have weights")
        
        if sparsity <= 0:
            W0 = module.weight.data.clone()
            return torch.ones_like(W0, dtype=torch.float32), W0

        W = module.weight.data.clone().float()
        orig_dtype = module.weight.data.dtype
        rows, cols = W.shape
        device = W.device
        
        # Get Hessian
        H = None
        if layer_name and layer_name in self.hessians:
            H = self.hessians[layer_name]
        elif inputs is not None:
            if inputs.dim() == 3:
                inputs = inputs.view(-1, inputs.size(-1))
            inputs = inputs.float().to(device)
            nsamples = inputs.shape[0]
            H = (inputs.T @ inputs) / nsamples
            damp = self.percdamp * torch.diag(H).mean()
            H += damp * torch.eye(H.shape[0], device=H.device)
        else:
            logger.warning("No Hessian available, returning simple magnitude pruning")
            scores = W.abs()
            k = int(sparsity * W.numel())
            # Use topk for exact k selection (avoids threshold tie issues)
            flat_scores = scores.flatten()
            _, indices_to_prune = torch.topk(flat_scores, k, largest=False)
            mask = torch.ones(flat_scores.numel(), dtype=torch.bool, device=W.device)
            mask[indices_to_prune] = False
            mask = mask.view(W.shape).float()
            return mask, W * mask

        # Resolve H from stored dict via fuzzy match if needed
        if H is None and self._calibrated and layer_name is not None:
            for name in self.hessians:
                if name.endswith(layer_name) or layer_name in name:
                    H = self.hessians[name]
                    break

        if H is None:
            logger.warning("SparseGPT: Hessian not available; falling back to magnitude pruning")
            scores = W.abs()
            k = int(sparsity * W.numel())
            flat_scores = scores.flatten()
            _, indices_to_prune = torch.topk(flat_scores, k, largest=False)
            keep = torch.ones(flat_scores.numel(), dtype=torch.bool, device=device)
            keep[indices_to_prune] = False
            keep = keep.view_as(W)
            W_new = (W * keep.float()).to(orig_dtype)
            return keep.float(), W_new

        # Move Hessian to device for reconstruction
        H = H.float().to(device)

        # Handle dead inputs (zero diagonal) like official implementations
        diagH = torch.diag(H)
        dead = diagH == 0
        if dead.any():
            H[dead, dead] = 1.0
            W[:, dead] = 0.0

        # Dampening for numerical stability (SparseGPT/GPTQ style)
        diagH = torch.diag(H)
        damp = self.percdamp * diagH.mean()
        idx = torch.arange(H.shape[0], device=device)
        H[idx, idx] += damp

        # Compute Cholesky + inverse + (upper) Cholesky factor of inverse Hessian
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            Hinv_factor = torch.linalg.cholesky(H_inv, upper=True)
        except RuntimeError as e:
            logger.warning(f"SparseGPT: Cholesky failed ({e}); adding extra dampening and retrying")
            H[idx, idx] += (10.0 * damp)
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            Hinv_factor = torch.linalg.cholesky(H_inv, upper=True)

        d = torch.diag(Hinv_factor)
        denom = (d.unsqueeze(0) ** 2) + 1e-10

        # Global unstructured prune mask from SparseGPT/GPTQ-style saliency
        saliency = (W ** 2) / denom
        num_prune = int(sparsity * W.numel())
        if num_prune <= 0:
            W_new = W.to(orig_dtype)
            return torch.ones_like(W, dtype=torch.float32), W_new

        flat = saliency.flatten()
        prune_idx = torch.topk(flat, num_prune, largest=False).indices
        prune_mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        prune_mask_flat[prune_idx] = True
        prune_mask = prune_mask_flat.view_as(W)  # True = prune

        # Reconstruction / error propagation (blockwise, GPTQ-style)
        bs = int(self.block_size) if self.block_size else 128
        W_work = W.clone()

        for i1 in range(0, cols, bs):
            i2 = min(i1 + bs, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Hinv1 = Hinv_factor[i1:i2, i1:i2]
            Err1 = torch.zeros((rows, count), device=device, dtype=W_work.dtype)

            for j in range(count):
                w = W1[:, j]
                to_prune = prune_mask[:, i1 + j]
                q = w.clone()
                q[to_prune] = 0.0

                dj = Hinv1[j, j]
                err = (w - q) / (dj + 1e-10)

                # Update remaining columns in this block
                W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
                Err1[:, j] = err

                # Enforce pruning at this column
                W1[:, j] = q

            W_work[:, i1:i2] = W1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv_factor[i1:i2, i2:]

        W_new = W_work.to(orig_dtype)
        keep_mask = (~prune_mask).float()
        return keep_mask, W_new


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


# =============================================================================
# OWL: Outlier-aware Wanda
# =============================================================================

class OWLPruning(WandaPruning):
    """
    OWL: Outlier-aware Weight pruning for LLMs.
    
    From Yin et al., 2024: "Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret
    Sauce for Pruning LLMs to High Sparsity"
    
    Key insight: Activation outliers (similar to supernodes) require special handling.
    OWL uses non-uniform layer-wise sparsity based on outlier ratio per layer.
    
    Layers with more outliers get lower sparsity (more weights kept), while
    layers with fewer outliers can be pruned more aggressively.
    
    Args:
        config: Pruning configuration
        num_calibration_samples: Number of samples for calibration
        outlier_threshold: Z-score threshold for outlier detection (default: 3.0)
        sparsity_range: (min_sparsity, max_sparsity) for layer-wise allocation
        
    Reference:
        Yin et al. "Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce
        for Pruning LLMs to High Sparsity"
        https://arxiv.org/abs/2310.05175
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
        outlier_threshold: float = 3.0,
        sparsity_range: Tuple[float, float] = (0.3, 0.7),
    ):
        super().__init__(config, num_calibration_samples)
        self.outlier_threshold = outlier_threshold
        self.sparsity_range = sparsity_range
        self.layer_outlier_ratios: Dict[str, float] = {}
        self.layer_sparsities: Dict[str, float] = {}
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Calibrate activation norms and compute outlier ratios per layer.
        """
        # First, run standard Wanda calibration
        super().calibrate(model, dataloader, device)
        
        # Compute outlier ratios per layer based on activation norms
        logger.info("Computing OWL outlier ratios...")
        
        for name, norm in self.activation_norms.items():
            if norm.numel() == 0:
                continue
            
            # Compute z-scores for activation norms
            mean = norm.mean()
            std = norm.std()
            if std > 1e-10:
                z_scores = (norm - mean) / std
                # Outlier ratio: fraction of features with |z| > threshold
                outlier_ratio = (z_scores.abs() > self.outlier_threshold).float().mean().item()
            else:
                outlier_ratio = 0.0
            
            self.layer_outlier_ratios[name] = outlier_ratio
            logger.debug(f"Layer {name}: outlier ratio = {outlier_ratio:.4f}")
        
        # Allocate layer-wise sparsities inversely proportional to outlier ratio
        self._allocate_layerwise_sparsity()
        
        logger.info(f"OWL calibration complete. Outlier ratios: "
                   f"min={min(self.layer_outlier_ratios.values()):.4f}, "
                   f"max={max(self.layer_outlier_ratios.values()):.4f}")
    
    def _allocate_layerwise_sparsity(self, target_sparsity: float = 0.5) -> None:
        """
        Allocate non-uniform sparsity based on outlier ratios.
        
        Layers with more outliers get lower sparsity (keep more weights).
        """
        if not self.layer_outlier_ratios:
            return
        
        min_sp, max_sp = self.sparsity_range
        
        # Normalize outlier ratios
        ratios = list(self.layer_outlier_ratios.values())
        min_r, max_r = min(ratios), max(ratios)
        
        for name, ratio in self.layer_outlier_ratios.items():
            if max_r > min_r:
                # Inverse mapping: high outlier ratio -> low sparsity
                normalized = (ratio - min_r) / (max_r - min_r)
                # Interpolate: layers with more outliers get sparsity closer to min_sp
                layer_sparsity = max_sp - normalized * (max_sp - min_sp)
            else:
                layer_sparsity = target_sparsity
            
            self.layer_sparsities[name] = layer_sparsity
    
    def get_layer_sparsity(self, layer_name: str, default_sparsity: float = 0.5) -> float:
        """Get the allocated sparsity for a specific layer."""
        # Try exact match first
        if layer_name in self.layer_sparsities:
            return self.layer_sparsities[layer_name]
        
        # Try partial match
        for name, sparsity in self.layer_sparsities.items():
            if name.endswith(layer_name) or layer_name in name:
                return sparsity
        
        return default_sparsity
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute OWL importance scores.
        
        Same as Wanda but with awareness of outlier channels.
        """
        # Get base Wanda scores
        importance = super().compute_importance_scores(module, inputs, layer_name, **kwargs)
        
        # Optionally boost importance of outlier channels
        if layer_name and layer_name in self.activation_norms:
            norm = self.activation_norms[layer_name].to(importance.device)
            mean = norm.mean()
            std = norm.std()
            if std > 1e-10:
                z_scores = (norm - mean) / std
                # Channels with high z-scores (outliers) get importance boost
                outlier_mask = z_scores.abs() > self.outlier_threshold
                boost_factor = 1.0 + z_scores.abs().clamp(0, 10) / 10  # [1.0, 2.0] boost
                importance = importance * boost_factor.unsqueeze(0)
        
        return importance


# =============================================================================
# LLM-Pruner: Structured Pruning with Dependency Awareness
# =============================================================================

class LLMPrunerChannelMode(BasePruningStrategy):
    """
    LLM-Pruner in Channel Mode: Structured pruning for LLMs.
    
    From Ma et al., 2023: "LLM-Pruner: On the Structural Pruning of Large Language Models"
    
    Key features:
    1. Dependency-aware grouping: Identifies coupled structures that must be pruned together
    2. Taylor-based importance: Uses first-order Taylor expansion for importance estimation
    3. Channel-level granularity: Prunes entire FFN channels (structured)
    
    This implementation focuses on FFN channel pruning (similar to SCAR) but uses
    the LLM-Pruner importance estimation.
    
    Args:
        config: Pruning configuration
        num_calibration_samples: Number of samples for calibration
        use_gradient: Whether to use gradient information (requires backward pass)
        
    Reference:
        Ma et al. "LLM-Pruner: On the Structural Pruning of Large Language Models"
        https://arxiv.org/abs/2305.11627
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
        use_gradient: bool = True,
    ):
        super().__init__(config)
        self.num_calibration_samples = num_calibration_samples
        self.use_gradient = use_gradient
        self.taylor_scores: Dict[str, torch.Tensor] = {}
        self.activation_means: Dict[str, torch.Tensor] = {}
        self._calibrated = False
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
        loss_fn = None,
    ) -> None:
        """
        Calibrate Taylor importance scores using calibration data.
        
        For gradient-based Taylor scores, we need to compute:
            importance(neuron_i) = |activation_i * gradient_i|
        
        Without gradients, we use activation magnitude as proxy.
        """
        logger.info(f"Calibrating LLM-Pruner with {self.num_calibration_samples} samples...")
        
        activation_stats: Dict[str, List[torch.Tensor]] = {}
        gradient_stats: Dict[str, List[torch.Tensor]] = {}
        
        hooks = []
        
        def make_fwd_hook(name: str):
            def hook(module, input, output):
                # Store output activations for MLP layers
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                    if act.dim() == 3:
                        # [B, S, D] -> mean over batch and sequence
                        act_mean = act.abs().mean(dim=(0, 1))
                    else:
                        act_mean = act.abs().mean(dim=0)
                    
                    if name not in activation_stats:
                        activation_stats[name] = []
                    activation_stats[name].append(act_mean.cpu())
            return hook
        
        # Register hooks for MLP layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(p in name for p in ["mlp", "up_proj", "gate_proj", "fc1"]):
                    hooks.append(module.register_forward_hook(make_fwd_hook(name)))
        
        # Run calibration
        model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_seen >= self.num_calibration_samples:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    model(input_ids, attention_mask=attention_mask)
                    batch_size = input_ids.size(0)
                else:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    model(inputs)
                    batch_size = inputs.size(0)
                
                samples_seen += batch_size
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Aggregate activation stats
        for name, acts in activation_stats.items():
            if acts:
                self.activation_means[name] = torch.stack(acts).mean(dim=0)
        
        # Taylor scores: For channel pruning, we use activation magnitude as importance
        # (Full Taylor would require gradients which need loss computation)
        for name, act_mean in self.activation_means.items():
            self.taylor_scores[name] = act_mean
        
        self._calibrated = True
        logger.info(f"LLM-Pruner calibration complete. Scored {len(self.taylor_scores)} layers.")
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute LLM-Pruner importance scores for channels.
        
        Uses Taylor-based importance: |activation × gradient| or just |activation|.
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        
        weight = module.weight.data
        
        # Try to get calibrated scores
        if layer_name and layer_name in self.taylor_scores:
            taylor = self.taylor_scores[layer_name].to(weight.device)
            # Combine with weight magnitude
            weight_mag = weight.abs().sum(dim=0)  # Per input channel
            if taylor.shape[0] == weight_mag.shape[0]:
                importance = taylor * weight_mag
            else:
                importance = weight_mag
        elif self._calibrated:
            # Try partial match
            for name in self.taylor_scores:
                if name.endswith(layer_name) or layer_name in name:
                    taylor = self.taylor_scores[name].to(weight.device)
                    weight_mag = weight.abs().sum(dim=0)
                    if taylor.shape[0] == weight_mag.shape[0]:
                        importance = taylor * weight_mag
                    else:
                        importance = weight_mag
                    break
            else:
                importance = weight.abs().sum(dim=0)
        else:
            # Fallback: weight magnitude
            importance = weight.abs().sum(dim=0)
        
        return importance
    
    def get_structured_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        dim: int = 1,
    ) -> torch.Tensor:
        """
        Get per-channel importance scores for structured pruning.
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        weight = module.weight.data  # [out_features, in_features]

        # Per-output-channel (rows) vs per-input-channel (cols)
        if dim == 0:
            weight_mag = weight.abs().sum(dim=1)  # [out_features]

            taylor = None
            if layer_name and layer_name in self.taylor_scores:
                taylor = self.taylor_scores[layer_name].to(weight.device)
            elif self._calibrated and layer_name:
                # Try partial match
                for name in self.taylor_scores:
                    if name.endswith(layer_name) or layer_name in name:
                        taylor = self.taylor_scores[name].to(weight.device)
                        break

            if taylor is not None and taylor.shape == weight_mag.shape:
                return (taylor.abs() * weight_mag).detach()

            return weight_mag.detach()

        if dim == 1:
            # We do not currently compute input-channel Taylor stats; fall back to column norms.
            return weight.abs().sum(dim=0).detach()  # [in_features]

        raise ValueError(f"Invalid dim={dim}; expected 0 (rows) or 1 (cols).")


# Convenience functions for new baselines

def compute_owl_scores(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_samples: int = 128,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Compute OWL scores and layer-wise sparsities.
    
    Returns:
        Tuple of (importance_scores, layer_sparsities)
    """
    strategy = OWLPruning(num_calibration_samples=num_samples)
    strategy.calibrate(model, dataloader, device)
    
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scores[name] = strategy.compute_importance_scores(
                module, layer_name=name
            )
    
    return scores, strategy.layer_sparsities


def compute_llmpruner_scores(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Compute LLM-Pruner Taylor-based importance scores.
    """
    strategy = LLMPrunerChannelMode(num_calibration_samples=num_samples)
    strategy.calibrate(model, dataloader, device)
    
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scores[name] = strategy.compute_importance_scores(
                module, layer_name=name
            )
    
    return scores


# =============================================================================
# FLAP: Fluctuation-based Adaptive Structured Pruning
# =============================================================================

class FLAPPruning(BasePruningStrategy):
    """
    FLAP: Fluctuation-based Adaptive Structured Pruning for LLMs.
    
    Key insight: Use activation fluctuation (variance) across calibration samples
    to identify channels that have consistent vs. variable activations.
    Channels with low fluctuation are more safely prunable.
    
    Args:
        config: Pruning configuration
        num_calibration_samples: Number of samples for calibration
        
    Reference:
        An et al. "Fluctuation-based Adaptive Structured Pruning for Large Language Models"
        https://arxiv.org/abs/2312.11983
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
    ):
        super().__init__(config)
        self.num_calibration_samples = num_calibration_samples
        self.activation_means: Dict[str, torch.Tensor] = {}
        self.activation_vars: Dict[str, torch.Tensor] = {}
        self._calibrated = False
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Calibrate by computing activation mean and variance per channel.
        """
        logger.info(f"Calibrating FLAP with {self.num_calibration_samples} samples...")
        
        # Running statistics
        running_sum: Dict[str, torch.Tensor] = {}
        running_sq_sum: Dict[str, torch.Tensor] = {}
        running_count: Dict[str, int] = {}
        
        hooks = []
        
        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                    if act.dim() == 3:
                        # [B, S, D] -> compute per-channel stats
                        # Flatten to [B*S, D]
                        act_flat = act.view(-1, act.shape[-1])
                    else:
                        act_flat = act.view(-1, act.shape[-1])
                    
                    ch_sum = act_flat.sum(dim=0).cpu()
                    ch_sq_sum = (act_flat ** 2).sum(dim=0).cpu()
                    count = act_flat.shape[0]
                    
                    if name not in running_sum:
                        running_sum[name] = ch_sum
                        running_sq_sum[name] = ch_sq_sum
                        running_count[name] = count
                    else:
                        running_sum[name] += ch_sum
                        running_sq_sum[name] += ch_sq_sum
                        running_count[name] += count
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(p in name for p in ["mlp", "up_proj", "gate_proj", "fc"]):
                    hooks.append(module.register_forward_hook(make_hook(name)))
        
        model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_seen >= self.num_calibration_samples:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    model(input_ids, attention_mask=attention_mask)
                    batch_size = input_ids.size(0)
                else:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    model(inputs)
                    batch_size = inputs.size(0)
                
                samples_seen += batch_size
        
        for hook in hooks:
            hook.remove()
        
        # Compute mean and variance
        for name in running_sum:
            n = running_count[name]
            mean = running_sum[name] / n
            var = (running_sq_sum[name] / n) - (mean ** 2)
            var = torch.clamp(var, min=0)  # Numerical stability
            
            self.activation_means[name] = mean
            self.activation_vars[name] = var
        
        self._calibrated = True
        logger.info(f"FLAP calibration complete. Scored {len(self.activation_means)} layers.")
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        FLAP importance: channels with HIGH mean activation and LOW variance
        are more important (consistent, strong signal).
        
        score = mean / (std + eps)  -- Signal-to-noise ratio
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        
        weight = module.weight.data
        eps = 1e-6
        
        if layer_name and layer_name in self.activation_means:
            mean = self.activation_means[layer_name].to(weight.device)
            var = self.activation_vars[layer_name].to(weight.device)
            std = torch.sqrt(var + eps)
            
            # SNR-based importance
            importance = mean.abs() / (std + eps)
            
            # Weight magnitude contribution
            weight_norm = weight.abs().sum(dim=0)
            if weight_norm.shape[0] == importance.shape[0]:
                importance = importance * weight_norm
        else:
            # Fallback to weight magnitude
            importance = weight.abs().sum(dim=0)
        
        return importance
    
    def get_structured_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        dim: int = 1,
    ) -> torch.Tensor:
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        if dim == 0:
            # Natural FLAP score: per-output-channel fluctuation/SNR
            return self.compute_importance_scores(module, inputs, layer_name)

        if dim == 1:
            # Column importance: fall back to weight column norms (input-channel contribution)
            weight = module.weight.data  # [out_features, in_features]
            return weight.abs().sum(dim=0).detach()

        raise ValueError(f"Invalid dim={dim}; expected 0 (rows) or 1 (cols).")


# =============================================================================
# RIA: Relative Importance and Activation
# =============================================================================

class RIAPruning(WandaPruning):
    """
    RIA: Relative Importance and Activation for structured pruning.
    
    Extends Wanda with relative (normalized) importance scores to handle
    scale differences across layers more gracefully.
    
    score_i = |W_i| × ||X_i||_2 / (layer_norm_factor)
    
    Reference:
        "Plug-and-Play: A Simple and Effective Pruning Approach for LLMs"
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
    ):
        super().__init__(config, num_calibration_samples)
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute RIA importance scores with layer-wise normalization.
        """
        # Get base Wanda scores
        importance = super().compute_importance_scores(module, inputs, layer_name, **kwargs)
        
        # Normalize by layer statistics (relative importance)
        layer_mean = importance.mean()
        layer_std = importance.std()
        
        if layer_std > 1e-8:
            # Z-score normalization
            importance = (importance - layer_mean) / layer_std
            # Shift to positive
            importance = importance - importance.min() + 1e-6
        
        return importance


# =============================================================================
# SlimLLM-style: Holistic Channel Importance
# =============================================================================

class SlimLLMPruning(BasePruningStrategy):
    """
    SlimLLM-style pruning: Holistic channel/head importance estimation.
    
    Key idea: Assess importance at the entire channel level by measuring
    the impact of zeroing each channel on output reconstruction error.
    
    For computational efficiency, we approximate this using:
    - Activation magnitude (how much the channel fires)
    - Weight magnitude (how much the channel affects output)
    - Gradient approximation (how much loss changes)
    
    Reference:
        Guo et al. "SlimLLM: An Expert Mixture Approach to Structured Pruning of LLMs"
        (2025)
    """
    
    def __init__(
        self,
        config: Optional[PruningConfig] = None,
        num_calibration_samples: int = 128,
    ):
        super().__init__(config)
        self.num_calibration_samples = num_calibration_samples
        self.channel_activations: Dict[str, torch.Tensor] = {}
        self.channel_gradients: Dict[str, torch.Tensor] = {}
        self._calibrated = False
    
    def calibrate(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cuda",
    ) -> None:
        """
        Calibrate by collecting activation statistics.
        """
        logger.info(f"Calibrating SlimLLM with {self.num_calibration_samples} samples...")
        
        activation_sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        
        hooks = []
        
        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                    if act.dim() == 3:
                        # [B, S, D] -> L2 norm per channel
                        act_norm = (act ** 2).sum(dim=(0, 1)).sqrt().cpu()
                    else:
                        act_norm = (act ** 2).sum(dim=0).sqrt().cpu()
                    
                    if name not in activation_sums:
                        activation_sums[name] = act_norm
                        counts[name] = 1
                    else:
                        activation_sums[name] += act_norm
                        counts[name] += 1
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(p in name for p in ["mlp", "up_proj", "gate_proj", "fc"]):
                    hooks.append(module.register_forward_hook(make_hook(name)))
        
        model.eval()
        samples_seen = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if samples_seen >= self.num_calibration_samples:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    model(input_ids, attention_mask=attention_mask)
                    batch_size = input_ids.size(0)
                else:
                    inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    model(inputs)
                    batch_size = inputs.size(0)
                
                samples_seen += batch_size
        
        for hook in hooks:
            hook.remove()
        
        # Average activations
        for name in activation_sums:
            self.channel_activations[name] = activation_sums[name] / counts[name]
        
        self._calibrated = True
        logger.info(f"SlimLLM calibration complete. Scored {len(self.channel_activations)} layers.")
    
    def compute_importance_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        SlimLLM holistic importance: activation_norm × weight_contribution
        """
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")
        
        weight = module.weight.data
        
        # Weight contribution per *output* channel (rows)
        weight_importance = weight.abs().sum(dim=1)  # Sum over input dim
        
        if layer_name and layer_name in self.channel_activations:
            act_importance = self.channel_activations[layer_name].to(weight.device)
            if act_importance.shape[0] == weight_importance.shape[0]:
                importance = act_importance * weight_importance
            else:
                importance = weight_importance
        else:
            importance = weight_importance
        
        return importance
    
    def get_structured_scores(
        self,
        module: nn.Module,
        inputs: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
        dim: int = 1,
    ) -> torch.Tensor:
        if not hasattr(module, "weight"):
            raise ValueError(f"Module {module} does not have weights")

        if dim == 0:
            # Natural SlimLLM score: per-output-channel holistic importance
            return self.compute_importance_scores(module, inputs, layer_name)

        if dim == 1:
            # Column importance: fall back to weight column norms (input-channel contribution).
            weight = module.weight.data  # [out_features, in_features]
            return weight.abs().sum(dim=0).detach()

        raise ValueError(f"Invalid dim={dim}; expected 0 (rows) or 1 (cols).")

