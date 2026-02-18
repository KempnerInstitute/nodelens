"""
Rayleigh Quotient alignment metric implementation.

This metric measures how well neural network weights align with the
principal components of their input activations.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("rayleigh_quotient", aliases=["rq", "RQ"])
class RayleighQuotient(BaseMetric):
    """
    Rayleigh Quotient alignment metric.

    Computes the proportion of variance in input activations that is
    captured by each neuron's weight vector. This measures how well
    the weights align with the covariance structure of the inputs.

    For weight vector w and input covariance C:
        RQ(w) = (w^T C w) / (w^T w)

    When relative=True (default), normalizes by trace(C) to get a
    proportion of total variance.

    Optimization Options:
        use_jit: bool = False - Use JIT-compiled RQ computation (20-50% faster)
        use_gpu_acceleration: bool = False - Use GPU-accelerated covariance
    """

    # Class-level set to track warned dimension pairs (avoid spam)
    _warned_dim_pairs: set = set()

    def __init__(
        self,
        relative: bool = True,
        min_samples: int = 2,
        scale_by_norm: bool = False,
        class_conditioned_targets: Optional[torch.Tensor] = None,
        regularization: float = 1e-6,
        **config: Any,
    ):
        """
        Initialize the Rayleigh Quotient metric.

        Args:
            relative: Whether to normalize by trace(C) for relative alignment
            min_samples: Minimum samples required for covariance computation
            scale_by_norm: Whether to scale covariance by its Frobenius norm
            regularization: Small value added to diagonal for numerical stability (default: 1e-6)
            use_jit: bool = False - Use JIT-compiled RQ computation
            use_gpu_acceleration: bool = False - Use GPU-accelerated covariance
            **config: Additional configuration parameters
        """
        super().__init__(**config)
        self.relative = relative
        self.min_samples = min_samples
        self.scale_by_norm = scale_by_norm
        self.regularization = regularization
        # Optional: class-conditioned covariance support (targets provided at compute time preferred)
        self._cc_targets = class_conditioned_targets

        # JIT-specific function reference
        self._jit_rq_compute = None
        if self._use_jit:
            jit_fn = self._get_jit_function("rayleigh_quotient")
            if jit_fn is not None:
                self._jit_rq_compute = jit_fn
                logger.info("RayleighQuotient: Using JIT-compiled computation")

    @property
    def requires_inputs(self) -> bool:
        return True

    @property
    def requires_weights(self) -> bool:
        return True

    @property
    def requires_outputs(self) -> bool:
        return False

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        covariance: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute Rayleigh Quotient values for each neuron.

        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            covariance: Pre-computed covariance matrix (optional, overrides inputs)
            **kwargs: Additional parameters

        Returns:
            RQ values for each output neuron [output_features]
        """
        if weights is None:
            raise ValueError("RayleighQuotient requires weights")

        # Use pre-computed covariance if provided (Streaming mode)
        if covariance is not None:
            return self._compute_from_covariance(covariance, weights)

        if inputs is None:
            raise ValueError("RayleighQuotient requires inputs (or pre-computed covariance)")

        # Handle patchwise inputs (3D tensors)
        if inputs.ndim == 3:
            # Compute patchwise RQ
            return self._compute_patchwise(inputs, weights, **kwargs)

        # Validate shapes for standard 2D computation
        if inputs.ndim != 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim != 2:
            weights = weights.reshape(weights.shape[0], -1)

        batch_size, input_features = inputs.shape
        output_features, weight_features = weights.shape

        # Check sample size
        if batch_size < self.min_samples:
            logger.warning(f"RQ: Only {batch_size} samples, minimum {self.min_samples} recommended. " "Returning zeros.")
            return torch.zeros(output_features, device=weights.device, dtype=weights.dtype)

        # Check dimension compatibility
        if input_features != weight_features:
            # Only warn once per unique dimension pair to avoid log spam
            dim_pair = (input_features, weight_features)
            if dim_pair not in RayleighQuotient._warned_dim_pairs:
                RayleighQuotient._warned_dim_pairs.add(dim_pair)
                logger.warning(
                    f"RQ: Dimension mismatch - inputs: {input_features}, weights: {weight_features}. "
                    "Truncating to common dimensions. (This warning shown once per dimension pair)"
                )
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_features = min_dim  # Update input_features after truncation

        # Move to appropriate device for computation
        # Force CPU for large input dimensions (covariance matrix would be huge)
        # down_proj has 14336 inputs -> cov is [14336, 14336] = 820MB
        use_cpu = self._should_use_cpu(inputs, weights) or input_features > 8192
        if use_cpu:
            logger.debug("RQ: Moving computation to CPU for large tensors")
            torch.device("cpu")
            inputs = inputs.cpu()
            weights = weights.cpu()

        # Optionally compute class-conditioned covariance and average
        use_class_cond = targets is not None or self._cc_targets is not None
        if use_class_cond:
            tgt = targets if targets is not None else self._cc_targets
            if tgt.ndim > 1:
                tgt = tgt.squeeze()

            # Handle unfolded CNN inputs where each sample becomes multiple patches
            # If inputs has more rows than targets, expand targets to match
            original_batch_size = tgt.shape[0]
            if original_batch_size != batch_size:
                if batch_size % original_batch_size == 0:
                    # Expand targets: each label applies to all patches from that sample
                    num_patches = batch_size // original_batch_size
                    tgt = tgt.repeat_interleave(num_patches)
                    logger.debug(f"RQ: Expanded targets from {original_batch_size} to {batch_size} " f"({num_patches} patches per sample)")
                else:
                    logger.warning(
                        f"RQ: Cannot expand targets {original_batch_size} to match " f"batch_size {batch_size}, falling back to unconditional"
                    )
                    use_class_cond = False

            if use_class_cond:
                classes = torch.unique(tgt)
                cov = torch.zeros(input_features, input_features, device=inputs.device, dtype=inputs.dtype)
                total_weight = 0.0
                for c in classes:
                    mask = tgt == c
                    if mask.sum() < self.min_samples:
                        continue
                    Xc = inputs[mask]
                    Xc_centered = Xc - Xc.mean(dim=0, keepdim=True)
                    cov_c = (Xc_centered.T @ Xc_centered) / max(1, (Xc.shape[0] - 1))
                    weight_c = float(mask.sum())
                    cov += cov_c * weight_c
                    total_weight += weight_c
                if total_weight > 0:
                    cov = cov / total_weight
                else:
                    use_class_cond = False  # Fall back to unconditional

        if not use_class_cond:
            # Use JIT-compiled version if available for simple unconditional case
            if self._jit_rq_compute is not None and not self.scale_by_norm:
                try:
                    rq_values = self._jit_rq_compute(inputs, weights, self.regularization)
                    # JIT version doesn't normalize by trace, do it here if relative
                    if self.relative:
                        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                        cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
                        trace_cov = torch.trace(cov.float())
                        if trace_cov > 1e-12:
                            rq_values = rq_values / trace_cov
                    return rq_values
                except Exception as e:
                    logger.warning(f"JIT RQ failed, falling back to standard: {e}")

            # Use GPU-accelerated covariance if available
            if self._use_gpu_acceleration and self._get_gpu_function("fast_covariance") is not None:
                try:
                    cov = self._get_gpu_function("fast_covariance")(inputs)
                except Exception as e:
                    logger.warning(f"GPU covariance failed, falling back to standard: {e}")
                    inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                    cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)
            else:
                # Compute unconditional covariance matrix
                inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
                cov = torch.matmul(inputs_centered.T, inputs_centered) / (batch_size - 1)

        return self._compute_from_covariance(cov, weights)

    def _compute_from_covariance(self, cov: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Helper to compute RQ from covariance matrix."""
        # Add regularization to diagonal for numerical stability
        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        # Scale covariance by norm if requested
        if self.scale_by_norm:
            cov_norm = torch.norm(cov, p="fro")
            if cov_norm > 0:
                cov = cov / cov_norm

        # Compute RQ for each neuron efficiently
        # w^T C w = sum((w @ C) * w, dim=1)
        wc = torch.matmul(weights, cov)  # [output_features, input_features]
        numerator = torch.sum(wc * weights, dim=1)  # [output_features]

        # w^T w
        denominator = torch.sum(weights * weights, dim=1)  # [output_features]

        # Compute RQ with numerical stability
        eps = 1e-12
        rq_values = torch.zeros_like(numerator)
        valid_mask = denominator > eps
        rq_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

        # Normalize by trace if relative
        if self.relative:
            # Convert to float32 for trace (bfloat16 not supported)
            trace_cov = torch.trace(cov.float())
            if trace_cov > eps:
                rq_values = rq_values / trace_cov
            else:
                logger.warning("RQ: Covariance trace near zero, cannot compute relative RQ")

        # Handle any numerical issues
        rq_values = torch.nan_to_num(rq_values, nan=0.0, posinf=0.0, neginf=0.0)

        return rq_values

    def compute_class_conditioned(
        self, inputs: torch.Tensor, weights: torch.Tensor, targets: torch.Tensor, return_delta_rq: bool = False, **kwargs: Any
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute class-conditioned Rayleigh Quotient.

        For each class c, computes RQ using class-specific covariance Σ_{X|y=c},
        then returns the average across classes weighted by class frequency.

        Optionally also computes ΔRQ = RQ(unconditional) - E[RQ(class-conditioned)],
        which measures how much alignment varies across classes.

        Args:
            inputs: Input activations [batch_size, input_features]
            weights: Layer weights [output_features, input_features]
            targets: Class labels [batch_size]
            return_delta_rq: If True, also return ΔRQ
            **kwargs: Additional parameters

        Returns:
            If return_delta_rq=False: class-conditioned RQ [output_features]
            If return_delta_rq=True: dict with keys 'rq_uncond', 'rq_cond', 'delta_rq'
        """
        # Flatten if needed
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Ensure targets are 1D
        if targets.ndim > 1:
            targets = targets.squeeze()

        batch_size, input_features = inputs.shape
        output_features, weight_features = weights.shape

        # Check compatibility
        if input_features != weight_features:
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_features = min_dim

        device = weights.device

        # Get unique classes
        classes = torch.unique(targets)

        # Compute class-conditioned RQ (weighted average)
        rq_cond_sum = torch.zeros(output_features, device=device)
        total_weight = 0.0

        for c in classes:
            mask = targets == c
            n_c = mask.sum()

            if n_c < self.min_samples:
                logger.warning(f"Class {c}: only {n_c} samples, skipping")
                continue

            # Extract class data
            inputs_c = inputs[mask]

            # Compute class-specific covariance
            inputs_c_centered = inputs_c - inputs_c.mean(dim=0, keepdim=True)
            cov_c = (inputs_c_centered.T @ inputs_c_centered) / max(1, n_c - 1)

            # Add regularization
            if self.regularization > 0:
                cov_c = cov_c + self.regularization * torch.eye(input_features, device=device, dtype=cov_c.dtype)

            # Compute RQ for this class
            wc = torch.matmul(weights, cov_c)
            numerator_c = torch.sum(wc * weights, dim=1)
            denominator_c = torch.sum(weights * weights, dim=1)

            eps = 1e-12
            rq_c = torch.zeros_like(numerator_c)
            valid_mask = denominator_c > eps
            rq_c[valid_mask] = numerator_c[valid_mask] / denominator_c[valid_mask]

            # Normalize by trace if relative
            if self.relative:
                # Convert to float32 for trace (bfloat16 not supported)
                trace_c = torch.trace(cov_c.float())
                if trace_c > eps:
                    rq_c = rq_c / trace_c

            # Weighted sum
            weight_c = n_c.float()
            rq_cond_sum += rq_c * weight_c
            total_weight += weight_c

        # Average across classes
        if total_weight > 0:
            rq_cond = rq_cond_sum / total_weight
        else:
            logger.warning("No valid classes found, returning zeros")
            rq_cond = torch.zeros(output_features, device=device)

        # Clean up numerical issues
        rq_cond = torch.nan_to_num(rq_cond, nan=0.0, posinf=0.0, neginf=0.0)

        if not return_delta_rq:
            return rq_cond

        # Also compute unconditional RQ
        rq_uncond = self.compute(inputs=inputs, weights=weights, **kwargs)

        # Compute ΔRQ
        delta_rq = rq_uncond - rq_cond

        return {"rq_uncond": rq_uncond, "rq_cond": rq_cond, "delta_rq": delta_rq}

    def _compute_patchwise(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        weight_by_variance: bool = True,
        max_patches: int = 64,
        use_vectorized: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute patch-wise RQ for CNN layers.

        Args:
            inputs: Input patches [batch_size, features, num_patches]
            weights: Flattened weights [output_features, features]
            weight_by_variance: Whether to weight patches by their variance
            max_patches: Maximum patches to use (subsample if more)
            use_vectorized: Use vectorized computation (faster, more memory)

        Returns:
            RQ values [output_features]
        """
        batch_size, features, num_patches = inputs.shape
        output_features = weights.shape[0]

        if batch_size < self.min_samples:
            logger.warning(f"Only {batch_size} samples, minimum {self.min_samples} recommended")
            return torch.zeros(output_features, device=weights.device, dtype=weights.dtype)

        # Ensure weight dimensions match
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Subsample patches if too many (memory/speed optimization)
        if num_patches > max_patches:
            logger.debug(f"Subsampling patches: {num_patches} -> {max_patches}")
            indices = torch.randperm(num_patches, device=inputs.device)[:max_patches]
            inputs = inputs[:, :, indices]
            num_patches = max_patches

        # Handle dimension mismatch
        min_dim = min(features, weights.shape[1])
        if features != weights.shape[1]:
            inputs = inputs[:, :min_dim, :]
            weights = weights[:, :min_dim]
            features = min_dim

        # Compute patch variances for weighting
        patch_var = torch.var(inputs, dim=0)  # [features, num_patches]
        patch_weights = patch_var.sum(dim=0) if weight_by_variance else torch.ones(num_patches, device=inputs.device)

        if use_vectorized and num_patches <= 256:
            # VECTORIZED: Compute all patches at once (faster but more memory)
            return self._compute_patchwise_vectorized(inputs, weights, patch_weights)
        else:
            # LOOP: Compute patches one by one (slower but less memory)
            return self._compute_patchwise_loop(inputs, weights, patch_weights)

    def _compute_patchwise_vectorized(self, inputs: torch.Tensor, weights: torch.Tensor, patch_weights: torch.Tensor) -> torch.Tensor:
        """
        Vectorized patchwise RQ computation - no explicit loop.

        Uses einsum for efficient batch computation over patches.
        Memory: O(num_patches * features^2) for covariances
        """
        batch_size, features, num_patches = inputs.shape
        output_features = weights.shape[0]
        device = weights.device
        eps = 1e-12

        # Center inputs: [B, F, P]
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)

        # Compute covariance for each patch using einsum
        # cov[p] = X_centered[:, :, p].T @ X_centered[:, :, p] / (B-1)
        # = einsum('bi,bj->ij' for each patch p)
        # Batched: einsum('bfp,bgp->fgp') gives [F, F, P]
        all_covs = torch.einsum("bfp,bgp->fgp", inputs_centered, inputs_centered) / (batch_size - 1)

        # Add regularization
        if self.regularization > 0:
            eye = torch.eye(features, device=device, dtype=all_covs.dtype)
            all_covs = all_covs + self.regularization * eye.unsqueeze(-1)

        # Compute w^T C w for all patches and neurons at once
        # weights: [N, F], all_covs: [F, G, P] where F=G
        # numerator[n, p] = sum_f sum_g w[n,f] * cov[f,g,p] * w[n,g]
        # = einsum('nf,fgp,ng->np')
        wc = torch.einsum("nf,fgp->ngp", weights, all_covs)  # [N, G, P]
        numerator = torch.einsum("ngp,ng->np", wc, weights)  # [N, P]

        # Compute w^T w (same for all patches)
        denominator = (weights**2).sum(dim=1)  # [N]

        # Compute RQ per patch
        patch_rq = torch.zeros(output_features, num_patches, device=device)
        valid_mask = denominator > eps
        patch_rq[valid_mask, :] = numerator[valid_mask, :] / denominator[valid_mask].unsqueeze(1)

        # Normalize by trace if relative
        if self.relative:
            # trace of each patch covariance
            traces = torch.diagonal(all_covs, dim1=0, dim2=1).sum(dim=0)  # [P]
            traces = torch.clamp(traces, min=eps)
            patch_rq = patch_rq / traces.unsqueeze(0)

        # Weighted average across patches
        patch_weights = torch.clamp(patch_weights, min=0)
        total_weight = patch_weights.sum()
        if total_weight > eps:
            final_rq = (patch_rq * patch_weights.unsqueeze(0)).sum(dim=1) / total_weight
        else:
            final_rq = patch_rq.mean(dim=1)

        return torch.nan_to_num(final_rq, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_patchwise_loop(self, inputs: torch.Tensor, weights: torch.Tensor, patch_weights: torch.Tensor) -> torch.Tensor:
        """
        Loop-based patchwise RQ computation - lower memory usage.

        Memory: O(features^2) for one covariance at a time
        """
        batch_size, features, num_patches = inputs.shape
        output_features = weights.shape[0]
        device = weights.device
        eps = 1e-12

        weighted_rq_sum = torch.zeros(output_features, device=device)
        total_weight = 0.0

        for p in range(num_patches):
            patch_data = inputs[:, :, p]  # [batch_size, features]
            patch_data_centered = patch_data - patch_data.mean(dim=0, keepdim=True)
            patch_cov = torch.matmul(patch_data_centered.T, patch_data_centered) / (batch_size - 1)

            if self.regularization > 0:
                patch_cov = patch_cov + self.regularization * torch.eye(features, device=device, dtype=patch_cov.dtype)

            if self.scale_by_norm:
                cov_norm = torch.norm(patch_cov, p="fro")
                if cov_norm > 0:
                    patch_cov = patch_cov / cov_norm

            # Compute RQ
            wc = torch.matmul(weights, patch_cov)
            numerator = torch.sum(wc * weights, dim=1)
            denominator = torch.sum(weights * weights, dim=1)

            patch_rq = torch.zeros_like(numerator)
            valid_mask = denominator > eps
            patch_rq[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

            if self.relative:
                trace = torch.trace(patch_cov.float())
                if trace > eps:
                    patch_rq = patch_rq / trace

            pw = patch_weights[p].item()
            weighted_rq_sum += patch_rq * pw
            total_weight += pw

        if total_weight > 0:
            final_rq = weighted_rq_sum / total_weight
        else:
            final_rq = weighted_rq_sum

        return torch.nan_to_num(final_rq, nan=0.0, posinf=0.0, neginf=0.0)


@register_metric("rq_patchwise")
class PatchWiseRayleighQuotient(RayleighQuotient):
    """
    Patch-wise variant of Rayleigh Quotient for convolutional layers.

    This variant computes RQ separately for each spatial location (patch)
    and then aggregates the results, weighted by patch variance.
    """

    def __init__(self, relative: bool = True, min_samples: int = 2, scale_by_norm: bool = False, weight_by_variance: bool = True, **config: Any):
        """
        Initialize patch-wise RQ metric.

        Args:
            relative: Whether to normalize by trace(C)
            min_samples: Minimum samples for covariance
            scale_by_norm: Whether to scale covariance by norm
            weight_by_variance: Whether to weight patches by their variance
            **config: Additional configuration
        """
        super().__init__(relative, min_samples, scale_by_norm, **config)
        self.weight_by_variance = weight_by_variance

    def compute(
        self, inputs: Optional[torch.Tensor] = None, weights: Optional[torch.Tensor] = None, outputs: Optional[torch.Tensor] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute patch-wise RQ for convolutional inputs.

        Args:
            inputs: Input patches [batch_size, features, num_patches]
            weights: Convolutional weights [output_channels, input_features]
            outputs: Not used

        Returns:
            RQ values for each output channel [output_channels]
        """
        if inputs is None or weights is None:
            raise ValueError("PatchWiseRQ requires both inputs and weights")

        # Handle different input formats
        if inputs.ndim == 2:
            # Regular RQ if not patch format
            return super().compute(inputs, weights, outputs, **kwargs)

        if inputs.ndim != 3:
            raise ValueError(f"PatchWiseRQ expects 3D input, got {inputs.ndim}D")

        batch_size, features, num_patches = inputs.shape
        output_channels = weights.shape[0]

        if batch_size < self.min_samples:
            return torch.zeros(output_channels, device=weights.device, dtype=weights.dtype)

        # Compute variance for each patch
        patch_var = torch.var(inputs, dim=0, keepdim=False)  # [features, num_patches]
        patch_total_var = patch_var.sum(dim=0)  # [num_patches]

        # Initialize results
        all_patch_rq = []
        all_patch_weights = []

        # Compute RQ for each patch
        for p in range(num_patches):
            patch_data = inputs[:, :, p]  # [batch_size, features]

            # Compute RQ for this patch
            patch_rq = super().compute(patch_data, weights, None)

            # Weight by patch variance if requested
            if self.weight_by_variance:
                patch_weight = patch_total_var[p]
            else:
                patch_weight = 1.0

            all_patch_rq.append(patch_rq * patch_weight)
            all_patch_weights.append(patch_weight)

        # Aggregate across patches
        total_weight = sum(all_patch_weights)
        if total_weight > 0:
            final_rq = torch.stack(all_patch_rq).sum(dim=0) / total_weight
        else:
            final_rq = torch.zeros(output_channels, device=weights.device, dtype=weights.dtype)

        return final_rq


@register_metric("rq_fast", aliases=["rq_gap", "rayleigh_quotient_fast"])
class FastRayleighQuotient(RayleighQuotient):
    """
    Fast Rayleigh Quotient approximation for CNNs using Global Average Pooling.

    Instead of unfolding patches, this version uses GAP to reduce spatial dimensions
    to get a channel-wise covariance, then computes RQ on the channel dimension.

    This is an APPROXIMATION but is:
    - 10-100x faster than patchwise RQ
    - Uses O(C^2) memory instead of O((C*k*k)^2)
    - Better for early conv layers with large spatial dimensions

    For inputs [B, C_in, H, W] and weights [C_out, C_in, k, k]:
    1. Apply GAP: inputs -> [B, C_in]
    2. Sum weights over kernel: weights -> [C_out, C_in]
    3. Compute standard RQ on channel dimension

    Use cases:
    - Quick experiments
    - Early conv layers (layer1, layer2) where exact RQ is too slow
    - When channel-level importance is sufficient

    Example:
        >>> rq_fast = FastRayleighQuotient()
        >>> # inputs: [batch, channels, height, width]
        >>> scores = rq_fast.compute(inputs=inputs, weights=conv.weight)
    """

    def __init__(self, relative: bool = True, regularization: float = 1e-6, **config: Any):
        """
        Initialize fast RQ metric.

        Args:
            relative: Whether to normalize by trace(C)
            regularization: Regularization for covariance
            **config: Additional configuration
        """
        super().__init__(relative=relative, regularization=regularization, **config)

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute fast RQ using GAP for CNN inputs.

        Args:
            inputs: Input activations [B, C_in, H, W] or [B, C_in]
            weights: Conv weights [C_out, C_in, k_H, k_W] or [C_out, C_in]
            outputs: Not used

        Returns:
            RQ values for each output channel [C_out]
        """
        if weights is None:
            raise ValueError("FastRQ requires weights")
        if inputs is None:
            raise ValueError("FastRQ requires inputs")

        # Handle CNN inputs: [B, C_in, H, W] -> GAP -> [B, C_in]
        if inputs.ndim == 4:
            # Global Average Pooling over spatial dimensions
            inputs = inputs.mean(dim=(2, 3))  # [B, C_in]
        elif inputs.ndim == 3:
            # Patchwise input [B, F, P] -> average over patches
            inputs = inputs.mean(dim=2)  # [B, F]

        # Handle Conv weights: [C_out, C_in, k_H, k_W] -> sum over kernel -> [C_out, C_in]
        if weights.ndim == 4:
            # Sum over kernel dimensions to get channel-wise weights
            weights = weights.sum(dim=(2, 3))  # [C_out, C_in]
        elif weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Now use standard 2D RQ computation
        return super().compute(inputs=inputs, weights=weights, **kwargs)


@register_metric("rq_spatial", aliases=["rayleigh_quotient_spatial"])
class SpatialRayleighQuotient(RayleighQuotient):
    """
    Spatial Rayleigh Quotient that treats spatial locations as additional samples.

    Instead of unfolding into patches, this version:
    1. Reshapes [B, C, H, W] -> [B*H*W, C]
    2. Computes channel-wise covariance over all spatial locations
    3. Computes RQ using kernel-summed weights

    This is FASTER than patchwise and captures spatial variation as sample diversity.

    Trade-off vs GAP:
    - GAP averages spatially, losing spatial variance information
    - Spatial treats each location as a sample, preserving variance

    Trade-off vs Patchwise:
    - Patchwise captures kernel-local correlations
    - Spatial captures global channel correlations only

    Example:
        >>> rq_spatial = SpatialRayleighQuotient()
        >>> scores = rq_spatial.compute(inputs=inputs, weights=conv.weight)
    """

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute spatial RQ.

        Args:
            inputs: [B, C_in, H, W] or [B, C_in]
            weights: [C_out, C_in, k_H, k_W] or [C_out, C_in]
            outputs: Not used

        Returns:
            RQ values [C_out]
        """
        if weights is None:
            raise ValueError("SpatialRQ requires weights")
        if inputs is None:
            raise ValueError("SpatialRQ requires inputs")

        # Handle CNN inputs: [B, C, H, W] -> [B*H*W, C]
        if inputs.ndim == 4:
            B, C, H, W = inputs.shape
            # Permute to [B, H, W, C] then reshape to [B*H*W, C]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        elif inputs.ndim == 3:
            # Patchwise input [B, F, P] -> transpose and reshape [B*P, F]
            B, F, P = inputs.shape
            inputs = inputs.permute(0, 2, 1).reshape(-1, F)

        # Handle Conv weights: sum over kernel
        if weights.ndim == 4:
            weights = weights.sum(dim=(2, 3))
        elif weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        # Standard 2D RQ on [samples, channels]
        return super().compute(inputs=inputs, weights=weights, **kwargs)
