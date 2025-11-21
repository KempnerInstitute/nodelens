"""
Pairwise Gaussian redundancy metric.

Computes redundancy between neurons using Gaussian approximation.
For Gaussian outputs Y_i = w_i^T X and Y_j = w_j^T X:
    R(Y_i, Y_j) = -0.5 * log(1 - ρ²)
where ρ is correlation in the Σ_X space.
"""

import logging
from typing import Any, Optional, Union, List

import torch

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("pairwise_redundancy_gaussian")
class PairwiseRedundancyGaussian(BaseMetric):
    """
    Compute per-neuron redundancy via pairwise Gaussian mutual information.

    For each neuron, computes average redundancy with K sampled partner neurons.
    Redundancy proxy: I(Y_i; Y_j) = -0.5 * log(1 - ρ²)

    This is a target-free redundancy measure based on correlation in the
    input covariance space.

    Example:
        >>> redundancy_metric = PairwiseRedundancyGaussian(num_pairs=10)
        >>> redundancy = redundancy_metric.compute(inputs, weights)
        >>> print(redundancy.shape)  # [num_neurons]
    """

    def __init__(
        self,
        num_pairs: int = 10,
        sampling_strategy: str = "random",
        mode: str = "output_based",  # 'output_based' or 'covariance_based'
        regularization: float = 1e-6,
        **config: Any,
    ):
        """
        Initialize pairwise redundancy metric.

        Args:
            num_pairs: Number of partner neurons to sample per neuron
            sampling_strategy: How to sample pairs ('random', 'nearest', 'all')
            mode: Computation mode
                - 'output_based': Compute from outputs directly (FAST, O(B·N²))
                - 'covariance_based': Compute via input covariance (SLOW, O(B·D²·N·K))
            regularization: Small value added to covariance diagonal
            **config: Additional configuration
        """
        super().__init__(**config)
        self.num_pairs = num_pairs
        self.sampling_strategy = sampling_strategy
        self.mode = mode
        self.regularization = regularization

    @property
    def requires_inputs(self) -> bool:
        return self.mode == "covariance_based"

    @property
    def requires_weights(self) -> bool:
        return self.mode == "covariance_based"

    @property
    def requires_outputs(self) -> bool:
        return self.mode == "output_based"

    def compute(
        self,
        inputs: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        target_indices: Optional[Union[List[int], torch.Tensor]] = None,
        allowed_partners: Optional[Union[List[int], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute per-neuron redundancy scores.

        Args:
            inputs: Input activations [batch_size, input_features] (for covariance_based)
            weights: Layer weights [num_neurons, input_features] (for covariance_based)
            outputs: Layer outputs [batch_size, num_neurons] (for output_based)
            target_indices: Optional list of indices to compute scores for (others 0)
            allowed_partners: Optional list of indices to restrict partner sampling to
            **kwargs: Additional parameters

        Returns:
            Per-neuron redundancy scores [num_neurons]
        """
        # Route to appropriate implementation
        if self.mode == "output_based":
            if outputs is None:
                # Compute outputs if not provided
                if inputs is None or weights is None:
                    raise ValueError("output_based mode needs outputs OR (inputs + weights)")
                # Flatten inputs if needed
                if inputs.ndim > 2:
                    inputs = inputs.reshape(inputs.shape[0], -1)
                if weights.ndim > 2:
                    weights = weights.reshape(weights.shape[0], -1)
                outputs = inputs @ weights.T

            return self._compute_output_based(outputs, target_indices, allowed_partners)

        elif self.mode == "covariance_based":
            if inputs is None or weights is None:
                raise ValueError("covariance_based mode requires inputs and weights")
            return self._compute_covariance_based(inputs, weights, target_indices, allowed_partners)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _compute_output_based(
        self, 
        outputs: torch.Tensor, 
        target_indices: Optional[Union[List[int], torch.Tensor]] = None,
        allowed_partners: Optional[Union[List[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute redundancy from outputs directly (FAST!).
        Optimized to only compute necessary correlations if indices are provided.
        """
        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)

        B, N = outputs.shape

        if B < 2:
            logger.warning("Output-based redundancy needs B >= 2, returning zeros")
            return torch.zeros(N, device=outputs.device)

        # Prepare indices
        if target_indices is not None:
            if not isinstance(target_indices, torch.Tensor):
                target_indices = torch.tensor(target_indices, device=outputs.device, dtype=torch.long)
            targets = target_indices
        else:
            targets = torch.arange(N, device=outputs.device)

        if allowed_partners is not None:
            if not isinstance(allowed_partners, torch.Tensor):
                allowed_partners = torch.tensor(allowed_partners, device=outputs.device, dtype=torch.long)
            sources = allowed_partners
        else:
            sources = torch.arange(N, device=outputs.device)

        # Center outputs
        # We need means for everyone to center correctly
        Y_mean = outputs.mean(dim=0, keepdim=True)
        Y_centered = outputs - Y_mean

        # Normalize outputs (standard deviation)
        # We normalize early to make matrix multiplication yield correlations directly
        Y_var = (Y_centered ** 2).sum(dim=0) / max(1, B - 1)
        Y_std = torch.sqrt(Y_var + 1e-12)
        Y_norm = Y_centered / Y_std

        # Extract subsets
        Y_targets = Y_norm[:, targets]  # [B, T]
        Y_sources = Y_norm[:, sources]  # [B, S]

        # Compute correlation matrix [T, S]
        # Corr = (X^T @ Y) / (B-1)
        # Since we already normalized, this is the correlation matrix directly
        corr_matrix = (Y_targets.T @ Y_sources) / max(1, B - 1)

        # Clip to valid range
        corr_matrix = torch.clamp(corr_matrix, -0.9999, 0.9999)

        # Redundancy: I(Yi; Yj) = -0.5·log(1 - ρ²)
        rho_sq = corr_matrix**2
        R_matrix = -0.5 * torch.log(1 - rho_sq + 1e-8)  # [T, S]

        # Mask self-redundancy (diagonal) if targets and sources overlap
        # We need to know which (t, s) pairs correspond to the same neuron
        # targets[t] == sources[s]
        # This can be slow if we check every pair.
        # Heuristic: if targets and sources are same (common case), mask diagonal
        if target_indices is None and allowed_partners is None:
            # Simple diagonal mask when targets == sources == all neurons
            R_matrix.fill_diagonal_(0)
            mask = torch.eye(len(targets), len(sources), dtype=torch.bool, device=outputs.device)
        else:
            # Create mask for identity pairs
            # t_grid = targets.unsqueeze(1)  # [T, 1]
            # s_grid = sources.unsqueeze(0)  # [1, S]
            # mask = (t_grid == s_grid)
            # R_matrix[mask] = 0
            # This works but allocates [T, S] bool matrix.
            # If T, S are small, it's fine. If T, S are huge (full layer), it's fine (same as before).
            t_grid = targets.unsqueeze(1)
            s_grid = sources.unsqueeze(0)
            mask = (t_grid == s_grid)
            R_matrix.masked_fill_(mask, 0.0)

        # Compute average redundancy per target
        # Result will be mapped back to full N-sized tensor
        redundancy_full = torch.zeros(N, device=outputs.device)
        
        num_targets = len(targets)
        num_sources = len(sources)
        
        # Determine how to aggregate based on sampling strategy
        if self.sampling_strategy == "all":
            # Average over all valid sources (excluding self)
            # Valid count per row: S - 1 (if self in sources) or S (if self not in sources)
            # We already zeroed self, so sum is correct.
            # Count is tricky.
            
            row_sums = R_matrix.sum(dim=1) # [T]
            
            # Calculate denominator (count of non-self partners)
            # is_in_source[t] = (targets[t] in sources)
            # Using the mask computed earlier: mask.sum(dim=1) is 1 if target[t] in sources, else 0
            self_counts = mask.sum(dim=1).float()
            denominators = num_sources - self_counts
            denominators = torch.clamp(denominators, min=1)
            
            scores = row_sums / denominators
            
        else:
            # Sampling (Random or Nearest)
            # Since we computed the full [T, S] matrix (which is efficient if T, S << N),
            # we can just sample from the rows of R_matrix.
            
            scores = torch.zeros(num_targets, device=outputs.device)
            
            for t_idx in range(num_targets):
                row = R_matrix[t_idx] # [S]
                
                # Exclude self
                # We can use the mask row
                valid_mask = ~mask[t_idx]
                valid_indices = torch.nonzero(valid_mask).squeeze()
                
                num_valid = len(valid_indices)
                if num_valid == 0:
                    continue
                    
                if self.sampling_strategy == "random":
                    num_sample = min(self.num_pairs, num_valid)
                    perm = torch.randperm(num_valid, device=outputs.device)[:num_sample]
                    sampled_indices = valid_indices[perm]
                    scores[t_idx] = row[sampled_indices].mean()
                    
                elif self.sampling_strategy == "nearest":
                    # "Nearest" usually means spatial/index distance.
                    # Here indices are targets[t_idx] and sources[valid_indices]
                    # Abs dist: |target_id - source_id|
                    target_id = targets[t_idx]
                    source_ids = sources[valid_indices]
                    dists = (target_id - source_ids).abs()
                    
                    num_sample = min(self.num_pairs, num_valid)
                    _, topk_indices = torch.topk(dists, num_sample, largest=False)
                    # topk_indices are indices into valid_indices
                    final_indices = valid_indices[topk_indices]
                    scores[t_idx] = row[final_indices].mean()

        # Scatter scores back to full tensor
        redundancy_full[targets] = scores

        return redundancy_full

    def _compute_covariance_based(
        self, 
        inputs: torch.Tensor, 
        weights: torch.Tensor,
        target_indices: Optional[Union[List[int], torch.Tensor]] = None,
        allowed_partners: Optional[Union[List[int], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute redundancy via input covariance.
        """

        # Flatten if needed
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        batch_size, input_features = inputs.shape
        num_neurons, weight_features = weights.shape

        # Check compatibility
        if input_features != weight_features:
            min_dim = min(input_features, weight_features)
            inputs = inputs[:, :min_dim]
            weights = weights[:, :min_dim]
            input_features = min_dim

        # Compute input covariance
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        cov = (inputs_centered.T @ inputs_centered) / max(1, batch_size - 1)

        # Add regularization
        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(input_features, device=cov.device, dtype=cov.dtype)

        # Compute redundancy for each neuron
        redundancy = torch.zeros(num_neurons, device=weights.device, dtype=weights.dtype)

        # Determine loop range
        if target_indices is not None:
            if isinstance(target_indices, torch.Tensor):
                loop_indices = target_indices.tolist()
            else:
                loop_indices = target_indices
        else:
            loop_indices = range(num_neurons)

        # Determine available partners
        if allowed_partners is not None:
            if isinstance(allowed_partners, torch.Tensor):
                partner_pool = allowed_partners.tolist()
            else:
                partner_pool = allowed_partners
        else:
            partner_pool = None

        for i in loop_indices:
            # Sample partner neurons
            partner_indices = self._sample_partners_fast(i, num_neurons, partner_pool)

            if len(partner_indices) == 0:
                continue

            # Compute correlation with each partner
            correlations = []
            for j in partner_indices:
                rho_sq = self._compute_correlation_squared(weights[i], weights[j], cov)
                correlations.append(rho_sq)

            # Average redundancy with partners
            if correlations:
                # I(Y_i; Y_j) = -0.5 * log(1 - ρ²)
                correlations_tensor = torch.stack(correlations)
                # Clip to avoid log(0)
                correlations_tensor = torch.clamp(correlations_tensor, max=0.9999)
                redundancy_values = -0.5 * torch.log(1 - correlations_tensor + 1e-8)
                redundancy[i] = redundancy_values.mean()

        return redundancy

    def _sample_partners_fast(self, neuron_idx: int, num_neurons: int, pool: Optional[List[int]] = None) -> torch.Tensor:
        """
        Sample partner neurons for redundancy computation.

        Args:
            neuron_idx: Index of current neuron
            num_neurons: Total number of neurons
            pool: Optional list of allowed partner indices

        Returns:
            Indices of partner neurons
        """
        # Define available pool
        if pool is None:
            available = list(range(num_neurons))
        else:
            available = list(pool)
            
        # Exclude self if present
        if neuron_idx in available:
            available.remove(neuron_idx)
            
        if not available:
            return torch.tensor([], dtype=torch.long)

        if self.sampling_strategy == "all":
            # Use all other neurons
            return torch.tensor(available, dtype=torch.long)

        elif self.sampling_strategy == "random":
            # Random sample
            num_to_sample = min(self.num_pairs, len(available))
            if num_to_sample == 0:
                return torch.tensor([], dtype=torch.long)

            # If pool is small, just take it
            if len(available) <= self.num_pairs:
                return torch.tensor(available, dtype=torch.long)
                
            indices = torch.randperm(len(available))[:num_to_sample]
            return torch.tensor([available[i] for i in indices], dtype=torch.long)

        elif self.sampling_strategy == "nearest":
            # Sample nearby indices (assumes some ordering)
            # This logic works best when pool is contiguous, but let's adapt
            num_to_sample = min(self.num_pairs, len(available))
            if num_to_sample == 0:
                return torch.tensor([], dtype=torch.long)

            # Get closest indices by integer distance
            distances = torch.abs(torch.tensor(available) - neuron_idx)
            _, nearest_indices = torch.topk(distances, num_to_sample, largest=False)
            return torch.tensor([available[i] for i in nearest_indices], dtype=torch.long)

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _compute_correlation_squared(self, w_i: torch.Tensor, w_j: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute squared correlation between two neurons in covariance space.

        Args:
            w_i: Weight vector of neuron i [features]
            w_j: Weight vector of neuron j [features]
            cov: Input covariance matrix [features, features]

        Returns:
            ρ² = (w_i^T Σ w_j)² / [(w_i^T Σ w_i)(w_j^T Σ w_j)]
        """
        # Compute variances
        var_i = w_i @ cov @ w_i  # scalar
        var_j = w_j @ cov @ w_j  # scalar

        # Compute covariance
        cov_ij = w_i @ cov @ w_j  # scalar

        # Compute squared correlation
        denominator = var_i * var_j

        if denominator < 1e-12:
            return torch.tensor(0.0, device=w_i.device, dtype=w_i.dtype)

        rho_sq = (cov_ij**2) / denominator

        # Clip to [0, 1]
        rho_sq = torch.clamp(rho_sq, min=0.0, max=1.0)

        return rho_sq

    def compute_pairwise_matrix(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute full pairwise redundancy matrix.

        Args:
            inputs: Input activations [batch_size, features]
            weights: Layer weights [num_neurons, features]

        Returns:
            Redundancy matrix [num_neurons, num_neurons]
            where R[i,j] = redundancy between neurons i and j
        """
        if inputs.ndim > 2:
            inputs = inputs.reshape(inputs.shape[0], -1)
        if weights.ndim > 2:
            weights = weights.reshape(weights.shape[0], -1)

        batch_size = inputs.shape[0]
        num_neurons = weights.shape[0]

        # Compute covariance
        inputs_centered = inputs - inputs.mean(dim=0, keepdim=True)
        cov = (inputs_centered.T @ inputs_centered) / max(1, batch_size - 1)

        if self.regularization > 0:
            cov = cov + self.regularization * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)

        # Compute pairwise redundancy
        redundancy_matrix = torch.zeros(num_neurons, num_neurons, device=weights.device, dtype=weights.dtype)

        for i in range(num_neurons):
            for j in range(i + 1, num_neurons):
                rho_sq = self._compute_correlation_squared(weights[i], weights[j], cov)
                # Clip correlation
                rho_sq = torch.clamp(rho_sq, max=0.9999)
                redundancy = -0.5 * torch.log(1 - rho_sq + 1e-8)

                # Symmetric
                redundancy_matrix[i, j] = redundancy
                redundancy_matrix[j, i] = redundancy

        return redundancy_matrix
