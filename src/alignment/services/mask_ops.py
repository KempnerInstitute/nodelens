"""
Mask Operations for pruning and dropout.

Utilities for creating and manipulating masks for structured
and unstructured pruning.
"""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


class MaskOperations:
    """
    Utilities for creating and manipulating pruning masks.

    Provides methods for:
    - Creating structured (neuron/channel) masks
    - Creating unstructured (weight-level) masks
    - Expanding masks to match weight shapes
    - Applying masks to modules
    """

    @staticmethod
    def create_structured_mask(scores: torch.Tensor, amount: float, mode: str = "low", min_keep: int = 1) -> torch.Tensor:
        """
        Create structured mask (neuron/channel level) from importance scores.

        Args:
            scores: Importance scores [num_neurons]
            amount: Fraction to prune (0 to 1)
            mode: 'low' (prune low scores), 'high' (prune high scores), 'random'
            min_keep: Minimum number of neurons to keep

        Returns:
            Binary mask [num_neurons] where 1=keep, 0=prune
        """
        num_neurons = scores.shape[0]
        num_to_prune = max(0, int(num_neurons * amount))
        num_to_keep = max(min_keep, num_neurons - num_to_prune)

        mask = torch.ones_like(scores, dtype=torch.bool)

        if num_to_prune == 0:
            return mask

        if mode == "low":
            # Prune lowest scores
            _, indices = torch.topk(scores, num_to_keep, largest=True)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[indices] = True

        elif mode == "high":
            # Prune highest scores
            _, indices = torch.topk(scores, num_to_keep, largest=False)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[indices] = True

        elif mode == "random":
            # Random pruning
            indices = torch.randperm(num_neurons)[:num_to_keep]
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[indices] = True

        else:
            raise ValueError(f"Unknown pruning mode: {mode}")

        logger.debug(f"Created structured mask: {mask.sum()}/{num_neurons} neurons kept")
        return mask

    @staticmethod
    def create_unstructured_mask(scores: torch.Tensor, amount: float, mode: str = "low") -> torch.Tensor:
        """
        Create unstructured mask (weight level) from importance scores.

        Args:
            scores: Importance scores (any shape)
            amount: Fraction to prune
            mode: 'low', 'high', or 'random'

        Returns:
            Binary mask same shape as scores
        """
        num_weights = scores.numel()
        num_to_prune = int(num_weights * amount)
        num_to_keep = num_weights - num_to_prune

        if num_to_prune == 0:
            return torch.ones_like(scores, dtype=torch.bool)

        # Flatten for processing
        scores_flat = scores.flatten()

        if mode == "low":
            _, indices = torch.topk(scores_flat, num_to_keep, largest=True)
        elif mode == "high":
            _, indices = torch.topk(scores_flat, num_to_keep, largest=False)
        elif mode == "random":
            indices = torch.randperm(num_weights)[:num_to_keep]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create flat mask
        mask_flat = torch.zeros_like(scores_flat, dtype=torch.bool)
        mask_flat[indices] = True

        # Reshape to original shape
        mask = mask_flat.reshape(scores.shape)

        logger.debug(f"Created unstructured mask: {mask.sum()}/{num_weights} weights kept")
        return mask

    @staticmethod
    def expand_neuron_mask_to_weights(neuron_mask: torch.Tensor, weight_shape: Tuple[int, ...], dim: int = 0) -> torch.Tensor:
        """
        Expand neuron/channel mask to full weight tensor.

        Args:
            neuron_mask: Binary mask [num_neurons]
            weight_shape: Target weight shape, e.g., [out, in] or [out, in, k, k]
            dim: Dimension corresponding to neurons (usually 0 for output neurons)

        Returns:
            Expanded mask matching weight_shape
        """
        # Create base mask
        mask = torch.ones(weight_shape, dtype=torch.bool, device=neuron_mask.device)

        # Apply neuron mask along specified dimension
        if dim == 0:
            # Mask output neurons/channels
            for i, keep in enumerate(neuron_mask):
                if not keep:
                    if len(weight_shape) == 2:  # Linear: [out, in]
                        mask[i, :] = False
                    elif len(weight_shape) == 4:  # Conv2d: [out, in, k, k]
                        mask[i, :, :, :] = False
                    elif len(weight_shape) == 3:  # Conv1d: [out, in, k]
                        mask[i, :, :] = False

        elif dim == 1:
            # Mask input neurons/channels (less common)
            for i, keep in enumerate(neuron_mask):
                if not keep:
                    if len(weight_shape) == 2:
                        mask[:, i] = False
                    elif len(weight_shape) == 4:
                        mask[:, i, :, :] = False
                    elif len(weight_shape) == 3:
                        mask[:, i, :] = False

        return mask

    @staticmethod
    def apply_mask_to_weights(weights: torch.Tensor, mask: torch.Tensor, mode: str = "multiply") -> torch.Tensor:
        """
        Apply mask to weights.

        Args:
            weights: Weight tensor
            mask: Binary mask (same shape as weights)
            mode: 'multiply' or 'zero'

        Returns:
            Masked weights
        """
        if mask.shape != weights.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match weights shape {weights.shape}")

        if mode == "multiply":
            return weights * mask.float()
        elif mode == "zero":
            masked_weights = weights.clone()
            masked_weights[~mask] = 0.0
            return masked_weights
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def get_mask_statistics(mask: torch.Tensor) -> dict:
        """
        Compute statistics about a mask.

        Args:
            mask: Binary mask

        Returns:
            Dictionary with statistics
        """
        total = mask.numel()
        kept = mask.sum().item()
        pruned = total - kept

        return {
            "total_elements": total,
            "kept_elements": kept,
            "pruned_elements": pruned,
            "sparsity": pruned / total if total > 0 else 0.0,
            "density": kept / total if total > 0 else 0.0,
        }

    @staticmethod
    def combine_masks(masks: list, operation: str = "and") -> torch.Tensor:
        """
        Combine multiple masks.

        Args:
            masks: List of binary masks (same shape)
            operation: 'and' (intersection) or 'or' (union)

        Returns:
            Combined mask
        """
        if not masks:
            raise ValueError("No masks provided")

        result = masks[0].clone()

        for mask in masks[1:]:
            if mask.shape != result.shape:
                raise ValueError("All masks must have the same shape")

            if operation == "and":
                result = result & mask
            elif operation == "or":
                result = result | mask
            else:
                raise ValueError(f"Unknown operation: {operation}")

        return result

    @staticmethod
    def global_threshold_mask(layer_scores: dict, global_amount: float, mode: str = "low") -> dict:
        """
        Create masks using a global threshold across all layers.

        Args:
            layer_scores: Dict mapping layer names to score tensors
            global_amount: Global fraction to prune
            mode: Pruning mode

        Returns:
            Dict mapping layer names to masks
        """
        # Concatenate all scores
        all_scores = []
        layer_info = []

        for layer_name, scores in layer_scores.items():
            all_scores.append(scores.flatten())
            layer_info.append((layer_name, scores.shape, len(scores.flatten())))

        all_scores_cat = torch.cat(all_scores)

        # Compute global threshold
        num_total = len(all_scores_cat)
        num_to_keep = int(num_total * (1 - global_amount))

        if mode == "low":
            threshold = torch.topk(all_scores_cat, num_to_keep, largest=True)[0][-1]

            def threshold_fn(s):
                return s >= threshold

        elif mode == "high":
            threshold = torch.topk(all_scores_cat, num_to_keep, largest=False)[0][-1]

            def threshold_fn(s):
                return s <= threshold

        else:
            raise ValueError(f"Global thresholding not supported for mode: {mode}")

        # Create per-layer masks
        masks = {}
        for layer_name, shape, _ in layer_info:
            scores = layer_scores[layer_name]
            mask = threshold_fn(scores)
            masks[layer_name] = mask

        logger.info(f"Global thresholding: {num_to_keep}/{num_total} elements kept")

        return masks
