"""Parallel batch pruning strategy for efficient evaluation of multiple configurations."""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ParallelBatchPruning:
    """
    Pruning strategy that evaluates all networks and sparsity levels in parallel.
    
    This strategy processes multiple networks and pruning configurations simultaneously
    using vectorized operations for maximum efficiency.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.eval_batches = getattr(config, 'eval_batches', None) if config else None
    
    def run_pruning_experiments(
        self, 
        networks: List[nn.Module],
        data_loader,
        strategies: List[str],
        selection_modes: List[str],
        pruning_amounts: List[float],
        device: str = 'cuda'
    ) -> Dict[str, any]:
        """
        Run pruning experiments for all configurations in parallel.
        
        Args:
            networks: List of neural networks to prune
            data_loader: DataLoader for evaluation
            strategies: List of pruning strategies to test
            selection_modes: List of selection modes (low, high, random)
            pruning_amounts: List of sparsity levels to test
            device: Device to run computations on
            
        Returns:
            Dictionary with pruning results for all strategies
        """
        logger.info("Running parallel batch pruning experiments")
        
        results = {"strategies": {}}
        
        # Save original states efficiently
        original_states = []
        for model in networks:
            state = {name: module.weight.data.clone() 
                    for name, module in model.named_modules() 
                    if hasattr(module, 'weight')}
            original_states.append(state)
        
        # Process each strategy
        for strategy_name in strategies:
            logger.info(f"Testing pruning strategy: {strategy_name}")
            
            for selection_mode in selection_modes:
                logger.info(f"  Selection mode: {selection_mode}")
                
                # Run parallel pruning for this configuration
                batch_results = self._parallel_batch_pruning(
                    networks, data_loader, strategy_name, selection_mode, 
                    pruning_amounts, original_states, device
                )
                
                # Store aggregated results
                strategy_key = f"{strategy_name}_{selection_mode}"
                strategy_results = {
                    "sparsities": batch_results["sparsities"].mean(dim=0).tolist(),
                    "accuracies_before_finetune": batch_results["accuracies_before"].mean(dim=0).tolist(),
                    "accuracies_after_finetune": batch_results["accuracies_after"].mean(dim=0).tolist(),
                    "losses_before_finetune": batch_results["losses_before"].mean(dim=0).tolist(),
                    "losses_after_finetune": batch_results["losses_after"].mean(dim=0).tolist(),
                    "improvements": (batch_results["accuracies_after"] - batch_results["accuracies_before"]).mean(dim=0).tolist()
                }
                
                # Add standard deviations if multiple networks
                if len(networks) > 1:
                    strategy_results["accuracies_before_finetune_std"] = batch_results["accuracies_before"].std(dim=0).tolist()
                    strategy_results["accuracies_after_finetune_std"] = batch_results["accuracies_after"].std(dim=0).tolist()
                
                results["strategies"][strategy_key] = strategy_results
        
        # Restore original weights
        for net_idx, model in enumerate(networks):
            for name, module in model.named_modules():
                if name in original_states[net_idx]:
                    module.weight.data = original_states[net_idx][name]
        
        return results
    
    def _parallel_batch_pruning(
        self,
        networks: List[nn.Module],
        data_loader,
        strategy_name: str,
        selection_mode: str,
        pruning_amounts: List[float],
        original_states: List[Dict[str, torch.Tensor]],
        device: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process all networks and pruning amounts in parallel using vectorized operations.
        """
        num_networks = len(networks)
        num_amounts = len(pruning_amounts)
        total_configs = num_networks * num_amounts
        
        logger.info(f"    Processing {total_configs} configurations in parallel")
        logger.info(f"    Networks: {num_networks}, Sparsity levels: {num_amounts}")
        
        # Initialize result tensors on device for speed
        accuracies_before = torch.zeros(num_networks, num_amounts, device=device)
        losses_before = torch.zeros(num_networks, num_amounts, device=device)
        sparsities = torch.zeros(num_networks, num_amounts)
        
        # Create all masks upfront
        logger.info("    Creating masks for all configurations...")
        all_masks = self._create_all_masks(networks, strategy_name, selection_mode, pruning_amounts)
        
        # Evaluate all configurations in parallel
        logger.info("    Starting parallel evaluation...")
        start_time = time.time()
        
        # Pre-allocate for all configurations
        all_correct = torch.zeros(total_configs, device=device)
        all_loss = torch.zeros(total_configs, device=device)
        total_samples = 0
        
        # Set all networks to eval mode
        for net in networks:
            net.eval()
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = targets.size(0)
                
                # Collect outputs from ALL configurations
                all_outputs = []
                
                for net_idx in range(num_networks):
                    net = networks[net_idx]
                    
                    for amount_idx in range(num_amounts):
                        # Apply configuration
                        self._apply_mask_config(net, all_masks[net_idx][amount_idx], original_states[net_idx])
                        
                        # Forward pass
                        outputs = net(inputs)
                        all_outputs.append(outputs)
                        
                        # Calculate sparsity (only once)
                        if batch_count == 0:
                            sparsities[net_idx, amount_idx] = self._calculate_sparsity(net)
                
                # Stack all outputs for vectorized processing
                stacked_outputs = torch.stack(all_outputs, dim=0)
                
                # Compute losses for all configs at once
                expanded_targets = targets.unsqueeze(0).expand(total_configs, -1)
                all_batch_losses = criterion(
                    stacked_outputs.reshape(-1, stacked_outputs.size(-1)), 
                    expanded_targets.reshape(-1)
                ).reshape(total_configs, batch_size)
                
                # Sum losses
                all_loss += all_batch_losses.sum(dim=1)
                
                # Get predictions and count correct
                all_preds = stacked_outputs.argmax(dim=2)
                correct = all_preds.eq(expanded_targets).sum(dim=1)
                all_correct += correct
                
                total_samples += batch_size
                batch_count += 1
                
                # Check if we've evaluated enough batches
                if self.eval_batches is not None and batch_count >= self.eval_batches:
                    break
        
        # Reshape results back to [num_networks, num_amounts]
        all_correct = all_correct.reshape(num_networks, num_amounts)
        all_loss = all_loss.reshape(num_networks, num_amounts)
        
        # Convert to accuracies and average losses
        accuracies_before = (all_correct * 100.0 / total_samples).cpu()
        losses_before = (all_loss / batch_count).cpu()
        
        eval_time = time.time() - start_time
        logger.info(f"    Parallel evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"    Average accuracy: {accuracies_before.mean():.2f}%")
        
        # For now, no fine-tuning in parallel mode
        accuracies_after = accuracies_before.clone()
        losses_after = losses_before.clone()
        
        # Reset networks to train mode
        for net in networks:
            net.train()
        
        return {
            "accuracies_before": accuracies_before,
            "losses_before": losses_before,
            "accuracies_after": accuracies_after,
            "losses_after": losses_after,
            "sparsities": sparsities
        }
    
    def _create_all_masks(
        self,
        networks: List[nn.Module],
        strategy_name: str,
        selection_mode: str,
        pruning_amounts: List[float]
    ) -> List[List[Dict[str, torch.Tensor]]]:
        """Create all masks for all networks and pruning amounts."""
        all_masks = []
        
        for net in networks:
            network_masks = []
            
            for amount in pruning_amounts:
                if strategy_name == "magnitude":
                    masks = self._create_magnitude_masks(net, amount, selection_mode)
                elif strategy_name == "random":
                    masks = self._create_random_masks(net, amount)
                elif strategy_name == "alignment":
                    # For alignment, use magnitude as a proxy in batch mode
                    masks = self._create_magnitude_masks(net, amount, selection_mode)
                else:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
                
                network_masks.append(masks)
            
            all_masks.append(network_masks)
        
        return all_masks
    
    def _create_magnitude_masks(
        self,
        model: nn.Module,
        amount: float,
        selection_mode: str
    ) -> Dict[str, torch.Tensor]:
        """Create magnitude-based masks for a model."""
        masks = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                importance = weight.abs()
                
                if hasattr(self.config, 'alignment_structured_pruning') and self.config.alignment_structured_pruning and len(weight.shape) >= 2:
                    # Structured pruning - prune entire neurons
                    neuron_importance = importance.mean(dim=tuple(range(1, len(weight.shape))))
                    mask = self._create_structured_mask(neuron_importance, amount, selection_mode, weight.shape)
                else:
                    # Unstructured pruning
                    mask = self._create_mask(importance, amount, selection_mode)
                
                masks[name] = mask
        
        return masks
    
    def _create_random_masks(
        self,
        model: nn.Module,
        amount: float
    ) -> Dict[str, torch.Tensor]:
        """Create random masks for a model."""
        masks = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                
                if hasattr(self.config, 'alignment_structured_pruning') and self.config.alignment_structured_pruning and len(weight.shape) >= 2:
                    # Structured random pruning
                    num_neurons = weight.shape[0]
                    num_to_prune = int(amount * num_neurons)
                    mask = torch.ones(num_neurons, device=weight.device)
                    if num_to_prune > 0:
                        indices = torch.randperm(num_neurons)[:num_to_prune]
                        mask[indices] = 0
                    # Expand to match weight dimensions
                    mask = mask.unsqueeze(1).expand_as(weight)
                else:
                    # Unstructured random pruning
                    mask = torch.rand_like(weight) > amount
                
                masks[name] = mask.float()
        
        return masks
    
    def _create_mask(
        self,
        importance: torch.Tensor,
        amount: float,
        selection_mode: str
    ) -> torch.Tensor:
        """Create a binary mask based on importance scores."""
        if amount == 0:
            return torch.ones_like(importance)
        elif amount >= 1:
            return torch.zeros_like(importance)
        
        flat_importance = importance.flatten()
        k = int(amount * flat_importance.numel())
        
        if k == 0:
            return torch.ones_like(importance)
        
        if selection_mode == "low":
            threshold = torch.kthvalue(flat_importance, k).values
            mask = importance > threshold
        elif selection_mode == "high":
            threshold = torch.kthvalue(flat_importance, flat_importance.numel() - k).values
            mask = importance < threshold
        elif selection_mode == "random":
            mask = torch.rand_like(importance) > amount
        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")
        
        return mask.float()
    
    def _create_structured_mask(
        self,
        neuron_importance: torch.Tensor,
        amount: float,
        selection_mode: str,
        weight_shape: torch.Size
    ) -> torch.Tensor:
        """Create a structured mask that prunes entire neurons."""
        num_neurons = neuron_importance.numel()
        num_to_prune = int(amount * num_neurons)
        
        if num_to_prune == 0:
            mask = torch.ones_like(neuron_importance)
        elif num_to_prune >= num_neurons:
            mask = torch.zeros_like(neuron_importance)
        else:
            if selection_mode == "low":
                _, indices = torch.topk(neuron_importance, num_neurons - num_to_prune)
                mask = torch.zeros_like(neuron_importance)
                mask[indices] = 1
            elif selection_mode == "high":
                _, indices = torch.topk(neuron_importance, num_to_prune)
                mask = torch.ones_like(neuron_importance)
                mask[indices] = 0
            elif selection_mode == "random":
                mask = torch.ones_like(neuron_importance)
                indices = torch.randperm(num_neurons)[:num_to_prune]
                mask[indices] = 0
            else:
                raise ValueError(f"Unknown selection mode: {selection_mode}")
        
        # Expand mask to match weight dimensions
        if len(weight_shape) == 2:
            # Linear layer: [out_features, in_features]
            mask = mask.unsqueeze(1).expand_as(torch.zeros(weight_shape))
        elif len(weight_shape) == 4:
            # Conv layer: [out_channels, in_channels, height, width]
            mask = mask.view(-1, 1, 1, 1).expand_as(torch.zeros(weight_shape))
        
        return mask
    
    def _apply_mask_config(
        self,
        model: nn.Module,
        masks: Dict[str, torch.Tensor],
        original_state: Dict[str, torch.Tensor]
    ):
        """Apply masks to a model after restoring original weights."""
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in original_state:
                    # Restore original weights
                    module.weight.data = original_state[name].clone()
                    
                    # Apply mask if exists
                    if name in masks:
                        module.weight.data *= masks[name]
    
    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate the sparsity of a model."""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0 