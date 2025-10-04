#!/usr/bin/env python
"""
Run an experiment with multiple pruning strategies.

This script executes neural network pruning using three different strategies:
1. high_rq: Prune neurons with highest alignment scores (weight magnitudes)
2. low_rq: Prune neurons with lowest alignment scores (weight magnitudes)
3. random: Prune neurons randomly

The results are plotted for comparison.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

# Configure basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Add the project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Import from the alignment package
from alignment.config import ExperimentConfig
from alignment.models.registry import create_model
from alignment.datasets import load_dataset
from alignment.metrics import get_metric
from alignment.utils.plotting import plot_dropout_results

def train_networks(networks, dataset, device, num_epochs=5, learning_rate=0.001):
    """Train multiple networks efficiently in one training loop."""
    logger.info(f"Training {len(networks)} networks for {num_epochs} epochs")
    
    # Track training history for plotting
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Set up optimizer for each network
    optimizers = []
    for network in networks:
        network.to(device)
        optimizers.append(torch.optim.Adam(network.parameters(), lr=learning_rate))
    
    # Train all networks for each epoch
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs", position=0)
    for epoch in epoch_pbar:
        # Initialize epoch stats
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        epoch_test_loss = 0.0
        epoch_test_acc = 0.0
        
        # Training phase
        net_pbar = tqdm(enumerate(zip(networks, optimizers)), 
                       desc=f"Epoch {epoch+1}/{num_epochs} networks", 
                       total=len(networks), 
                       position=1, 
                       leave=False)
                       
        for network_idx, (network, optimizer) in net_pbar:
            network.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Train on each batch
            batch_pbar = tqdm(dataset.train_loader, 
                             desc=f"Network {network_idx+1}/{len(networks)}", 
                             position=2, 
                             leave=False)
                             
            for inputs, targets in batch_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = network(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update batch progress bar
                if total > 0:
                    batch_pbar.set_postfix({
                        'loss': running_loss / len(batch_pbar), 
                        'acc': 100.0 * correct / total
                    })
            
            # Calculate network training statistics
            if total > 0:
                network_train_loss = running_loss / len(dataset.train_loader)
                network_train_acc = 100.0 * correct / total
                
                # Accumulate for epoch average
                epoch_train_loss += network_train_loss
                epoch_train_acc += network_train_acc
                
                # Update network progress bar
                net_pbar.set_postfix({
                    'train_loss': network_train_loss, 
                    'train_acc': network_train_acc
                })
            
            # Evaluation phase
            network.eval()
            test_correct = 0
            test_total = 0
            test_loss_sum = 0.0
            
            # Evaluation progress bar
            eval_pbar = tqdm(dataset.test_loader, 
                            desc=f"Evaluating network {network_idx+1}", 
                            position=2, 
                            leave=False)
                            
            with torch.no_grad():
                for inputs, targets in eval_pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = network(inputs)
                    
                    # Calculate loss
                    loss = F.cross_entropy(outputs, targets, reduction='sum')
                    test_loss_sum += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
                    
                    # Update eval progress bar
                    if test_total > 0:
                        eval_pbar.set_postfix({
                            'test_loss': test_loss_sum / test_total, 
                            'test_acc': 100.0 * test_correct / test_total
                        })
            
            # Calculate network testing statistics
            if test_total > 0:
                network_test_loss = test_loss_sum / test_total
                network_test_acc = 100.0 * test_correct / test_total
                
                # Accumulate for epoch average
                epoch_test_loss += network_test_loss
                epoch_test_acc += network_test_acc
        
        # Calculate and store epoch averages
        epoch_train_loss /= len(networks)
        epoch_train_acc /= len(networks)
        epoch_test_loss /= len(networks)
        epoch_test_acc /= len(networks)
        
        training_history['train_loss'].append(epoch_train_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['test_loss'].append(epoch_test_loss)
        training_history['test_acc'].append(epoch_test_acc)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': epoch_train_loss,
            'train_acc': f"{epoch_train_acc:.2f}%",
            'test_loss': epoch_test_loss,
            'test_acc': f"{epoch_test_acc:.2f}%"
        })
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.2f}%, "
                  f"Test Loss={epoch_test_loss:.4f}, Test Acc={epoch_test_acc:.2f}%")
    
    logger.info(f"Completed training {len(networks)} networks.")
    return networks, training_history

def run_pruning(networks, dataset, dropout_fractions, device, pruning_mode="layer_wise", dropout_mode="scaled"):
    """Run pruning with three different strategies and collect results in a more efficient way."""
    # Define the strategies to evaluate
    strategies = ["high_rq", "low_rq", "random"]
    
    # Initialize results structure
    results = {
        'dropout_fractions': dropout_fractions,
        'accuracies': {strategy: [] for strategy in strategies},
        'stds': {strategy: [] for strategy in strategies},
        'losses': {strategy: [] for strategy in strategies}
    }
    
    # Store original weights and biases for all networks
    logger.info("Saving original network weights")
    original_weights = {}
    original_biases = {}
    
    for net_idx, network in enumerate(networks):
        original_weights[net_idx] = {}
        original_biases[net_idx] = {}
        
        for layer_idx, layer in enumerate(network.alignment_layers):
            if hasattr(layer, "weight") and layer.weight is not None:
                original_weights[net_idx][layer_idx] = layer.weight.data.clone()
                if hasattr(layer, "bias") and layer.bias is not None:
                    original_biases[net_idx][layer_idx] = layer.bias.data.clone()
    
    # First, get original accuracy for all networks (0% pruning)
    logger.info("Evaluating original network performance (0% pruning)")
    orig_accs = []
    orig_losses = []
    
    # Move all networks to device
    for network in networks:
        network.to(device)
    
    # Evaluate all networks at baseline
    for network in networks:
        network.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataset.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = network(inputs)
                
                # Compute loss
                loss = F.cross_entropy(outputs, targets, reduction='sum')
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        if total > 0:
            orig_accs.append(100.0 * correct / total)
            orig_losses.append(total_loss / total)
    
    # Calculate baseline statistics
    orig_mean_acc = np.mean(orig_accs)
    orig_std_acc = np.std(orig_accs)
    orig_mean_loss = np.mean(orig_losses)
    
    # Add baseline values to all strategies (they start from the same point)
    for strategy in strategies:
        results['accuracies'][strategy].append(orig_mean_acc)
        results['stds'][strategy].append(orig_std_acc)
        results['losses'][strategy].append(orig_mean_loss)
    
    logger.info(f"Original accuracy: {orig_mean_acc:.2f}% ± {orig_std_acc:.2f}%")
    
    # Iterate through pruning fractions (skip the first one which is 0%)
    fraction_pbar = tqdm(enumerate(dropout_fractions), 
                       total=len(dropout_fractions), 
                       desc="Pruning fractions", 
                       position=0)
    
    for fraction_idx, fraction in fraction_pbar:
        if fraction_idx == 0 and fraction == 0.0:
            continue
            
        fraction_pbar.set_description(f"Pruning fraction: {fraction:.2f}")
        
        # For each strategy, process all networks at once 
        strategy_pbar = tqdm(strategies, desc=f"Pruning strategies", position=1, leave=False)
        
        for strategy in strategy_pbar:
            strategy_pbar.set_description(f"Strategy: {strategy}")
            
            # Restore original weights for all networks at once
            for net_idx, network in enumerate(networks):
                for layer_idx in original_weights[net_idx]:
                    layer = network.alignment_layers[layer_idx]
                    layer.weight.data = original_weights[net_idx][layer_idx].clone()
                    if layer_idx in original_biases[net_idx] and hasattr(layer, "bias") and layer.bias is not None:
                        layer.bias.data = original_biases[net_idx][layer_idx].clone()
            
            # Apply pruning to all networks in parallel
            for net_idx, network in enumerate(networks):
                # Apply pruning based on the mode
                if pruning_mode == "global_joint":
                    # Collect all neurons and their scores across all layers
                    all_neurons = []
                    
                    for layer_idx, layer in enumerate(network.alignment_layers):
                        if layer_idx not in original_weights[net_idx]:
                            continue
                            
                        weights = layer.weight.data
                        input_dim = weights.shape[1]
                        
                        # Calculate importance scores
                        neuron_scores = torch.norm(weights, dim=0).cpu().numpy()
                        
                        # Store (layer_idx, neuron_idx, score) tuples
                        for j, score in enumerate(neuron_scores):
                            all_neurons.append((layer_idx, j, score))
                    
                    # Sort neurons by score based on strategy
                    if strategy == "high_rq":
                        # Sort by highest scores first
                        all_neurons.sort(key=lambda x: x[2], reverse=True)
                    elif strategy == "low_rq":
                        # Sort by lowest scores first
                        all_neurons.sort(key=lambda x: x[2])
                    elif strategy == "random":
                        # Shuffle neurons randomly
                        import random
                        random.shuffle(all_neurons)
                    
                    # Calculate how many neurons to prune
                    total_neurons = len(all_neurons)
                    num_to_drop = int(total_neurons * fraction)
                    
                    if num_to_drop > 0:
                        # Get indices to drop
                        to_drop = all_neurons[:num_to_drop]
                        
                        # Apply pruning
                        for layer_idx, neuron_idx, _ in to_drop:
                            layer = network.alignment_layers[layer_idx]
                            # Zero out weights for this neuron
                            if neuron_idx < layer.weight.data.shape[1]:
                                layer.weight.data[:, neuron_idx] = 0.0
                                if hasattr(layer, "bias") and layer.bias is not None and neuron_idx < layer.bias.data.shape[0]:
                                    layer.bias.data[neuron_idx] = 0.0
                
                elif pruning_mode == "layer_wise":
                    # Apply pruning to each layer individually - can optimize with tensor operations
                    for layer_idx, layer in enumerate(network.alignment_layers):
                        if layer_idx not in original_weights[net_idx]:
                            continue
                            
                        weights = layer.weight.data
                        input_dim = weights.shape[1]
                        
                        # Calculate importance scores - vectorized version
                        neuron_scores = torch.norm(weights, dim=0).cpu().numpy()
                        
                        # Calculate how many neurons to prune in this layer
                        num_to_drop = int(input_dim * fraction)
                        
                        if num_to_drop > 0:
                            # Get neurons to drop based on strategy
                            if strategy == "high_rq":
                                # Sort by highest scores first (descending)
                                sorted_indices = np.argsort(neuron_scores)[::-1]
                                to_drop = sorted_indices[:num_to_drop]
                            elif strategy == "low_rq":
                                # Sort by lowest scores first (ascending)
                                sorted_indices = np.argsort(neuron_scores)
                                to_drop = sorted_indices[:num_to_drop]
                            elif strategy == "random":
                                # Choose random neurons
                                all_indices = np.arange(input_dim)
                                np.random.shuffle(all_indices)
                                to_drop = all_indices[:num_to_drop]
                            
                            # Apply pruning
                            if to_drop.size > 0:
                                # Create a mask tensor
                                mask = torch.ones_like(weights)
                                
                                # Convert numpy array to tensor and ensure it's properly formatted for indexing
                                to_drop_tensor = torch.tensor(to_drop.tolist(), device=weights.device, dtype=torch.long)
                                
                                # Use tensor indexing
                                mask[:, to_drop_tensor] = 0.0
                                
                                # Apply mask to weights
                                layer.weight.data = weights * mask
                                
                                # Apply to bias if it exists
                                if hasattr(layer, "bias") and layer.bias is not None:
                                    bias_mask = torch.ones_like(layer.bias.data)
                                    valid_indices = [idx for idx in to_drop if idx < layer.bias.data.shape[0]]
                                    if valid_indices:
                                        bias_indices = torch.tensor(valid_indices, device=layer.bias.device, dtype=torch.long)
                                        bias_mask[bias_indices] = 0.0
                                        layer.bias.data = layer.bias.data * bias_mask
            
            # Evaluate all pruned networks efficiently
            network_accs = []
            network_losses = []
            
            for network in networks:
                # Evaluate
                network.eval()
                correct = 0
                total = 0
                total_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in dataset.test_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = network(inputs)
                        
                        # Compute loss
                        loss = F.cross_entropy(outputs, targets, reduction='sum')
                        total_loss += loss.item()
                        
                        # Compute accuracy
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                if total > 0:
                    accuracy = 100.0 * correct / total
                    loss = total_loss / total
                    
                    network_accs.append(accuracy)
                    network_losses.append(loss)
            
            # Calculate statistics for this fraction and strategy
            if network_accs:
                mean_acc = np.mean(network_accs)
                std_acc = np.std(network_accs)
                mean_loss = np.mean(network_losses)
                
                # Add to results
                results['accuracies'][strategy].append(mean_acc)
                results['stds'][strategy].append(std_acc)
                results['losses'][strategy].append(mean_loss)
                
                logger.info(f"  {strategy}: Accuracy {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Restore original weights for all networks
    logger.info("Restoring original weights")
    for net_idx, network in enumerate(networks):
        for layer_idx in original_weights[net_idx]:
            layer = network.alignment_layers[layer_idx]
            layer.weight.data = original_weights[net_idx][layer_idx].clone()
            if layer_idx in original_biases[net_idx] and hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data = original_biases[net_idx][layer_idx].clone()
    
    return results

def plot_results(results, pruning_mode, dropout_mode, output_dir="multi_strategy_output"):
    """Create plots showing the results of different pruning strategies."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    fractions = results['dropout_fractions']
    strategies = list(results['accuracies'].keys())
    
    # Set up plot colors and markers
    colors = {'high_rq': 'red', 'low_rq': 'green', 'random': 'blue'}
    markers = {'high_rq': 'o', 'low_rq': 's', 'random': '^'}
    labels = {
        'high_rq': 'Prune Highest Magnitude',
        'low_rq': 'Prune Lowest Magnitude', 
        'random': 'Prune Random'
    }
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    
    for strategy in strategies:
        accs = results['accuracies'][strategy]
        stds = results['stds'][strategy]
        
        plt.errorbar(
            fractions, 
            accs, 
            yerr=stds,
            label=labels[strategy],
            color=colors[strategy],
            marker=markers[strategy],
            capsize=4,
            markersize=8,
            linewidth=2
        )
    
    plt.title(f'Accuracy vs. Pruning Fraction\n({pruning_mode} pruning, {dropout_mode} mode)', fontsize=14)
    plt.xlabel('Pruning Fraction', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save plot
    accuracy_plot_file = os.path.join(output_dir, f'accuracy_plot_{pruning_mode}.png')
    plt.savefig(accuracy_plot_file, dpi=300)
    plt.close()
    
    # Also use the alignment plotting utility if available
    try:
        from alignment.utils.plotting import plot_dropout_results
        
        saved_plots = plot_dropout_results(
            results,
            output_dir,
            title_prefix=f"Multi-Strategy Pruning Test",
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode
        )
        
        if saved_plots:
            logger.info(f"Generated {len(saved_plots)} plots using alignment plotting utility")
            return saved_plots
    except Exception as e:
        logger.warning(f"Could not use alignment plotting utility: {str(e)}")
    
    return [accuracy_plot_file]

def main():
    """Main function to run the multi-strategy pruning experiment."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config_path = "configs/config_alignment_experiment.yaml"
    config = ExperimentConfig.load(config_path)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset.dataset_name}")
    dataset = load_dataset(config.dataset)
    
    # Create multiple networks (replicates)
    num_networks = min(getattr(config.training, "replicates", 5), 3)  # Limit to 3 networks for faster testing
    logger.info(f"Creating {num_networks} networks")
    
    networks = []
    for i in range(num_networks):
        logger.info(f"Creating model {i+1}/{num_networks}: {config.model.model_name}")
        network = create_model(config.model)
        networks.append(network)
    
    # Get dropout fractions from config
    dropin_min = config.alignment.dropout_min
    dropin_max = config.alignment.dropout_max
    num_dropout_fractions = config.alignment.dropout_steps
    dropout_fractions = np.linspace(dropin_min, dropin_max, num_dropout_fractions).tolist()
    
    # Get pruning and dropout modes
    pruning_mode = getattr(config.extra, "dropout_pruning_mode", "layer_wise")
    dropout_mode = getattr(config.extra, "dropout_mode", "scaled")
    
    # Train all networks
    trained_networks, training_history = train_networks(
        networks, 
        dataset, 
        device,
        num_epochs=getattr(config.training, "epochs", 10),
        learning_rate=getattr(config.training, "learning_rate", 0.001)
    )
    
    # Run pruning with different strategies
    pruning_results = run_pruning(
        trained_networks,
        dataset,
        dropout_fractions,
        device,
        pruning_mode=pruning_mode,
        dropout_mode=dropout_mode
    )
    
    # Add training history to results
    pruning_results['training_history'] = training_history
    
    # Create output directory
    results_dir = "multi_strategy_output"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate plots
    try:
        logger.info("Generating multi-strategy pruning plots")
        
        saved_plots = plot_results(
            pruning_results,
            pruning_mode,
            dropout_mode,
            output_dir=results_dir
        )
        
        if saved_plots:
            logger.info(f"Generated {len(saved_plots)} plots: {saved_plots}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Multi-strategy experiment completed successfully!")
    
if __name__ == "__main__":
    main() 