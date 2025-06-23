#!/usr/bin/env python
"""
Run a fixed version of the alignment experiment.
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
    """Train multiple networks on the given dataset."""
    logger.info(f"Training {len(networks)} networks for {num_epochs} epochs")
    
    # Track training history for plotting
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Train each network
    trained_networks = []
    for i, network in enumerate(networks):
        logger.info(f"Training network {i+1}/{len(networks)}")
        
        # Move network to the device
        network = network.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        
        # Training loop
        network.train()
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Train on each batch
            for inputs, targets in tqdm(dataset.train_loader, desc=f"Network {i+1}, Epoch {epoch+1}/{num_epochs}"):
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
            
            # Calculate epoch statistics
            train_loss = running_loss / len(dataset.train_loader)
            train_acc = 100.0 * correct / total
            
            # Evaluate on test set
            network.eval()
            test_correct = 0
            test_total = 0
            test_loss_sum = 0.0
            
            with torch.no_grad():
                for inputs, targets in dataset.test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = network(inputs)
                    
                    # Calculate loss
                    loss = F.cross_entropy(outputs, targets, reduction='sum')
                    test_loss_sum += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = 100.0 * test_correct / test_total
            test_loss = test_loss_sum / test_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            # Log progress
            logger.info(f"Network {i+1}, Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                       f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
        
        # Add to trained networks
        trained_networks.append(network)
        
        # Accumulate training history (average across networks)
        if i == 0:
            # First network, initialize history
            training_history = history
        else:
            # Average with previous networks
            for key in training_history:
                if key in history:
                    # Calculate running average
                    for epoch_idx in range(len(history[key])):
                        if epoch_idx < len(training_history[key]):
                            training_history[key][epoch_idx] = (training_history[key][epoch_idx] * i + history[key][epoch_idx]) / (i + 1)
    
    logger.info(f"Completed training {len(networks)} networks")
    return trained_networks, training_history

def test_pruning_strategies(networks, dataset, dropout_fractions, device):
    """Test different pruning strategies on trained networks."""
    logger.info(f"Testing pruning strategies on {len(networks)} networks")
    
    # Initialize results
    results = {
        'dropout_fractions': dropout_fractions,
        'accuracies': {'high_rq': [], 'low_rq': [], 'random': []},
        'stds': {'high_rq': [], 'low_rq': [], 'random': []},
        'losses': {'high_rq': [], 'low_rq': [], 'random': []}
    }
    
    # For each pruning percentage, test all networks
    for prune_idx, prune_percent in enumerate(dropout_fractions):
        logger.info(f"Testing pruning percentage: {prune_percent*100:.1f}%")
        
        # Store results for each network at this pruning percentage
        strategy_results = {
            'high_rq': {'acc': [], 'loss': []},
            'low_rq': {'acc': [], 'loss': []},
            'random': {'acc': [], 'loss': []}
        }
        
        # Process each network
        for net_idx, network in enumerate(networks):
            logger.info(f"Processing network {net_idx+1}/{len(networks)}")
            
            # Save original weights
            original_weights = {}
            original_biases = {}
            
            for i, layer in enumerate(network.alignment_layers):
                if hasattr(layer, "weight") and layer.weight is not None:
                    original_weights[i] = layer.weight.data.clone()
                    if hasattr(layer, "bias") and layer.bias is not None:
                        original_biases[i] = layer.bias.data.clone()
            
            # Get original accuracy
            if prune_idx == 0 or net_idx == 0:
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
                
                orig_accuracy = 100.0 * correct / total
                orig_loss = total_loss / total
                
                logger.info(f"Original accuracy: {orig_accuracy:.2f}%, loss: {orig_loss:.4f}")
            
            # Test each strategy
            for strategy in ["high_rq", "low_rq", "random"]:
                # Restore original weights
                for i, layer in enumerate(network.alignment_layers):
                    if i in original_weights:
                        layer.weight.data = original_weights[i].clone()
                        if i in original_biases and hasattr(layer, "bias") and layer.bias is not None:
                            layer.bias.data = original_biases[i].clone()
                
                # Skip pruning for 0% case
                if prune_percent == 0.0:
                    acc = orig_accuracy
                    loss = orig_loss
                else:
                    # Apply pruning based on strategy
                    total_neurons = 0
                    total_pruned = 0
                    
                    # Apply pruning to each layer
                    for i, layer in enumerate(network.alignment_layers):
                        if i not in original_weights:
                            continue
                        
                        # Compute neuron importance scores (using weight magnitude as proxy)
                        weights = layer.weight.data
                        input_dim = weights.shape[1]
                        total_neurons += input_dim
                        
                        # Calculate importance scores (weight magnitude)
                        neuron_scores = [torch.norm(weights[:, j]).item() for j in range(input_dim)]
                        
                        # Calculate how many neurons to prune
                        num_to_drop = max(1, int(input_dim * prune_percent)) if prune_percent > 0 else 0
                        total_pruned += num_to_drop
                        
                        if num_to_drop > 0:
                            # Get indices to drop based on strategy
                            if strategy == "high_rq":  # Drop highest alignment neurons
                                sorted_indices = np.argsort(neuron_scores)[::-1]  # Sort descending
                                to_drop = sorted_indices[:num_to_drop]
                            elif strategy == "low_rq":  # Drop lowest alignment neurons
                                sorted_indices = np.argsort(neuron_scores)  # Sort ascending
                                to_drop = sorted_indices[:num_to_drop]
                            else:  # Random pruning
                                all_indices = list(range(input_dim))
                                np.random.shuffle(all_indices)
                                to_drop = all_indices[:num_to_drop]
                            
                            # Zero out weights for these neurons
                            for idx in to_drop:
                                if idx < weights.shape[1]:
                                    layer.weight.data[:, idx] = 0.0
                                    if hasattr(layer, "bias") and layer.bias is not None and idx < layer.bias.data.shape[0]:
                                        layer.bias.data[idx] = 0.0
                    
                    logger.info(f"Pruned {total_pruned}/{total_neurons} neurons with strategy: {strategy}")
                    
                    # Evaluate pruned network
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
                    
                    acc = 100.0 * correct / total
                    loss = total_loss / total
                
                # Store results for this network and strategy
                strategy_results[strategy]['acc'].append(acc)
                strategy_results[strategy]['loss'].append(loss)
                
                logger.info(f"Strategy {strategy}, Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        
        # Calculate mean and std for each strategy at this pruning percentage
        for strategy in ["high_rq", "low_rq", "random"]:
            # Get all networks' results for this strategy and pruning percentage
            accs = strategy_results[strategy]['acc']
            losses = strategy_results[strategy]['loss']
            
            # Calculate statistics
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_loss = np.mean(losses)
            
            # Add to results
            results['accuracies'][strategy].append(mean_acc)
            results['stds'][strategy].append(std_acc)
            results['losses'][strategy].append(mean_loss)
            
            logger.info(f"Pruning {prune_percent*100:.1f}%, Strategy {strategy}: "
                       f"Mean acc: {mean_acc:.2f}%, Std: {std_acc:.2f}%")
    
    return results

def main():
    """Main function to run the experiment."""
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
    num_networks = min(getattr(config.training, "replicates", 5), 2)  # Limit to 2 networks for faster testing
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
    
    # Train all networks
    trained_networks, training_history = train_networks(
        networks, 
        dataset, 
        device,
        num_epochs=getattr(config.training, "epochs", 5),
        learning_rate=getattr(config.training, "learning_rate", 0.001)
    )
    
    # Test pruning strategies
    pruning_results = test_pruning_strategies(
        trained_networks,
        dataset,
        dropout_fractions,
        device
    )
    
    # Add training history to results
    pruning_results['training_history'] = training_history
    
    # Create output directory
    results_dir = "debug_output"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate plots
    try:
        logger.info("Generating pruning plots with error bars")
        
        # Get pruning mode and dropout mode from config
        pruning_mode = getattr(config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(config.extra, "dropout_mode", "scaled")
        
        saved_plots = plot_dropout_results(
            pruning_results,
            results_dir,
            title_prefix="Fixed Pruning Test",
            pruning_mode=pruning_mode,
            dropout_mode=dropout_mode
        )
        
        if saved_plots:
            logger.info(f"Generated {len(saved_plots)} plots: {saved_plots}")
        
    except Exception as e:
        logger.error(f"Error generating dropout plots: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Experiment completed successfully!")
    
if __name__ == "__main__":
    main() 