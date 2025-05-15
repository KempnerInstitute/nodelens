import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
import copy

# Add parent directory to path to access src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.alignment.models.models import MLP
from src.alignment.datasets import load_dataset
from src.alignment.dropout import test_pruning_strategies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load MNIST dataset for testing
    dataset_config = {
        "dataset_name": "MNIST",
        "data_path": "data",
        "batch_size": 1024,
        "model_name": "mlp",
    }
    dataset = load_dataset(dataset_config, device=device)
    logger.info(f"Loaded {dataset_config['dataset_name']} dataset")

    # Create a simple MLP model for MNIST
    network = MLP(
        input_dim=784,  # 28x28 for MNIST
        num_hidden=[100, 100],
        output_dim=10,
        dropout_rate=0.0,
    )
    
    # Get linear layers from the MLP model
    # First find all linear layers in the network
    linear_layers = []
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)
    
    # Set alignment layers for pruning (all linear layers)
    network.alignment_layers = linear_layers
    network.alignment_layer_names = [f"Layer {i}" for i in range(len(network.alignment_layers))]
    
    network.to(device)
    logger.info(f"Created MLP model with {len(network.alignment_layers)} alignment layers")
    
    # Train the network for a single epoch
    logger.info("Training network...")
    train_network(network, dataset, device)
    
    # Print statistics about the original model
    logger.info("Evaluating original model...")
    print_layer_stats(network)
    orig_accuracy, orig_loss = dataset.evaluate(network, device)
    logger.info(f"Original model - Accuracy: {orig_accuracy:.2f}%, Loss: {orig_loss:.4f}")
    
    # Save original weights for verification
    original_weights = {}
    for i, layer in enumerate(network.alignment_layers):
        if hasattr(layer, "weight") and layer.weight is not None:
            original_weights[i] = layer.weight.data.clone()
    
    # Test different pruning strategies
    logger.info("Testing pruning strategies...")
    pruning_percents = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Test each strategy individually to confirm they differ
    strategies = ["high_rq", "low_rq", "random"]
    all_results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} pruning strategy...")
        
        # Make a fresh copy of the network for each strategy
        network_copy = copy.deepcopy(network).to(device)
        
        # Verify the copy has the same weights
        for i, layer in enumerate(network_copy.alignment_layers):
            if i in original_weights:
                weight_diff = torch.sum(torch.abs(layer.weight.data - original_weights[i])).item()
                if weight_diff > 1e-5:
                    logger.error(f"Weights for layer {i} changed before pruning! Diff: {weight_diff}")
        
        # Test this specific strategy
        strategy_results = test_pruning_strategies(
            network_copy, 
            dataset, 
            pruning_percents=pruning_percents,
            device=device,
            strategy=strategy
        )
        
        # Store results for this strategy
        all_results[strategy] = strategy_results
        
        # Print results
        logger.info(f"{strategy} pruning results:")
        for i, percent in enumerate(pruning_percents):
            acc = strategy_results[f"{strategy}_acc"][i]
            loss = strategy_results[f"{strategy}_loss"][i]
            logger.info(f"  {percent*100:.0f}% pruned: Accuracy = {acc:.2f}%, Loss = {loss:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        results = all_results[strategy]
        plt.plot(
            [p*100 for p in pruning_percents],
            results[f"{strategy}_acc"],
            marker="o" if strategy == "high_rq" else ("s" if strategy == "low_rq" else "^"),
            label=f"{strategy}",
            linewidth=2
        )
    
    plt.xlabel("Pruning Percentage", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Pruning Strategies Comparison", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("pruning_strategies_debug.png", dpi=300, bbox_inches="tight")
    logger.info("Saved plot to pruning_strategies_debug.png")
    
    # Check if strategies are different
    if len(strategies) > 1:
        are_different = False
        for i in range(len(strategies)-1):
            for j in range(i+1, len(strategies)):
                s1, s2 = strategies[i], strategies[j]
                accs1 = all_results[s1][f"{s1}_acc"]
                accs2 = all_results[s2][f"{s2}_acc"]
                
                # Calculate difference
                diff = np.mean(np.abs(np.array(accs1) - np.array(accs2)))
                logger.info(f"Mean accuracy difference between {s1} and {s2}: {diff:.2f}%")
                
                if diff > 1.0:  # Consider different if average difference > 1%
                    are_different = True
        
        if are_different:
            logger.info("SUCCESS: Pruning strategies show different results!")
        else:
            logger.error("FAILURE: Pruning strategies show very similar results.")

def train_network(network, dataset, device, num_epochs=3):
    """Train a network for a specified number of epochs"""
    logger.info(f"Training a single network")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create DataLoader for training
        train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset, batch_size=1024, shuffle=True
        )
        
        # Train for one epoch
        network.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = network(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update running loss
            running_loss += loss.item()
            
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        # Evaluate on test set
        test_acc, test_loss = dataset.evaluate(network, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

def print_layer_stats(network):
    """Print statistics about each alignment layer in the network"""
    for i, layer in enumerate(network.alignment_layers):
        if hasattr(layer, "weight") and layer.weight is not None:
            weights = layer.weight.data
            total_weights = weights.numel()
            zero_weights = (weights == 0).sum().item()
            zero_percent = 100.0 * zero_weights / total_weights
            
            # Count pruned neurons (rows with all zeros)
            pruned_neurons = 0
            total_neurons = weights.size(0)
            for j in range(total_neurons):
                if torch.all(weights[j] == 0):
                    pruned_neurons += 1
            
            logger.info(f"Layer {i}: Shape {weights.shape}, zeros: {zero_weights}/{total_weights} ({zero_percent:.2f}%), pruned neurons: {pruned_neurons}/{total_neurons} ({100.0 * pruned_neurons / total_neurons:.2f}%)")

if __name__ == "__main__":
    main() 