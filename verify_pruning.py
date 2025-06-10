import os
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
import copy

# Add source to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.alignment.models.models import MLP
from src.alignment.datasets import load_dataset
from src.alignment.dropout import progressive_dropout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def print_layer_stats(network, layer_name="All Layers"):
    """Print detailed statistics about each layer in the network"""
    logger.info(f"\n=== {layer_name} ===")
    total_params = 0
    total_zeros = 0
    
    for i, layer in enumerate(network.alignment_layers):
        if hasattr(layer, "weight") and layer.weight is not None:
            weights = layer.weight.data
            total_weights = weights.numel()
            total_params += total_weights
            
            # Count zero weights
            zero_weights = (weights == 0).sum().item()
            zero_percent = 100.0 * zero_weights / total_weights
            total_zeros += zero_weights
            
            # Count pruned neurons (rows with all zeros)
            pruned_neurons = 0
            total_neurons = weights.size(0)
            for j in range(total_neurons):
                if torch.all(weights[j] == 0):
                    pruned_neurons += 1
            
            logger.info(f"Layer {i}: Shape {weights.shape}, zeros: {zero_weights}/{total_weights} ({zero_percent:.2f}%), pruned neurons: {pruned_neurons}/{total_neurons} ({100.0 * pruned_neurons / total_neurons:.2f}%)")
    
    if total_params > 0:
        logger.info(f"Total: {total_zeros}/{total_params} zeros ({100.0 * total_zeros / total_params:.2f}%)")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load MNIST dataset
    dataset_config = {
        "dataset_name": "MNIST",
        "data_path": "data",
        "batch_size": 128,
        "model_name": "mlp"
    }
    dataset = load_dataset(dataset_config, device=device)
    logger.info(f"Loaded MNIST dataset")
    
    # Create MLP model
    network = MLP(
        input_dim=784,  # 28x28 for MNIST
        num_hidden=[100, 100],
        output_dim=10,
        dropout_rate=0.0,
    )
    
    # Add alignment layers needed for pruning
    linear_layers = []
    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(module)
    
    network.alignment_layers = linear_layers
    network.alignment_names = [f"layer_{i}" for i in range(len(network.alignment_layers))]
    
    network.to(device)
    logger.info(f"Created model with {len(network.alignment_layers)} alignment layers")
    
    # Train the model briefly
    logger.info("Training network...")
    train_network(network, dataset, device)
    
    # Evaluate original model
    orig_accuracy, orig_loss = dataset.evaluate(network, device)
    logger.info(f"Original model - Accuracy: {orig_accuracy:.2f}%, Loss: {orig_loss:.4f}")
    
    # Print statistics of original model
    print_layer_stats(network, "ORIGINAL MODEL")
    
    # Make a copy of the model to preserve the original
    pruned_network = copy.deepcopy(network).to(device)
    
    # Define pruning parameters for testing
    pruning_modes = ["global_joint", "layer_wise"]
    strategies = ["high_rq", "low_rq", "random"]
    dropout_modes = ["scaled", "zero"]
    fractions = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Test each pruning configuration
    for pruning_mode in pruning_modes:
        for dropout_mode in dropout_modes:
            for strategy in strategies:
                logger.info(f"\n\n===== Testing {strategy} pruning with {pruning_mode} mode, {dropout_mode} dropout =====")
                
                # Create a fresh copy of the network
                test_network = copy.deepcopy(network).to(device)
                
                # Apply progressive dropout
                logger.info(f"Applying {strategy} pruning at 50% fraction...")
                
                # For debugging, show exact weight values for a specific part of the model
                sample_layer = 0
                sample_row = 0
                sample_weights = test_network.alignment_layers[sample_layer].weight.data[sample_row, :10].clone()
                logger.info(f"Sample weights before pruning: {sample_weights}")
                
                # Apply pruning
                _, _ = progressive_dropout(
                    networks=[test_network],
                    dataset=dataset,
                    dropout_fractions=[0.0, 0.5],  # Just test 0% and 50% pruning
                    metric=None,  # Will use default L2 norm
                    device=device,
                    pruning_mode=pruning_mode,
                    dropout_mode=dropout_mode,
                    strategy=strategy,
                    show_progress=False
                )
                
                # Show results after pruning
                logger.info(f"After pruning:")
                print_layer_stats(test_network, f"PRUNED MODEL ({strategy}, {pruning_mode}, {dropout_mode})")
                
                # Show sample weights after pruning
                sample_weights_after = test_network.alignment_layers[sample_layer].weight.data[sample_row, :10].clone()
                logger.info(f"Sample weights after pruning: {sample_weights_after}")
                
                # Verify if weights changed
                if torch.all(sample_weights == sample_weights_after):
                    logger.warning("WARNING: Sample weights did not change after pruning!")
                
                # Test the pruned model to check accuracy drop
                pruned_accuracy, pruned_loss = dataset.evaluate(test_network, device)
                
                # Check if pruning had an effect on accuracy
                accuracy_drop = orig_accuracy - pruned_accuracy
                logger.info(f"Pruned model - Accuracy: {pruned_accuracy:.2f}% (drop: {accuracy_drop:.2f}%), Loss: {pruned_loss:.4f}")
                
                # Get the weights after evaluation to verify they haven't changed
                sample_weights_eval = test_network.alignment_layers[sample_layer].weight.data[sample_row, :10].clone()
                if not torch.all(sample_weights_after == sample_weights_eval):
                    logger.error("ERROR: Weights changed during evaluation!")
                    logger.error(f"Before eval: {sample_weights_after}")
                    logger.error(f"After eval:  {sample_weights_eval}")
                
                # Manual verification of pruning by inspecting neuron activations
                logger.info("Testing if pruned neurons are actually inactive...")
                total_neurons = 0
                pruned_neurons = 0
                
                for layer_idx, layer in enumerate(test_network.alignment_layers):
                    if hasattr(layer, "weight") and layer.weight is not None:
                        weights = layer.weight.data
                        for neuron_idx in range(weights.shape[0]):
                            total_neurons += 1
                            if torch.all(weights[neuron_idx] == 0):
                                pruned_neurons += 1
                
                logger.info(f"Total pruned neurons: {pruned_neurons}/{total_neurons} ({100.0*pruned_neurons/total_neurons:.2f}%)")
                
                # Final test: verify actual neuron activations during forward pass
                test_network.eval()
                with torch.no_grad():
                    # Get a batch of data
                    batch_inputs, _ = next(iter(dataset.test_loader))
                    batch_inputs = batch_inputs.to(device)
                    
                    # Register hooks to capture activations
                    activations = {}
                    hooks = []
                    
                    def capture_activation(name):
                        def hook(module, input, output):
                            activations[name] = output.detach()
                        return hook
                    
                    # Register hook for each layer
                    for layer_idx, layer in enumerate(test_network.alignment_layers):
                        hooks.append(layer.register_forward_hook(capture_activation(f"layer_{layer_idx}")))
                    
                    # Forward pass
                    _ = test_network(batch_inputs)
                    
                    # Check activations
                    for layer_idx, layer in enumerate(test_network.alignment_layers):
                        layer_name = f"layer_{layer_idx}"
                        if layer_name in activations:
                            act = activations[layer_name]
                            
                            # Check if pruned neurons are actually not activating
                            weights = layer.weight.data
                            for neuron_idx in range(weights.shape[0]):
                                if torch.all(weights[neuron_idx] == 0):
                                    # This neuron should have zero activation
                                    if neuron_idx < act.shape[1]:  # Make sure index is valid
                                        neuron_act = act[:, neuron_idx]
                                        if not torch.all(neuron_act == 0):
                                            logger.error(f"Layer {layer_idx}, Neuron {neuron_idx} has non-zero activation despite being pruned!")
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()

def train_network(network, dataset, device, num_epochs=3):
    """Train a network for a specified number of epochs"""
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Train for one epoch
        network.train()
        for i, (inputs, targets) in enumerate(dataset.train_loader):
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
            
            # Only train on a few batches for speed
            if i >= 10:
                break
        
        # Print epoch statistics
        epoch_loss = running_loss / min(len(dataset.train_loader), 10)
        epoch_acc = 100. * correct / total
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        # Evaluate on test set
        test_acc, test_loss = dataset.evaluate(network, device)
        logger.info(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

if __name__ == "__main__":
    main() 