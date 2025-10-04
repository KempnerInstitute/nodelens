#!/usr/bin/env python
"""
Direct Pruning Test

This script tests pruning functionality directly, without relying on the complex
experiment infrastructure. It loads a model, applies pruning, and logs the results
including before/after weights and accuracies.
"""

import os
import sys
import logging
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('direct_pruning_test.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path if not already there
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import alignment modules
from alignment.models.registry import create_model
from alignment.datasets import load_dataset
from alignment.metrics import get_metric
from alignment.dropout import progressive_dropout
from alignment.config import ExperimentConfig

def count_zero_weights(model):
    """Count zero weights in the model."""
    zero_weights = 0
    total_weights = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_zeros = (param.data == 0).sum().item()
            layer_total = param.data.numel()
            zero_weights += layer_zeros
            total_weights += layer_total
            
            if hasattr(param, 'shape'):
                logger.info(f"Layer {name}: {layer_zeros}/{layer_total} zeros "
                          f"({100.0*layer_zeros/layer_total:.2f}% pruned), shape {param.shape}")
    
    if total_weights > 0:
        logger.info(f"Total: {zero_weights}/{total_weights} zeros "
                  f"({100.0*zero_weights/total_weights:.2f}% pruned)")
    
    return zero_weights, total_weights

def verify_pruning(config_path=None, model_name='mlp', dataset_name='mnist', strategy='low_rq',
                  pruning_mode='layer_wise', dropout_mode='unscaled', pruning_percent=0.5,
                  device=None):
    """Test pruning directly."""
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Fix dataset name casing - use uppercase for standard datasets
    if dataset_name.lower() == 'mnist':
        dataset_name = 'MNIST'
    elif dataset_name.lower() == 'cifar10':
        dataset_name = 'CIFAR10'
    elif dataset_name.lower() == 'cifar100':
        dataset_name = 'CIFAR100'
        
    # Define data path
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Create dataset config
    dataset_config = {
        'dataset_name': dataset_name,
        'batch_size': 128,
        'data_dir': data_root,
        'download': True
    }
    
    # Make sure data directory exists
    os.makedirs(data_root, exist_ok=True)
    
    dataset = load_dataset(dataset_config)
    
    # Get input dimension from dataset
    if dataset_name.lower() in ['mnist', 'fashion_mnist']:
        input_dim = 784  # 28*28
    else:
        # Try to infer from the dataset
        sample_data = next(iter(dataset.train_loader))[0]
        input_dim = sample_data[0].flatten().shape[0]
        logger.info(f"Inferred input dimension: {input_dim}")
    
    # Create a simple model
    class MLP(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
            # Define alignment layers for pruning
            self.alignment_layers = [self.fc1, self.fc2, self.fc3]
            self.alignment_names = ['layer_0', 'layer_1', 'layer_2']
            self.hidden = {}
            
        def forward(self, x):
            # Store inputs for hooking
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten
            
            # Forward with storing activations
            self.hidden['layer_0'] = x
            x = self.relu(self.fc1(x))
            
            self.hidden['layer_1'] = x
            x = self.relu(self.fc2(x))
            
            self.hidden['layer_2'] = x
            x = self.fc3(x)
            
            return x
            
    # Create model based on dataset
    model = MLP(input_dim=input_dim, hidden_dim=100, output_dim=10)
    model.to(device)
    
    # Check for alignment layers
    if not hasattr(model, 'alignment_layers'):
        logger.error("Model doesn't have alignment_layers attribute")
        return
    
    logger.info(f"Created model with {len(model.alignment_layers)} alignment layers")
    
    # Make a copy for pruning
    pruned_model = copy.deepcopy(model)
    pruned_model.to(device)
    
    # Make sure parameters are on device
    for param in pruned_model.parameters():
        param.data = param.data.to(device)
    
    # Evaluate original model
    model.eval()
    orig_accuracy, orig_loss = dataset.evaluate(model, device)
    logger.info(f"Original model: accuracy={orig_accuracy:.2f}%, loss={orig_loss:.4f}")
    
    # Check zero weights before pruning
    logger.info("Weights before pruning:")
    orig_zeros, orig_total = count_zero_weights(pruned_model)
    
    # Setup metric
    metric = get_metric('rq')
    
    # Apply pruning
    logger.info(f"Applying pruning: strategy={strategy}, mode={pruning_mode}, "
              f"dropout_mode={dropout_mode}, percent={pruning_percent*100:.1f}%")
    
    # Call progressive_dropout directly
    network_accuracies, network_losses = progressive_dropout(
        [pruned_model],
        dataset,
        [0.0, pruning_percent],  # Use just two fractions: 0% and the target %
        metric,
        device,
        pruning_mode=pruning_mode,
        dropout_mode=dropout_mode,
        strategy=strategy,
        show_progress=False
    )
    
    # Check if pruning was applied
    logger.info("Weights after pruning:")
    pruned_zeros, pruned_total = count_zero_weights(pruned_model)
    
    # Calculate how many weights were pruned
    weights_pruned = pruned_zeros - orig_zeros
    pruned_percent = 100.0 * weights_pruned / pruned_total if pruned_total > 0 else 0
    logger.info(f"Weights pruned: {weights_pruned}/{pruned_total} ({pruned_percent:.2f}%)")
    
    # Evaluate pruned model
    pruned_model.eval()
    pruned_accuracy, pruned_loss = dataset.evaluate(pruned_model, device)
    logger.info(f"Pruned model: accuracy={pruned_accuracy:.2f}%, loss={pruned_loss:.4f}")
    
    # Calculate accuracy change
    acc_change = pruned_accuracy - orig_accuracy
    logger.info(f"Accuracy change: {acc_change:.2f}% points")
    
    # Summary
    logger.info("\nSUMMARY:")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Pruning mode: {pruning_mode}")
    logger.info(f"Dropout mode: {dropout_mode}")
    logger.info(f"Pruning percentage: {pruning_percent*100:.1f}%")
    logger.info(f"Original accuracy: {orig_accuracy:.2f}%")
    logger.info(f"Pruned accuracy: {pruned_accuracy:.2f}%")
    logger.info(f"Accuracy change: {acc_change:.2f}% points")
    logger.info(f"Weights pruned: {weights_pruned}/{pruned_total} ({pruned_percent:.2f}%)")
    
    return {
        'orig_accuracy': orig_accuracy,
        'pruned_accuracy': pruned_accuracy,
        'acc_change': acc_change,
        'weights_pruned': weights_pruned,
        'total_weights': pruned_total,
        'pruned_percent': pruned_percent
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test pruning directly')
    parser.add_argument('--model', type=str, default='mlp', help='Model name')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--strategy', type=str, default='low_rq', 
                        choices=['high_rq', 'low_rq', 'random'], help='Pruning strategy')
    parser.add_argument('--pruning-mode', type=str, default='layer_wise',
                       choices=['global_joint', 'layer_wise', 'layer_isolated'],
                       help='Pruning mode')
    parser.add_argument('--dropout-mode', type=str, default='unscaled',
                       choices=['scaled', 'unscaled'], help='Dropout mode')
    parser.add_argument('--pruning-percent', type=float, default=0.5,
                       help='Pruning percentage (0.0-1.0)')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Convert percent to fraction if needed
    if args.pruning_percent > 1.0:
        args.pruning_percent /= 100.0
    
    # Run test
    verify_pruning(
        model_name=args.model,
        dataset_name=args.dataset,
        strategy=args.strategy,
        pruning_mode=args.pruning_mode,
        dropout_mode=args.dropout_mode,
        pruning_percent=args.pruning_percent,
        device=args.device
    )

if __name__ == '__main__':
    main() 