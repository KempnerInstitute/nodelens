#!/usr/bin/env python3
"""
Test script for training consolidation.

This demonstrates how experiments can use the new ExperimentTrainer.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from src.alignment.experiments.training_utils import (
    create_experiment_trainer, 
    train_with_metrics,
    convert_training_history
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=100, input_size=784, num_classes=10):
    """Create dummy dataset for testing."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def test_single_network_training():
    """Test training a single network."""
    logger.info("Testing single network training...")
    
    # Create model and data
    model = SimpleModel()
    train_dataset = create_dummy_data(200)
    val_dataset = create_dummy_data(50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create trainer from config
    config = {
        'training_epochs': 3,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'log_interval': 10,
    }
    
    trainer = create_experiment_trainer(model, config, device='cpu')
    
    # Train
    history = train_with_metrics(trainer, train_loader, val_loader)
    
    # Convert results
    results = convert_training_history(history, num_networks=1)
    
    logger.info(f"Training completed:")
    logger.info(f"  Final train loss: {results['final_train_loss']:.4f}")
    logger.info(f"  Final train accuracy: {results['final_train_accuracy']:.2f}%")
    logger.info(f"  Final val loss: {results.get('final_val_loss', 'N/A')}")
    logger.info(f"  Final val accuracy: {results.get('final_val_accuracy', 'N/A')}")
    
    return results


def test_multi_network_training():
    """Test training multiple networks."""
    logger.info("\nTesting multi-network training...")
    
    # Create multiple models
    num_networks = 3
    models = [SimpleModel() for _ in range(num_networks)]
    
    # Create data
    train_dataset = create_dummy_data(200)
    val_dataset = create_dummy_data(50)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Create trainer with multi-network config
    config = {
        'training_epochs': 3,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'num_networks': num_networks,
        'tensorized_training': True,
        'log_interval': 10,
    }
    
    trainer = create_experiment_trainer(models, config, device='cpu')
    
    # Train
    history = train_with_metrics(trainer, train_loader, val_loader)
    
    # Convert results
    results = convert_training_history(history, num_networks=num_networks)
    
    logger.info(f"Multi-network training completed:")
    logger.info(f"  Average final train loss: {results['final_train_loss']:.4f}")
    logger.info(f"  Average final train accuracy: {results['final_train_accuracy']:.2f}%")
    
    if 'per_network_results' in results:
        for i, network_results in results['per_network_results'].items():
            logger.info(f"  Network {i}:")
            logger.info(f"    Train loss: {network_results['final_train_loss']:.4f}")
            logger.info(f"    Train accuracy: {network_results['final_train_accuracy']:.2f}%")
    
    return results


def test_migration_example():
    """
    Example of how to migrate an experiment's _train_model method.
    
    This shows the before/after pattern for consolidation.
    """
    logger.info("\nDemonstrating migration pattern...")
    
    class OldExperiment:
        """Old style with custom _train_model."""
        def _train_model(self, model, train_loader, epochs=10):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
            return {'final_loss': avg_loss}
    
    class NewExperiment:
        """New style using ExperimentTrainer."""
        def __init__(self, config):
            self.config = config
            
        def _train_model(self, model, train_loader, val_loader=None):
            # Create trainer
            trainer = create_experiment_trainer(model, self.config)
            
            # Train with metrics
            history = train_with_metrics(trainer, train_loader, val_loader)
            
            # Convert to expected format
            return convert_training_history(history)
    
    # Demo usage
    model = SimpleModel()
    train_dataset = create_dummy_data(100)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    # New style
    config = {'training_epochs': 2, 'learning_rate': 0.001}
    new_exp = NewExperiment(config)
    results = new_exp._train_model(model, train_loader)
    
    logger.info("Migration example completed")
    logger.info(f"  Using new trainer: {results['training_epochs']} epochs trained")
    
    return results


if __name__ == "__main__":
    # Run tests
    test_single_network_training()
    test_multi_network_training()
    test_migration_example()
    
    logger.info("\nAll tests completed successfully!")
    logger.info("The ExperimentTrainer can replace custom training implementations.") 