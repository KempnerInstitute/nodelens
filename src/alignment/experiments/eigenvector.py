"""
Eigenvector dropout experiment.

This module implements pruning based on PCA/eigendecomposition,
dropping neurons based on their contribution to principal components.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
import logging
from dataclasses import dataclass, field

from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.core.registry import register_experiment

logger = logging.getLogger(__name__)


@dataclass
class EigenvectorConfig(ExperimentConfig):
    """Configuration for eigenvector dropout experiment."""
    
    # Dropout configuration
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"
    
    # Eigenvector configuration
    compute_layer_pca: bool = True  # Whether to compute PCA per layer
    n_components_ratio: float = 0.99  # Variance ratio to keep
    eigenvector_strategy: str = "low"  # "low" = drop low eigenvalue components, "high" = drop high
    
    # Training configuration
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    
    # Evaluation
    eval_batches: Optional[int] = None
    exclude_classification_layer: bool = True
    num_random_trials: int = 3


@register_experiment("eigenvector_dropout")
class EigenvectorDropoutExperiment(BaseExperiment):
    """
    Experiment for eigenvector-based dropout.
    
    This experiment:
    1. Trains a model (if configured)
    2. Computes PCA/eigendecomposition of layer activations
    3. Prunes neurons based on their eigenvalue rankings
    4. Evaluates performance at different pruning levels
    """
    
    def __init__(self, config: EigenvectorConfig):
        """Initialize eigenvector dropout experiment."""
        super().__init__(config)
        self.config = config
        self.eigendecomposition = {}
        self.original_weights = {}
        
    def _compute_layer_eigendecomposition(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute eigendecomposition for each layer's activations.
        
        Returns:
            Dictionary mapping layer names to (eigenvalues, eigenvectors)
        """
        logger.info("Computing eigendecomposition for each layer")
        
        layer_eigen = {}
        eval_batches = self.config.eval_batches or len(self.data_loader)
        
        # Collect activations for each layer
        layer_activations = {
            name: [] for name in self.wrapped_model.tracked_layers
            if not (self.config.exclude_classification_layer and "classifier" in name.lower())
        }
        
        # Gather activations
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if batch_idx >= eval_batches:
                    break
                    
                inputs = inputs.to(self.config.device)
                
                # Forward pass
                _, activations = self.wrapped_model.forward_with_activations(inputs)
                
                # Collect layer outputs
                for layer_name in layer_activations:
                    layer_output = activations.get(f"{layer_name}_output")
                    if layer_output is not None:
                        # Flatten spatial dimensions for conv layers
                        if len(layer_output.shape) > 2:
                            layer_output = layer_output.flatten(2).mean(dim=2)
                        layer_activations[layer_name].append(layer_output.cpu())
        
        # Compute eigendecomposition for each layer
        for layer_name, act_list in layer_activations.items():
            if not act_list:
                logger.warning(f"No activations collected for layer {layer_name}")
                continue
                
            # Concatenate all activations
            all_activations = torch.cat(act_list, dim=0)  # [total_samples, features]
            
            # Compute covariance matrix
            activations_centered = all_activations - all_activations.mean(dim=0, keepdim=True)
            cov_matrix = torch.mm(activations_centered.t(), activations_centered) / (all_activations.size(0) - 1)
            
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (descending)
            idx = eigenvalues.argsort(descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            layer_eigen[layer_name] = (eigenvalues, eigenvectors)
            
            # Log variance explained
            total_var = eigenvalues.sum()
            if total_var > 0:
                var_explained = (eigenvalues.cumsum(0) / total_var)
                n_components = (var_explained <= self.config.n_components_ratio).sum() + 1
                logger.info(f"Layer {layer_name}: {n_components}/{len(eigenvalues)} components "
                           f"explain {self.config.n_components_ratio*100:.1f}% variance")
        
        return layer_eigen
    
    def _create_eigenvector_masks(
        self,
        eigendecomposition: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        dropout_rate: float,
        strategy: str = "low"
    ) -> Dict[str, torch.Tensor]:
        """
        Create dropout masks based on eigenvalue rankings.
        
        Args:
            eigendecomposition: Layer eigenvalues and eigenvectors
            dropout_rate: Fraction of neurons to drop
            strategy: "low" drops low eigenvalue neurons, "high" drops high
            
        Returns:
            Dictionary of masks for each layer
        """
        masks = {}
        
        for layer_name, (eigenvalues, eigenvectors) in eigendecomposition.items():
            num_neurons = len(eigenvalues)
            num_drop = int(num_neurons * dropout_rate)
            
            if num_drop == 0:
                masks[layer_name] = torch.ones(num_neurons, dtype=torch.bool)
                continue
            
            # Create mask based on eigenvalue ranking
            mask = torch.ones(num_neurons, dtype=torch.bool)
            
            if strategy == "low":
                # Drop neurons with lowest eigenvalues
                mask[-num_drop:] = False
            elif strategy == "high":
                # Drop neurons with highest eigenvalues
                mask[:num_drop] = False
            else:  # random
                random_indices = torch.randperm(num_neurons)[:num_drop]
                mask[random_indices] = False
            
            masks[layer_name] = mask
            
            logger.debug(f"Layer {layer_name}: keeping {mask.sum().item()}/{num_neurons} neurons")
        
        return masks
    
    def _project_weights_to_eigenspace(self, layer_name: str, eigenvectors: torch.Tensor):
        """
        Project layer weights to eigenspace.
        
        Args:
            layer_name: Name of the layer
            eigenvectors: Eigenvector matrix for the layer
        """
        layer = self.wrapped_model.get_layer(layer_name)
        if layer is None or not hasattr(layer, 'weight'):
            return
            
        # Store original weights
        if layer_name not in self.original_weights:
            self.original_weights[layer_name] = layer.weight.data.clone()
            if hasattr(layer, 'bias') and layer.bias is not None:
                self.original_weights[layer_name + "_bias"] = layer.bias.data.clone()
        
        # Project weights to eigenspace
        weight = layer.weight.data
        if len(weight.shape) == 2:  # Linear layer
            # Weight shape: [out_features, in_features]
            # Eigenvectors shape: [in_features, n_components]
            # We need to project the input dimension
            projected_weight = torch.mm(weight, eigenvectors)
            layer.weight.data = torch.mm(projected_weight, eigenvectors.t())
        elif len(weight.shape) == 4:  # Conv layer
            # For conv layers, we'll just apply the mask directly
            # A more sophisticated approach would reshape and project properly
            pass
    
    def _apply_eigenvector_masks(self, masks: Dict[str, torch.Tensor], eigendecomposition: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Apply eigenvector-based dropout masks."""
        for layer_name, mask in masks.items():
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is None:
                continue
                
            eigenvalues, eigenvectors = eigendecomposition[layer_name]
            
            # Store original weights
            if layer_name not in self.original_weights:
                self.original_weights[layer_name] = layer.weight.data.clone()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    self.original_weights[layer_name + "_bias"] = layer.bias.data.clone()
            
            # Apply mask in eigenspace
            if hasattr(layer, 'weight'):
                weight = self.original_weights[layer_name].clone()
                
                if len(weight.shape) == 2:  # Linear layer
                    # Keep only selected eigenvectors
                    selected_eigenvectors = eigenvectors[:, mask]
                    
                    # Project weights to reduced eigenspace and back
                    projected = torch.mm(weight, selected_eigenvectors)
                    layer.weight.data = torch.mm(projected, selected_eigenvectors.t())
                    
                    # Also mask the output neurons directly
                    layer.weight.data[~mask] = 0
                    
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data = self.original_weights[layer_name + "_bias"].clone()
                        layer.bias.data[~mask] = 0
                        
                elif len(weight.shape) == 4:  # Conv layer
                    # For conv layers, directly mask output channels
                    layer.weight.data = weight
                    layer.weight.data[~mask] = 0
                    
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data = self.original_weights[layer_name + "_bias"].clone()
                        layer.bias.data[~mask] = 0
    
    def _restore_original_weights(self):
        """Restore original weights."""
        for layer_name, original_weight in self.original_weights.items():
            if "_bias" in layer_name:
                continue
                
            layer = self.wrapped_model.get_layer(layer_name)
            if layer is not None and hasattr(layer, 'weight'):
                layer.weight.data = original_weight.clone()
                
                bias_key = layer_name + "_bias"
                if bias_key in self.original_weights and hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data = self.original_weights[bias_key].clone()
    
    def _evaluate_model(self) -> Tuple[float, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                if self.config.eval_batches and batch_idx >= self.config.eval_batches:
                    break
                    
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _train_model(self):
        """Train the model if configured."""
        if not self.config.train_before_dropout:
            logger.info("Skipping initial training")
            return
            
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Setup optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config.training_epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.data_loader):
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            avg_loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the eigenvector dropout experiment.
        
        Returns:
            Dictionary containing experiment results
        """
        logger.info("Starting eigenvector dropout experiment")
        
        # Train model
        self._train_model()
        
        # Compute eigendecomposition
        self.eigendecomposition = self._compute_layer_eigendecomposition()
        
        # Initialize results
        results = {
            "dropout_rates": self.config.dropout_rates,
            "eigenvalues": {
                layer: eigenvalues.tolist()[:20]  # Store top 20 eigenvalues
                for layer, (eigenvalues, _) in self.eigendecomposition.items()
            },
            "variance_explained": {},
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []}
        }
        
        # Compute variance explained
        for layer, (eigenvalues, _) in self.eigendecomposition.items():
            total_var = eigenvalues.sum()
            if total_var > 0:
                var_explained = (eigenvalues.cumsum(0) / total_var).tolist()[:20]
                results["variance_explained"][layer] = var_explained
        
        # Evaluate at each dropout rate
        for dropout_rate in self.config.dropout_rates:
            logger.info(f"Evaluating dropout rate: {dropout_rate}")
            
            # Evaluate each strategy
            for strategy in ["low", "high", "random"]:
                if strategy == "random":
                    # Average over multiple trials
                    trial_losses = []
                    trial_accs = []
                    
                    for trial in range(self.config.num_random_trials):
                        # Create masks
                        masks = self._create_eigenvector_masks(
                            self.eigendecomposition, dropout_rate, strategy
                        )
                        
                        # Apply masks
                        self._apply_eigenvector_masks(masks, self.eigendecomposition)
                        
                        # Evaluate
                        loss, acc = self._evaluate_model()
                        trial_losses.append(loss)
                        trial_accs.append(acc)
                        
                        # Restore weights
                        self._restore_original_weights()
                    
                    avg_loss = np.mean(trial_losses)
                    avg_acc = np.mean(trial_accs)
                    
                else:
                    # Create masks
                    masks = self._create_eigenvector_masks(
                        self.eigendecomposition, dropout_rate, strategy
                    )
                    
                    # Apply masks
                    self._apply_eigenvector_masks(masks, self.eigendecomposition)
                    
                    # Evaluate
                    avg_loss, avg_acc = self._evaluate_model()
                    
                    # Restore weights
                    self._restore_original_weights()
                
                # Store results
                results["losses"][strategy].append(avg_loss)
                results["accuracies"][strategy].append(avg_acc)
                
                logger.info(f"  {strategy}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")
                
                # Log metrics
                self.log_metrics(len(results["losses"][strategy]) - 1, {
                    f"{strategy}_loss": avg_loss,
                    f"{strategy}_accuracy": avg_acc,
                    "dropout_rate": dropout_rate
                })
        
        # Save results
        self.results.update(results)
        self.save_results()
        
        # Save checkpoint
        self.save_checkpoint(
            step=len(self.config.dropout_rates),
            metrics={"final_results": results}
        )
        
        logger.info("Eigenvector dropout experiment completed")
        
        return results 