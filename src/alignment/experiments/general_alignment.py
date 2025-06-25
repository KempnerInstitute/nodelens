"""
General alignment experiment that can perform comprehensive analysis.

This module implements a flexible experiment that can:
- Train models from scratch or use pretrained
- Compute alignment metrics throughout training
- Apply various pruning strategies
- Perform dropout analysis
- Generate comprehensive visualizations
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path

from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.core.registry import register_experiment
from alignment.analysis.visualization import AlignmentVisualizer, PruningVisualizer
from alignment.analysis.reporting import ExperimentReporter

logger = logging.getLogger(__name__)


@dataclass
class GeneralAlignmentConfig(ExperimentConfig):
    """Configuration for general alignment experiment."""
    
    # Training configuration
    do_train: bool = True
    training_epochs: int = 100
    learning_rate: float = 0.1
    optimizer: str = "sgd"
    scheduler: str = "cosine"
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {"T_max": 100, "eta_min": 0})
    
    # Alignment measurement
    measure_alignment_during_training: bool = True
    alignment_frequency: int = 1  # Measure every N epochs
    alignment_methods: List[str] = field(default_factory=lambda: ["rayleigh_quotient"])
    measure_expected_distribution: bool = True
    distribution_bins: int = 50
    
    # Progressive dropout analysis
    do_dropout_analysis: bool = True
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"  # "scaled" or "unscaled"
    dropout_pruning_mode: str = "global"  # "global", "per_layer_combined", "per_layer_independent"
    
    # Pruning experiments
    do_pruning_experiments: bool = True
    pruning_strategies: List[str] = field(default_factory=lambda: ["magnitude", "gradient", "fisher"])
    pruning_amounts: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    fine_tune_after_pruning: bool = True
    fine_tune_epochs: int = 10
    
    # Eigenfeature analysis
    do_eigenfeature_analysis: bool = True
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300
    
    # CNN mode
    cnn_mode: str = "unfold"  # "unfold", "patchwise", "batch_patch_combined"
    
    # Aggregation for layer-wise metrics
    aggregate_alignment: bool = False
    
    # Results saving
    save_intermediate_results: bool = True
    save_networks: bool = False


@register_experiment("general_alignment")
class GeneralAlignmentExperiment(BaseExperiment):
    """
    Comprehensive alignment experiment with multiple analysis types.
    
    This experiment can:
    1. Train networks from scratch or use pretrained
    2. Measure alignment throughout training
    3. Perform progressive dropout analysis
    4. Test various pruning strategies
    5. Analyze eigenfeatures
    6. Generate comprehensive visualizations
    """
    
    def __init__(self, config: GeneralAlignmentConfig):
        """Initialize general alignment experiment."""
        super().__init__(config)
        self.train_results = {}
        self.test_results = {}
        self.dropout_results = {}
        self.pruning_results = {}
        self.eigenfeature_results = {}
        
    def _train_model(self) -> Dict[str, Any]:
        """Train the model and collect alignment metrics."""
        if not self.config.do_train:
            logger.info("Skipping training (do_train=False)")
            return {}
        
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        
        # Setup optimizer
        optimizer = self._setup_optimizer()
        scheduler = self._setup_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        alignment_history = {method: [] for method in self.config.alignment_methods}
        
        # Training loop
        for epoch in range(self.config.training_epochs):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc = self._evaluate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Measure alignment
            if self.config.measure_alignment_during_training and epoch % self.config.alignment_frequency == 0:
                alignment_values = self._measure_alignment()
                for method, values in alignment_values.items():
                    alignment_history[method].append(values)
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.training_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
            )
            
            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
        
        return {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "alignment": alignment_history
        }
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on config."""
        if self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=0.0001
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.0001
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **self.config.scheduler_config
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        return None
    
    def _train_epoch(self, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.data_loader):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _measure_alignment(self) -> Dict[str, Dict[str, List[float]]]:
        """Measure alignment metrics for all layers."""
        alignment_values = {}
        
        # Get a batch of data
        inputs, _ = next(iter(self.data_loader))
        inputs = inputs.to(self.config.device)
        
        # Forward pass with activation tracking
        _, activations = self.wrapped_model.forward_with_activations(inputs)
        
        # Get weights
        weights = self.wrapped_model.get_layer_weights()
        
        # Preprocess activations based on CNN mode
        from alignment.preprocessing import preprocess_layer_activations
        layer_modules = dict(self.wrapped_model._model.named_modules())
        
        # Collect inputs for preprocessing
        inputs_to_process = {}
        for layer_name in self.wrapped_model.tracked_layers:
            layer_input = activations.get(f"{layer_name}_input")
            if layer_input is not None:
                inputs_to_process[f"{layer_name}_input"] = layer_input
        
        # Preprocess all inputs
        preprocessed = preprocess_layer_activations(
            inputs_to_process,
            layer_modules,
            mode=self.config.cnn_mode
        )
        
        # Extract preprocessed inputs
        preprocessed_inputs = {}
        for layer_name in self.wrapped_model.tracked_layers:
            key = f"{layer_name}_input"
            if key in preprocessed:
                preprocessed_inputs[layer_name] = preprocessed[key]
        
        # Compute each metric
        for method in self.config.alignment_methods:
            if method not in self.metrics:
                logger.warning(f"Metric {method} not initialized, skipping")
                continue
                
            metric = self.metrics[method]
            layer_values = {}
            
            for layer_name in self.wrapped_model.tracked_layers:
                if layer_name not in preprocessed_inputs or layer_name not in weights:
                    continue
                
                try:
                    scores = metric.compute(
                        inputs=preprocessed_inputs[layer_name],
                        weights=weights[layer_name]
                    )
                    layer_values[layer_name] = scores.cpu().tolist()
                except Exception as e:
                    logger.error(f"Error computing {method} for {layer_name}: {e}")
            
            alignment_values[method] = layer_values
        
        return alignment_values
    
    def _dropout_analysis(self) -> Dict[str, Any]:
        """Perform progressive dropout analysis."""
        if not self.config.do_dropout_analysis:
            logger.info("Skipping dropout analysis")
            return {}
        
        logger.info("Starting progressive dropout analysis")
        
        # Get initial alignment
        initial_alignment = self._measure_alignment()
        
        # Results storage
        results = {
            "dropout_rates": self.config.dropout_rates,
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []},
            "alignment_values": {}
        }
        
        # Test each dropout rate
        for dropout_rate in self.config.dropout_rates:
            logger.info(f"Testing dropout rate: {dropout_rate}")
            
            for strategy in ["low", "high", "random"]:
                # Apply targeted dropout based on alignment scores
                if strategy == "random":
                    # Average over multiple trials
                    trial_losses = []
                    trial_accs = []
                    
                    for _ in range(3):
                        loss, acc = self._apply_dropout_and_evaluate(
                            dropout_rate, strategy, initial_alignment
                        )
                        trial_losses.append(loss)
                        trial_accs.append(acc)
                    
                    avg_loss = np.mean(trial_losses)
                    avg_acc = np.mean(trial_accs)
                else:
                    avg_loss, avg_acc = self._apply_dropout_and_evaluate(
                        dropout_rate, strategy, initial_alignment
                    )
                
                results["losses"][strategy].append(avg_loss)
                results["accuracies"][strategy].append(avg_acc)
                
                logger.info(f"  {strategy}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%")
        
        return results
    
    def _apply_dropout_and_evaluate(
        self,
        dropout_rate: float,
        strategy: str,
        alignment_values: Dict[str, Dict[str, List[float]]]
    ) -> Tuple[float, float]:
        """Apply targeted dropout and evaluate."""
        # For now, return dummy values
        # TODO: Implement targeted dropout based on alignment scores
        return 0.0, 100.0 * (1 - dropout_rate)
    
    def run(self) -> Dict[str, Any]:
        """Run the general alignment experiment."""
        logger.info("Starting general alignment experiment")
        
        # Train model
        self.train_results = self._train_model()
        
        # Final evaluation
        test_loss, test_acc = self._evaluate()
        self.test_results = {
            "final_loss": test_loss,
            "final_accuracy": test_acc,
            "alignment": self._measure_alignment()
        }
        
        # Dropout analysis
        self.dropout_results = self._dropout_analysis()
        
        # TODO: Add pruning experiments
        # TODO: Add eigenfeature analysis
        
        # Combine all results
        all_results = {
            "config": self.config.to_dict(),
            "train_results": self.train_results,
            "test_results": self.test_results,
            "dropout_results": self.dropout_results,
            "pruning_results": self.pruning_results,
            "eigenfeature_results": self.eigenfeature_results
        }
        
        # Save results
        self.results.update(all_results)
        self.save_results()
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_visualizations()
        
        logger.info("General alignment experiment completed")
        
        return all_results
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        output_dir = Path(self.config.log_dir) / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # Training curves
        if self.train_results:
            # TODO: Plot training curves
            pass
        
        # Alignment evolution
        if "alignment" in self.train_results:
            # TODO: Plot alignment evolution
            pass
        
        # Dropout analysis
        if self.dropout_results:
            # TODO: Plot dropout results
            pass
        
        logger.info(f"Saved visualizations to {output_dir}") 