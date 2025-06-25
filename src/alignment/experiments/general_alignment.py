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
    pruning_selection_mode: str = "low"  # Which weights to prune: "low", "high", "random"
    
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
        from alignment.data.processing import preprocess_layer_activations
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
    
    def _pruning_experiments(self) -> Dict[str, Any]:
        """Perform pruning experiments with various strategies."""
        if not self.config.do_pruning_experiments:
            logger.info("Skipping pruning experiments")
            return {}
        
        logger.info("Starting pruning experiments")
        
        # Import pruning utilities
        from alignment.pruning.strategies import MagnitudePruning
        from alignment.pruning.base import PruningConfig
        
        results = {
            "strategies": {},
            "final_model_performance": {}
        }
        
        # Save original model state
        original_state = self.model.state_dict()
        
        # Get selection modes to test (convert single value to list for consistency)
        selection_modes = self.config.pruning_selection_mode
        if not isinstance(selection_modes, list):
            selection_modes = [selection_modes]
        
        for strategy_name in self.config.pruning_strategies:
            logger.info(f"Testing pruning strategy: {strategy_name}")
            
            # If we have multiple selection modes, test each one
            for selection_mode in selection_modes:
                # Create a key that includes selection mode if testing multiple
                if len(selection_modes) > 1:
                    result_key = f"{strategy_name}_{selection_mode}"
                    logger.info(f"  Selection mode: {selection_mode}")
                else:
                    result_key = strategy_name
                
                strategy_results = {
                    "pruning_amounts": [],
                    "accuracies_before_finetune": [],
                    "losses_before_finetune": [],
                    "accuracies_after_finetune": [],
                    "losses_after_finetune": [],
                    "sparsities": [],
                    "weight_distributions_before": [],
                    "weight_distributions_after": []
                }
                
                for amount in self.config.pruning_amounts:
                    logger.info(f"    Pruning amount: {amount * 100:.0f}%")
                    
                    # Remove any existing pruning masks before resetting
                    for name, module in self.model.named_modules():
                        if hasattr(module, 'weight_mask'):
                            delattr(module, 'weight_mask')
                        if hasattr(module, '_pruning_hook'):
                            module._pruning_hook.remove()
                            delattr(module, '_pruning_hook')
                    
                    # Reset model to original state
                    self.model.load_state_dict(original_state)
                    
                    # Create pruning strategy with config
                    pruning_config = PruningConfig(
                        amount=amount,
                        global_pruning=True,
                        pruning_mode=selection_mode  # Use current selection mode
                    )
                    
                                    # Import additional strategies if needed
                strategy = None
                
                if strategy_name == "magnitude":
                    strategy = MagnitudePruning(config=pruning_config)
                elif strategy_name in ["alignment", "rayleigh_quotient"]:
                    from alignment.pruning.strategies import AlignmentPruning
                    strategy = AlignmentPruning(
                        metric='rayleigh_quotient',
                        config=pruning_config
                    )
                elif strategy_name == "hybrid":
                    from alignment.pruning.strategies import HybridPruning
                    strategy = HybridPruning(
                        alignment_metric='rayleigh_quotient',
                        alpha=0.5,  # Equal weighting
                        config=pruning_config
                    )
                elif strategy_name == "gradient":
                    from alignment.pruning.strategies import GradientPruning
                    strategy = GradientPruning(config=pruning_config)
                elif strategy_name == "fisher":
                    from alignment.pruning.strategies import FisherPruning
                    strategy = FisherPruning(config=pruning_config)
                else:
                    logger.warning(f"Unsupported pruning strategy: {strategy_name}")
                    continue
                
                # Get sample inputs for alignment-based pruning
                sample_inputs = None
                if strategy_name in ["alignment", "rayleigh_quotient", "hybrid"]:
                    # Get a batch of data for alignment computation
                    data_iter = iter(self.data_loader)
                    sample_batch, _ = next(data_iter)
                    sample_inputs = sample_batch.to(self.config.device)
                    
                                    # Apply pruning to each layer
                layer_sparsities = {}
                for name, module in self.model.named_modules():
                    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                        # For alignment-based pruning, we need to get layer inputs
                        layer_inputs = None
                        if sample_inputs is not None:
                            # Forward pass to get layer inputs
                            # This is a simplified approach - in practice you might want
                            # to use hooks to capture the exact inputs to each layer
                            with torch.no_grad():
                                # For now, pass the sample inputs to all layers
                                # A more sophisticated approach would track activations
                                layer_inputs = sample_inputs
                        
                        # Prune this layer
                        strategy.prune(module, inputs=layer_inputs)
                        # Get sparsity
                        sparsity = strategy.get_sparsity(module)
                        layer_sparsities[name] = sparsity
                    
                    # Calculate overall sparsity
                    total_params = 0
                    zero_params = 0
                    for module in self.model.modules():
                        if hasattr(module, 'weight'):
                            total_params += module.weight.numel()
                            zero_params += (module.weight == 0).sum().item()
                    
                    overall_sparsity = zero_params / total_params if total_params > 0 else 0
                    
                    # Evaluate pruned model BEFORE fine-tuning
                    test_loss_before, test_acc_before = self._evaluate()
                    logger.info(f"      Before fine-tuning: Loss={test_loss_before:.4f}, Accuracy={test_acc_before:.2f}%")
                    
                    # Capture weight distribution before fine-tuning
                    weight_dist_before = self._get_weight_distribution()
                    
                    # Store before fine-tuning results
                    strategy_results["pruning_amounts"].append(amount)
                    strategy_results["accuracies_before_finetune"].append(test_acc_before)
                    strategy_results["losses_before_finetune"].append(test_loss_before)
                    strategy_results["sparsities"].append(overall_sparsity)
                    strategy_results["weight_distributions_before"].append(weight_dist_before)
                    
                    # Fine-tune if configured
                    test_loss_after = test_loss_before
                    test_acc_after = test_acc_before
                    
                    if self.config.fine_tune_after_pruning:
                        logger.info(f"      Fine-tuning for {self.config.fine_tune_epochs} epochs")
                        
                        # Setup optimizer for fine-tuning
                        optimizer = torch.optim.Adam(
                            self.model.parameters(),
                            lr=self.config.learning_rate * 0.1  # Lower learning rate
                        )
                        
                        # Track fine-tuning progress
                        finetune_losses = []
                        finetune_accs = []
                        
                        for epoch in range(self.config.fine_tune_epochs):
                            train_loss, train_acc = self._train_epoch(
                                optimizer,
                                nn.CrossEntropyLoss()
                            )
                            finetune_losses.append(train_loss)
                            finetune_accs.append(train_acc)
                            
                            if (epoch + 1) % 5 == 0:
                                logger.info(f"        Fine-tune epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                        
                        # Re-evaluate after fine-tuning
                        test_loss_after, test_acc_after = self._evaluate()
                        logger.info(f"      After fine-tuning: Loss={test_loss_after:.4f}, Accuracy={test_acc_after:.2f}%")
                        
                        # Capture weight distribution after fine-tuning
                        weight_dist_after = self._get_weight_distribution()
                    else:
                        weight_dist_after = weight_dist_before
                    
                    # Store after fine-tuning results
                    strategy_results["accuracies_after_finetune"].append(test_acc_after)
                    strategy_results["losses_after_finetune"].append(test_loss_after)
                    strategy_results["weight_distributions_after"].append(weight_dist_after)
                    
                    # Log improvement
                    acc_improvement = test_acc_after - test_acc_before
                    logger.info(f"      Results: Sparsity={overall_sparsity:.2%}, "
                              f"Acc before={test_acc_before:.2f}%, Acc after={test_acc_after:.2f}%, "
                              f"Improvement={acc_improvement:+.2f}%")
                
                results["strategies"][result_key] = strategy_results
        
        # Final cleanup: remove any remaining pruning masks
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                delattr(module, 'weight_mask')
            if hasattr(module, '_pruning_hook'):
                module._pruning_hook.remove()
                delattr(module, '_pruning_hook')
        
        # Restore original model
        self.model.load_state_dict(original_state)
        
        return results
    
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
        
        # Pruning experiments
        self.pruning_results = self._pruning_experiments()
        
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
    
    def _get_weight_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Get weight distribution statistics for each layer."""
        weight_stats = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.detach().cpu()
                
                # Get the actual weight values (not masked values)
                if hasattr(module, 'weight_mask'):
                    # For pruned weights, we want to see the non-zero values
                    mask = module.weight_mask.detach().cpu()
                    non_zero_weights = weight[mask != 0]
                else:
                    non_zero_weights = weight.flatten()
                
                if len(non_zero_weights) > 0:
                    weight_stats[name] = {
                        'mean': float(non_zero_weights.mean()),
                        'std': float(non_zero_weights.std()),
                        'min': float(non_zero_weights.min()),
                        'max': float(non_zero_weights.max()),
                        'percentiles': {
                            '1': float(torch.quantile(non_zero_weights, 0.01)),
                            '25': float(torch.quantile(non_zero_weights, 0.25)),
                            '50': float(torch.quantile(non_zero_weights, 0.50)),
                            '75': float(torch.quantile(non_zero_weights, 0.75)),
                            '99': float(torch.quantile(non_zero_weights, 0.99))
                        },
                        'sparsity': float((weight == 0).sum()) / weight.numel() if weight.numel() > 0 else 0
                    }
        
        return weight_stats
    
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
        
        # Pruning experiments - now enhanced with before/after comparisons
        if self.pruning_results and "strategies" in self.pruning_results:
            from alignment.analysis.visualization.pruning_plots import PruningVisualizer
            import matplotlib.pyplot as plt
            
            for strategy_name, strategy_results in self.pruning_results["strategies"].items():
                if not strategy_results.get("pruning_amounts"):
                    continue
                    
                # Create visualizer
                visualizer = PruningVisualizer()
                
                # 1. Plot accuracy comparison before/after fine-tuning
                fig = visualizer.plot_accuracy_vs_sparsity_comparison(
                    sparsities=strategy_results["sparsities"],
                    accuracies_before=strategy_results["accuracies_before_finetune"],
                    accuracies_after=strategy_results["accuracies_after_finetune"],
                    title=f"{strategy_name.capitalize()} Pruning: Before vs After Fine-tuning"
                )
                fig.savefig(output_dir / f"pruning_{strategy_name}_accuracy_comparison.png", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
                plt.close(fig)
                
                # 2. Plot fine-tuning improvement
                improvements = [
                    after - before 
                    for before, after in zip(
                        strategy_results["accuracies_before_finetune"],
                        strategy_results["accuracies_after_finetune"]
                    )
                ]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(improvements)), improvements, 
                      tick_label=[f"{s:.0%}" for s in strategy_results["sparsities"]])
                ax.set_xlabel("Sparsity Level")
                ax.set_ylabel("Accuracy Improvement (%)")
                ax.set_title(f"{strategy_name.capitalize()} Pruning: Fine-tuning Improvement")
                ax.grid(True, alpha=0.3)
                fig.savefig(output_dir / f"pruning_{strategy_name}_improvement.png",
                           dpi=self.config.plot_dpi, bbox_inches='tight')
                plt.close(fig)
                
                # 3. Weight distribution evolution (if available)
                if strategy_results.get("weight_distributions_before"):
                    # Plot weight distribution changes for highest sparsity
                    idx = -1  # Last (highest) sparsity
                    fig = visualizer.plot_weight_distribution_comparison(
                        weights_before=strategy_results["weight_distributions_before"][idx],
                        weights_after=strategy_results["weight_distributions_after"][idx],
                        sparsity=strategy_results["sparsities"][idx],
                        title=f"{strategy_name.capitalize()} Pruning at {strategy_results['sparsities'][idx]:.0%} Sparsity"
                    )
                    fig.savefig(output_dir / f"pruning_{strategy_name}_weight_dist_comparison.png",
                               dpi=self.config.plot_dpi, bbox_inches='tight')
                    plt.close(fig)
        
        logger.info(f"Saved visualizations to {output_dir}") 