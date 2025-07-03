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
    
    # Multi-network configuration
    num_networks: int = 1  # Number of networks to train (1 = single network, >1 = parallel)
    parallel_batch_size: Optional[int] = None  # Batch size for parallel training (None = use batch_size)
    use_tensorized_training: bool = True  # Use efficient tensorized ops when possible
    aggregate_metrics: bool = True  # Aggregate metrics across networks
    save_individual_networks: bool = False  # Save each network separately
    
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
    pruning_strategies: List[str] = field(default_factory=lambda: ["magnitude", "random", "alignment"])
    pruning_amounts: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    fine_tune_after_pruning: bool = True
    fine_tune_epochs: int = 10
    pruning_selection_mode: str = "low"  # Which weights to prune: "low", "high", "random"
    pruning_alignment_metric: str = "rayleigh_quotient"  # Metric for alignment-based pruning
    pruning_hybrid_alpha: float = 0.5  # Weight for alignment in hybrid pruning (0-1)
    pruning_scope: str = "layer"  # "global" or "layer" - how to select neurons/weights for pruning
    
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
        
        # Multi-network mode
        self.is_multi_network = config.num_networks > 1
        if self.is_multi_network:
            self.networks = []
            self.wrapped_networks = []
            self._initialize_multiple_networks()
            # Setup parallel processing
            import multiprocessing
            self.num_workers = min(config.num_networks, multiprocessing.cpu_count())
            logger.info(f"Multi-network mode: {config.num_networks} networks, {self.num_workers} workers")
    
    def _initialize_multiple_networks(self):
        """Initialize multiple networks with different seeds."""
        base_seed = self.config.seed
        
        for i in range(self.config.num_networks):
            # Set unique seed for each network
            seed = base_seed + i
            self._set_seed(seed)
            
            # Create a new model instance
            # Store original model/wrapped_model temporarily
            original_model = self.model
            original_wrapped = self.wrapped_model
            
            # Initialize new model (this sets self.model and self.wrapped_model)
            self._initialize_model()
            
            # Store the new model
            self.networks.append(self.model)
            self.wrapped_networks.append(self.wrapped_model)
            
            # Restore original references
            self.model = original_model
            self.wrapped_model = original_wrapped
            
            logger.info(f"Initialized network {i+1}/{self.config.num_networks} with seed {seed}")
        
        # For multi-network mode, model/wrapped_model will be None
        self.model = None
        self.wrapped_model = None
        
    def _train_model(self) -> Dict[str, Any]:
        """Train the model and collect alignment metrics."""
        if not self.config.do_train:
            logger.info("Skipping training (do_train=False)")
            return {}
        
        if self.is_multi_network:
            return self._train_multiple_networks()
        else:
            return self._train_single_network()
    
    def _train_single_network(self) -> Dict[str, Any]:
        """Train a single network (original implementation)."""
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
    
    def _train_multiple_networks(self) -> Dict[str, Any]:
        """Train multiple networks in parallel."""
        logger.info(f"Training {self.config.num_networks} networks for {self.config.training_epochs} epochs")
        
        # Determine batch size for parallel training
        batch_size = self.config.parallel_batch_size or self.config.batch_size
        
        if self.config.use_tensorized_training and self.config.num_networks <= 8:
            # Use tensorized training for efficiency
            return self._train_networks_tensorized(batch_size)
        else:
            # Use multiprocessing for larger numbers of networks
            return self._train_networks_parallel(batch_size)
    
    def _train_networks_tensorized(self, batch_size: int) -> Dict[str, Any]:
        """Train networks using tensorized operations."""
        from alignment.training.multi_network import train_networks_fully_tensorized
        
        # Setup optimizer class and kwargs
        if self.config.optimizer.lower() == "sgd":
            optimizer_class = torch.optim.SGD
            optimizer_kwargs = {
                "lr": self.config.learning_rate,
                "momentum": 0.9,
                "weight_decay": 0.0001
            }
        else:
            optimizer_class = torch.optim.Adam
            optimizer_kwargs = {
                "lr": self.config.learning_rate,
                "weight_decay": 0.0001
            }
        
        # Callbacks for alignment measurement
        alignment_histories = [{method: [] for method in self.config.alignment_methods} 
                              for _ in range(self.config.num_networks)]
        
        def alignment_callback(model, epoch, batch_idx):
            if self.config.measure_alignment_during_training and epoch % self.config.alignment_frequency == 0:
                if batch_idx == 0:  # Only measure at start of epoch
                    # Measure alignment for each network
                    for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
                        # Temporarily set as current model for measurement
                        self.model = net
                        self.wrapped_model = wrapped_net
                        alignment_values = self._measure_alignment()
                        for method, values in alignment_values.items():
                            alignment_histories[i][method].append(values)
                    # Reset
                    self.model = None
                    self.wrapped_model = None
        
        # Train networks
        trained_networks, history = train_networks_fully_tensorized(
            networks=self.networks,
            train_loader=self.data_loader,
            val_loader=None,  # TODO: Add validation loader support
            epochs=self.config.training_epochs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            device=self.config.device,
            checkpoint_dir=Path(self.config.checkpoint_dir) if self.config.save_intermediate_results else None,
            log_interval=self.config.log_interval,
            eval_interval=self.config.checkpoint_interval,
            callbacks=[alignment_callback] if self.config.measure_alignment_during_training else None
        )
        
        # Update networks with trained versions
        self.networks = trained_networks
        
        # Aggregate results
        if self.config.aggregate_metrics:
            # Average metrics across networks
            aggregated_history = {
                "train_losses": history["train_loss"],
                "train_accs": history["train_acc"],
                "val_losses": history["val_loss"],
                "val_accs": history["val_acc"],
                "alignment": {}
            }
            
            # Aggregate alignment metrics
            for method in self.config.alignment_methods:
                method_values = {}
                for layer in self.wrapped_networks[0].tracked_layers:
                    # Collect values from all networks for this layer
                    all_values = []
                    for net_history in alignment_histories:
                        if method in net_history and net_history[method]:
                            for epoch_values in net_history[method]:
                                if layer in epoch_values:
                                    all_values.extend(epoch_values[layer])
                    if all_values:
                        method_values[layer] = all_values
                
                if method_values:
                    aggregated_history["alignment"][method] = method_values
            
            return aggregated_history
        else:
            # Return individual results
            return {
                "networks": [
                    {
                        "train_losses": history["train_loss"],
                        "train_accs": history["train_acc"],
                        "val_losses": history["val_loss"],
                        "val_accs": history["val_acc"],
                        "alignment": alignment_histories[i]
                    }
                    for i in range(self.config.num_networks)
                ]
            }
    
    def _train_networks_parallel(self, batch_size: int) -> Dict[str, Any]:
        """Train networks using multiprocessing (for larger numbers)."""
        # TODO: Implement multiprocessing version
        logger.warning("Multiprocessing training not yet implemented, falling back to sequential")
        
        # Train each network sequentially for now
        all_results = []
        
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            logger.info(f"Training network {i+1}/{self.config.num_networks}")
            
            # Set current network
            self.model = net
            self.wrapped_model = wrapped_net
            
            # Train
            result = self._train_single_network()
            all_results.append(result)
            
        # Reset
        self.model = None
        self.wrapped_model = None
        
        # Aggregate or return individual results
        if self.config.aggregate_metrics:
            return self._aggregate_training_results(all_results)
        else:
            return {"networks": all_results}
    
    def _aggregate_training_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate training results from multiple networks."""
        import numpy as np
        
        # Average numerical metrics
        aggregated = {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "alignment": {}
        }
        
        # Get number of epochs from first result
        num_epochs = len(results[0]["train_losses"])
        
        # Average losses and accuracies
        for epoch in range(num_epochs):
            aggregated["train_losses"].append(
                np.mean([r["train_losses"][epoch] for r in results])
            )
            aggregated["train_accs"].append(
                np.mean([r["train_accs"][epoch] for r in results])
            )
            aggregated["val_losses"].append(
                np.mean([r["val_losses"][epoch] for r in results])
            )
            aggregated["val_accs"].append(
                np.mean([r["val_accs"][epoch] for r in results])
            )
        
        # Aggregate alignment metrics
        for method in self.config.alignment_methods:
            method_values = {}
            
            # Get all layers
            if results[0]["alignment"] and method in results[0]["alignment"]:
                sample_alignment = results[0]["alignment"][method]
                if sample_alignment:  # Check if not empty
                    layers = list(sample_alignment[0].keys()) if isinstance(sample_alignment, list) else []
                    
                    for layer in layers:
                        # Collect all values for this layer across networks and epochs
                        all_values = []
                        for result in results:
                            if method in result["alignment"]:
                                for epoch_values in result["alignment"][method]:
                                    if layer in epoch_values:
                                        all_values.extend(epoch_values[layer])
                        
                        if all_values:
                            method_values[layer] = all_values
            
            if method_values:
                aggregated["alignment"][method] = method_values
        
        return aggregated
    
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
        
        if self.is_multi_network:
            return self._dropout_analysis_multi()
        else:
            return self._dropout_analysis_single()
    
    def _dropout_analysis_single(self) -> Dict[str, Any]:
        """Perform dropout analysis on a single network."""
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
    
    def _dropout_analysis_multi(self) -> Dict[str, Any]:
        """Perform dropout analysis on multiple networks."""
        all_results = []
        
        # Run dropout analysis for each network
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            logger.info(f"Dropout analysis for network {i+1}/{self.config.num_networks}")
            
            # Temporarily set as current model
            self.model = net
            self.wrapped_model = wrapped_net
            
            # Run analysis
            result = self._dropout_analysis_single()
            all_results.append(result)
        
        # Reset
        self.model = None
        self.wrapped_model = None
        
        # Aggregate results
        if self.config.aggregate_metrics:
            return self._aggregate_dropout_results(all_results)
        else:
            return {"networks": all_results}
    
    def _aggregate_dropout_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate dropout results from multiple networks."""
        import numpy as np
        
        # Structure: same as single network but with averaged values
        aggregated = {
            "dropout_rates": results[0]["dropout_rates"],
            "accuracies": {"low": [], "high": [], "random": []},
            "losses": {"low": [], "high": [], "random": []},
            "alignment_values": {}
        }
        
        # Average accuracies and losses
        for strategy in ["low", "high", "random"]:
            for i in range(len(results[0]["dropout_rates"])):
                acc_values = [r["accuracies"][strategy][i] for r in results]
                loss_values = [r["losses"][strategy][i] for r in results]
                
                aggregated["accuracies"][strategy].append(np.mean(acc_values))
                aggregated["losses"][strategy].append(np.mean(loss_values))
        
        return aggregated
    
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
        
        if self.is_multi_network:
            return self._pruning_experiments_multi()
        else:
            return self._pruning_experiments_single()
    
    def _pruning_experiments_single(self) -> Dict[str, Any]:
        """Perform pruning experiments on a single network."""
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
                        if hasattr(module, '_original_weight'):
                            delattr(module, '_original_weight')
                    
                    # Reset model to original state
                    self.model.load_state_dict(original_state)
                    
                    # Create pruning strategy with config
                    pruning_config = PruningConfig(
                        amount=amount,
                        global_pruning=self.config.pruning_scope == 'global' if hasattr(self.config, 'pruning_scope') else False,
                        pruning_mode=selection_mode,  # Use current selection mode
                        structured=True if strategy_name in ["alignment", "hybrid"] else False  # Structured for alignment
                    )
                    
                    # Import additional strategies if needed
                    strategy = None
                    
                    if strategy_name == "magnitude":
                        if pruning_config.global_pruning:
                            from alignment.pruning.strategies import GlobalMagnitudePruning
                            strategy = GlobalMagnitudePruning(config=pruning_config)
                        else:
                            strategy = MagnitudePruning(config=pruning_config)
                    elif strategy_name == "alignment":
                        from alignment.pruning.strategies import AlignmentPruning, GlobalAlignmentPruning, CascadingAlignmentPruning
                        # Get the alignment metric from config (default to rayleigh_quotient)
                        alignment_metric = getattr(self.config, 'pruning_alignment_metric', 'rayleigh_quotient')
                        
                        if self.config.pruning_scope == 'global':
                            strategy = GlobalAlignmentPruning(
                                metric=alignment_metric,
                                config=pruning_config
                            )
                        elif self.config.pruning_scope == 'cascading':
                            # Cascading always uses structured pruning
                            pruning_config.structured = True
                            strategy = CascadingAlignmentPruning(
                                metric=alignment_metric,
                                direction=getattr(self.config, 'cascading_direction', 'forward'),
                                config=pruning_config
                            )
                        else:  # layer scope (default)
                            strategy = AlignmentPruning(
                                metric=alignment_metric,
                                config=pruning_config
                            )
                    elif strategy_name == "cascading_alignment":
                        # Legacy cascading_alignment handling
                        logger.warning("'cascading_alignment' algorithm is deprecated. Use algorithms=['alignment'] with scope='cascading'")
                        from alignment.pruning.strategies import CascadingAlignmentPruning
                        alignment_metric = getattr(self.config, 'pruning_alignment_metric', 'rayleigh_quotient')
                        pruning_config.structured = True
                        strategy = CascadingAlignmentPruning(
                            metric=alignment_metric,
                            direction='forward',
                            config=pruning_config
                        )
                    elif strategy_name == "hybrid":
                        from alignment.pruning.strategies import HybridPruning
                        alignment_metric = getattr(self.config, 'pruning_alignment_metric', 'rayleigh_quotient')
                        alpha = getattr(self.config, 'pruning_hybrid_alpha', 0.5)
                        # Note: Hybrid doesn't have a global variant yet
                        strategy = HybridPruning(
                            alignment_metric=alignment_metric,
                            alpha=alpha,
                            config=pruning_config
                        )
                    elif strategy_name == "gradient":
                        logger.warning("Gradient pruning is not suitable for post-training pruning on converged models")
                        from alignment.pruning.strategies import GradientPruning
                        strategy = GradientPruning(config=pruning_config)
                    elif strategy_name == "fisher":
                        logger.warning("Fisher pruning is not suitable for post-training pruning on converged models")
                        from alignment.pruning.strategies import FisherPruning
                        strategy = FisherPruning(config=pruning_config)
                    elif strategy_name == "random":
                        from alignment.pruning.strategies import RandomPruning
                        strategy = RandomPruning(config=pruning_config)
                    else:
                        logger.warning(f"Unsupported pruning strategy: {strategy_name}")
                        continue
                    
                    # Get sample inputs for alignment-based pruning
                    layer_inputs_dict = {}
                    if strategy_name in ["alignment", "hybrid", "cascading_alignment"]:
                        # Get a batch of data for alignment computation
                        data_iter = iter(self.data_loader)
                        sample_batch, _ = next(data_iter)
                        sample_inputs = sample_batch.to(self.config.device)
                        
                        # For alignment-based pruning, we ALWAYS need inputs for all layers
                        # (not just for global pruning)
                        # Use hooks to capture inputs for all layers
                        hooks = []
                        
                        def capture_input(name):
                            def hook(module, input, output):
                                layer_inputs_dict[name] = input[0].detach()
                            return hook
                        
                        # Register hooks
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                                hook = module.register_forward_hook(capture_input(name))
                                hooks.append(hook)
                        
                        # Forward pass to capture inputs
                        with torch.no_grad():
                            _ = self.model(sample_inputs)
                        
                        # Remove hooks
                        for hook in hooks:
                            hook.remove()
                        
                        logger.info(f"      Captured inputs for {len(layer_inputs_dict)} layers")
                    
                    # Apply pruning
                    if pruning_config.global_pruning and hasattr(strategy, 'prune_model'):
                        # Global pruning across all layers
                        if strategy_name == "alignment":
                            # Global alignment pruning needs layer inputs
                            masks = strategy.prune_model(self.model, layer_inputs_dict, amount=amount)
                        else:
                            # Global magnitude pruning
                            masks = strategy.prune_model(self.model, amount=amount)
                        
                        # Calculate overall sparsity from masks
                        total_params = sum(mask.numel() for mask in masks.values())
                        zero_params = sum((mask == 0).sum().item() for mask in masks.values())
                        overall_sparsity = zero_params / total_params if total_params > 0 else 0
                        
                    elif self.config.pruning_scope == 'cascading' and strategy_name == "alignment":
                        # Cascading alignment needs special handling
                        
                        # TODO: Extend cascading to other algorithms (magnitude, gradient, etc)
                        # For now, cascading only works with alignment-based pruning
                        
                        # Create a function to get current layer inputs
                        def get_layer_inputs_fn():
                            # Capture current inputs with hooks
                            current_inputs = {}
                            hooks = []
                            
                            def capture_input(name):
                                def hook(module, input, output):
                                    current_inputs[name] = input[0].detach()
                                return hook
                            
                            # Register hooks
                            for name, module in self.model.named_modules():
                                if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                                    hook = module.register_forward_hook(capture_input(name))
                                    hooks.append(hook)
                            
                            # Forward pass
                            with torch.no_grad():
                                _ = self.model(sample_inputs)
                            
                            # Remove hooks
                            for hook in hooks:
                                hook.remove()
                            
                            return current_inputs
                        
                        # Apply cascading pruning
                        masks = strategy.prune_model(self.model, get_layer_inputs_fn, amount=amount)
                        
                        # Calculate overall sparsity
                        total_params = sum(mask.numel() for mask in masks.values())
                        zero_params = sum((mask == 0).sum().item() for mask in masks.values())
                        overall_sparsity = zero_params / total_params if total_params > 0 else 0
                    else:
                        # Layer-wise pruning (current behavior)
                        layer_sparsities = {}
                        for name, module in self.model.named_modules():
                            if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
                                # For alignment-based pruning, we need layer inputs
                                layer_inputs = None
                                if strategy_name in ["alignment", "hybrid", "cascading_alignment"]:
                                    if name in layer_inputs_dict:
                                        layer_inputs = layer_inputs_dict[name]
                                    else:
                                        # This should not happen if hooks worked correctly
                                        logger.error(f"No captured inputs for layer {name} - this will cause incorrect pruning!")
                                        continue  # Skip this layer rather than using wrong inputs
                                
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
            if hasattr(module, '_original_weight'):
                delattr(module, '_original_weight')
        
        # Restore original model
        self.model.load_state_dict(original_state)
        
        return results
    
    def _pruning_experiments_multi(self) -> Dict[str, Any]:
        """Perform pruning experiments on multiple networks."""
        all_results = []
        
        # Run pruning experiments for each network
        for i, (net, wrapped_net) in enumerate(zip(self.networks, self.wrapped_networks)):
            logger.info(f"Pruning experiments for network {i+1}/{self.config.num_networks}")
            
            # Temporarily set as current model
            self.model = net
            self.wrapped_model = wrapped_net
            
            # Run experiments
            result = self._pruning_experiments_single()
            all_results.append(result)
        
        # Reset
        self.model = None
        self.wrapped_model = None
        
        # Aggregate results
        if self.config.aggregate_metrics:
            return self._aggregate_pruning_results(all_results)
        else:
            return {"networks": all_results}
    
    def _aggregate_pruning_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate pruning results from multiple networks."""
        import numpy as np
        
        # Get structure from first result
        aggregated = {
            "strategies": {},
            "final_model_performance": {}
        }
        
        # Aggregate each strategy
        first_result = results[0]
        for strategy_key in first_result["strategies"]:
            strategy_data = first_result["strategies"][strategy_key]
            
            # Initialize aggregated strategy data
            agg_strategy = {
                "pruning_amounts": strategy_data["pruning_amounts"],
                "accuracies_before_finetune": [],
                "losses_before_finetune": [],
                "accuracies_after_finetune": [],
                "losses_after_finetune": [],
                "sparsities": []
            }
            
            # Average metrics across networks
            num_amounts = len(strategy_data["pruning_amounts"])
            for i in range(num_amounts):
                # Collect values from all networks
                acc_before = [r["strategies"][strategy_key]["accuracies_before_finetune"][i] for r in results]
                loss_before = [r["strategies"][strategy_key]["losses_before_finetune"][i] for r in results]
                acc_after = [r["strategies"][strategy_key]["accuracies_after_finetune"][i] for r in results]
                loss_after = [r["strategies"][strategy_key]["losses_after_finetune"][i] for r in results]
                sparsity = [r["strategies"][strategy_key]["sparsities"][i] for r in results]
                
                # Average
                agg_strategy["accuracies_before_finetune"].append(np.mean(acc_before))
                agg_strategy["losses_before_finetune"].append(np.mean(loss_before))
                agg_strategy["accuracies_after_finetune"].append(np.mean(acc_after))
                agg_strategy["losses_after_finetune"].append(np.mean(loss_after))
                agg_strategy["sparsities"].append(np.mean(sparsity))
            
            aggregated["strategies"][strategy_key] = agg_strategy
        
        return aggregated
    
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
            
            # Group results by algorithm (for multi-selection mode comparison)
            algorithm_results = {}
            
            for strategy_key, strategy_results in self.pruning_results["strategies"].items():
                if not strategy_results.get("pruning_amounts"):
                    continue
                
                # Extract algorithm name and selection mode from key
                if "_" in strategy_key and strategy_key.split("_")[-1] in ["low", "high", "random"]:
                    # Format: "algorithm_selectionmode"
                    parts = strategy_key.rsplit("_", 1)
                    algorithm = parts[0]
                    selection_mode = parts[1]
                else:
                    # Single selection mode
                    algorithm = strategy_key
                    selection_mode = self.config.pruning_selection_mode
                    if isinstance(selection_mode, list):
                        selection_mode = selection_mode[0]
                
                # Initialize algorithm group if needed
                if algorithm not in algorithm_results:
                    algorithm_results[algorithm] = {
                        "sparsities": strategy_results["sparsities"],
                        "before": {},
                        "after": {}
                    }
                
                # Store accuracies by selection mode
                algorithm_results[algorithm]["before"][selection_mode] = strategy_results["accuracies_before_finetune"]
                algorithm_results[algorithm]["after"][selection_mode] = strategy_results["accuracies_after_finetune"]
            
            # Create visualizer
            visualizer = PruningVisualizer()
            
            # Generate plots for each algorithm
            for algorithm, results in algorithm_results.items():
                # Only create comparison plots if we have multiple selection modes
                if len(results["before"]) > 1:
                    # Create before/after comparison plots
                    fig_before, fig_after = visualizer.plot_accuracy_vs_sparsity_comparison(
                        sparsities=results["sparsities"],
                        accuracies_before=results["before"],
                        accuracies_after=results["after"],
                        title=f"{algorithm.capitalize()} Pruning",
                        save_path_prefix=str(output_dir / f"pruning_{algorithm}_accuracy")
                    )
                    plt.close(fig_before)
                    plt.close(fig_after)
                else:
                    # Single selection mode - create simple plot
                    selection_mode = list(results["before"].keys())[0]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot before and after on same plot
                    ax.plot([s * 100 for s in results["sparsities"]], 
                           results["before"][selection_mode],
                           'o-', label='Before Fine-tuning', color='#FF6B6B', 
                           linewidth=2.5, markersize=8)
                    ax.plot([s * 100 for s in results["sparsities"]], 
                           results["after"][selection_mode],
                           'o-', label='After Fine-tuning', color='#4ECDC4',
                           linewidth=2.5, markersize=8)
                    
                    ax.set_xlabel('Pruning %', fontsize=12)
                    ax.set_ylabel('Accuracy (%)', fontsize=12)
                    ax.set_title(f'{algorithm.capitalize()} Pruning ({selection_mode} mode)', 
                                fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 105)
                    
                    fig.tight_layout()
                    fig.savefig(output_dir / f"pruning_{algorithm}_accuracy.png",
                               dpi=self.config.plot_dpi, bbox_inches='tight')
                    plt.close(fig)
                
                # Also create improvement plot for each selection mode
                for selection_mode in results["before"]:
                    if selection_mode in results["after"]:
                        improvements = [
                            after - before 
                            for before, after in zip(
                                results["before"][selection_mode],
                                results["after"][selection_mode]
                            )
                        ]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(range(len(improvements)), improvements, 
                                      tick_label=[f"{s:.0%}" for s in results["sparsities"]],
                                      color=['#4ECDC4' if imp >= 0 else '#FF6B6B' for imp in improvements],
                                      alpha=0.8)
                        
                        # Add value labels on bars
                        for bar, imp in zip(bars, improvements):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{imp:+.1f}%', ha='center', 
                                   va='bottom' if height >= 0 else 'top',
                                   fontsize=10, fontweight='bold')
                        
                        ax.set_xlabel("Sparsity Level", fontsize=12)
                        ax.set_ylabel("Accuracy Improvement (%)", fontsize=12)
                        ax.set_title(f"{algorithm.capitalize()} Pruning ({selection_mode} mode): Fine-tuning Improvement",
                                    fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3, axis='y')
                        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                        
                        fig.tight_layout()
                        suffix = f"_{selection_mode}" if len(results["before"]) > 1 else ""
                        fig.savefig(output_dir / f"pruning_{algorithm}_improvement{suffix}.png",
                                   dpi=self.config.plot_dpi, bbox_inches='tight')
                        plt.close(fig)
        
        logger.info(f"Saved visualizations to {output_dir}") 