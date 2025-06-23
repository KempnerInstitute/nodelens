"""
General alignment experiment for comprehensive analysis.

This experiment provides a flexible framework for:
1. Training models on specified datasets
2. Computing alignment metrics on all or specified layers
3. Applying pruning strategies based on metric results
4. Tracking performance and alignment throughout the process
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass, field

from alignment.experiments.base import BaseExperiment, ExperimentConfig
from alignment.core.registry import register_experiment
from alignment.data import get_dataset
from alignment.training import BaseTrainer, TrainingConfig
from alignment.pruning import get_pruning_strategy
from alignment.metrics import get_metric

logger = logging.getLogger(__name__)


@dataclass
class GeneralAlignmentConfig(ExperimentConfig):
    """Configuration for general alignment experiments."""
    
    # Dataset configuration
    dataset_name: str = "mnist"
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=lambda: {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam",
        "scheduler": "cosine"
    })
    
    # Metrics configuration
    alignment_metrics: List[str] = field(default_factory=lambda: [
        "rayleigh_quotient",
        "mutual_information_gaussian",
        "weight_cosine_similarity"
    ])
    compute_metrics_on: Optional[List[str]] = None  # None means all layers
    
    # Pruning configuration
    pruning_strategy: str = "magnitude"
    pruning_config: Dict[str, Any] = field(default_factory=lambda: {
        "amount": 0.5,
        "structured": False
    })
    pruning_based_on_metric: Optional[str] = None  # Use metric to guide pruning
    
    # Experiment flow
    train_model: bool = True
    compute_initial_metrics: bool = True
    apply_pruning: bool = True
    fine_tune_after_pruning: bool = True
    fine_tune_epochs: int = 5
    
    # Analysis configuration
    track_performance: bool = True
    save_checkpoints: bool = True
    save_metrics_history: bool = True


@register_experiment("general_alignment")
class GeneralAlignmentExperiment(BaseExperiment):
    """
    General experiment for alignment analysis with pruning.
    
    This experiment provides a complete pipeline for:
    1. Model training on specified dataset
    2. Comprehensive metric computation
    3. Pruning with various strategies
    4. Fine-tuning and analysis
    """
    
    def __init__(self, config: GeneralAlignmentConfig):
        """Initialize the general alignment experiment."""
        super().__init__(config)
        self.config: GeneralAlignmentConfig = config
        
        # Results storage
        self.results = {
            "initial_metrics": {},
            "pruning_results": {},
            "final_metrics": {},
            "performance_history": {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }
        }
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment pipeline."""
        logger.info("Starting general alignment experiment")
        
        # Step 1: Setup dataset
        train_loader, val_loader, test_loader = self._setup_dataset()
        
        # Step 2: Train model (if requested)
        if self.config.train_model:
            logger.info("Training model...")
            self._train_model(train_loader, val_loader)
        
        # Step 3: Compute initial metrics
        if self.config.compute_initial_metrics:
            logger.info("Computing initial alignment metrics...")
            self.results["initial_metrics"] = self._compute_metrics(test_loader)
            
            # Log initial metrics
            self._log_metrics("initial", self.results["initial_metrics"])
        
        # Step 4: Apply pruning (if requested)
        if self.config.apply_pruning:
            logger.info(f"Applying {self.config.pruning_strategy} pruning...")
            pruning_masks = self._apply_pruning(test_loader)
            self.results["pruning_results"]["masks"] = pruning_masks
            self.results["pruning_results"]["sparsity"] = self._compute_sparsity()
        
        # Step 5: Fine-tune after pruning (if requested)
        if self.config.apply_pruning and self.config.fine_tune_after_pruning:
            logger.info("Fine-tuning pruned model...")
            self._fine_tune(train_loader, val_loader)
        
        # Step 6: Compute final metrics
        logger.info("Computing final metrics...")
        self.results["final_metrics"] = self._compute_metrics(test_loader)
        self._log_metrics("final", self.results["final_metrics"])
        
        # Step 7: Analyze results
        self.results["analysis"] = self._analyze_results()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def _setup_dataset(self):
        """Setup dataset loaders."""
        dataset_class = get_dataset(self.config.dataset_name)
        
        # Create train dataset
        train_dataset = dataset_class(
            train=True,
            **self.config.dataset_config
        )
        
        # Create test dataset
        test_dataset = dataset_class(
            train=False,
            **self.config.dataset_config
        )
        
        # Create loaders
        batch_size = self.config.training_config.get("batch_size", 32)
        
        train_loader = train_dataset.get_train_loader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Split validation from training if needed
        val_loader = train_dataset.get_val_loader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        test_loader = test_dataset.get_test_loader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        logger.info(f"Dataset {self.config.dataset_name} loaded successfully")
        return train_loader, val_loader, test_loader
    
    def _train_model(self, train_loader, val_loader):
        """Train the model using the training configuration."""
        # Create training config
        train_config = TrainingConfig(**self.config.training_config)
        train_config.checkpoint_dir = self.config.checkpoint_dir
        
        # Create trainer
        trainer = BaseTrainer(
            model=self.model,
            config=train_config,
            callbacks=[self._training_callback]
        )
        
        # Define metric function for training
        def compute_accuracy(outputs, targets):
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).float().mean()
            return {"accuracy": correct.item()}
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            metric_fn=compute_accuracy
        )
        
        # Store training history
        self.results["performance_history"]["train_loss"] = history["train_loss"]
        self.results["performance_history"]["train_acc"] = [
            m.get("accuracy", 0) for m in history["train_metrics"]
        ]
        self.results["performance_history"]["val_loss"] = history["val_loss"]
        self.results["performance_history"]["val_acc"] = [
            m.get("accuracy", 0) for m in history["val_metrics"]
        ]
    
    def _compute_metrics(self, data_loader) -> Dict[str, Dict[str, float]]:
        """Compute alignment metrics on specified layers."""
        metrics_results = {}
        
        # Determine which layers to compute metrics on
        if self.config.compute_metrics_on:
            layers = self.config.compute_metrics_on
        else:
            layers = self.wrapped_model.tracked_layers
        
        # Get batch of data for metric computation
        data_batch = next(iter(data_loader))[0].to(self.config.device)
        
        # Compute each metric
        for metric_name in self.config.alignment_metrics:
            metric = get_metric(metric_name)()
            metrics_results[metric_name] = {}
            
            # Compute metric for each layer
            for layer_name in layers:
                try:
                    # Get layer weights
                    weights = self.wrapped_model.get_layer_weights([layer_name])
                    if layer_name in weights:
                        # Compute metric
                        score = metric.compute(
                            inputs=data_batch,
                            weights=weights[layer_name]
                        )
                        
                        # Store result
                        if isinstance(score, torch.Tensor):
                            metrics_results[metric_name][layer_name] = score.mean().item()
                        else:
                            metrics_results[metric_name][layer_name] = float(score)
                
                except Exception as e:
                    logger.warning(f"Failed to compute {metric_name} for {layer_name}: {e}")
        
        return metrics_results
    
    def _apply_pruning(self, data_loader) -> Dict[str, torch.Tensor]:
        """Apply pruning strategy to the model."""
        # Get pruning strategy
        strategy = get_pruning_strategy(
            self.config.pruning_strategy,
            **self.config.pruning_config
        )
        
        # If using metric-based pruning, compute importance scores
        if self.config.pruning_based_on_metric:
            importance_scores = self._compute_metric_based_importance()
        else:
            importance_scores = None
        
        # Apply pruning to each layer
        pruning_masks = {}
        
        for layer_name in self.wrapped_model.tracked_layers:
            module = dict(self.model.named_modules())[layer_name]
            
            if hasattr(module, 'weight'):
                # Get data for gradient-based pruning if needed
                if self.config.pruning_strategy in ['gradient', 'fisher', 'taylor']:
                    # Compute gradients
                    data_batch, targets = next(iter(data_loader))
                    data_batch = data_batch.to(self.config.device)
                    targets = targets.to(self.config.device)
                    
                    outputs = self.model(data_batch)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                else:
                    data_batch = None
                
                # Apply pruning
                mask = strategy.compute_mask(
                    module,
                    self.config.pruning_config.get("amount", 0.5),
                    importance_scores=importance_scores.get(layer_name) if importance_scores else None
                )
                
                # Apply mask
                strategy.apply_mask(module, mask)
                pruning_masks[layer_name] = mask
        
        return pruning_masks
    
    def _compute_metric_based_importance(self) -> Dict[str, torch.Tensor]:
        """Compute importance scores based on alignment metrics."""
        importance_scores = {}
        metric_name = self.config.pruning_based_on_metric
        
        if metric_name in self.results["initial_metrics"]:
            metric_results = self.results["initial_metrics"][metric_name]
            
            for layer_name, score in metric_results.items():
                # Convert metric score to importance
                # Higher alignment = higher importance
                importance_scores[layer_name] = torch.tensor(score)
        
        return importance_scores
    
    def _fine_tune(self, train_loader, val_loader):
        """Fine-tune the model after pruning."""
        # Update training config for fine-tuning
        fine_tune_config = TrainingConfig(**self.config.training_config)
        fine_tune_config.epochs = self.config.fine_tune_epochs
        fine_tune_config.learning_rate *= 0.1  # Lower learning rate
        
        # Create trainer
        trainer = BaseTrainer(
            model=self.model,
            config=fine_tune_config
        )
        
        # Fine-tune
        trainer.train(train_loader=train_loader, val_loader=val_loader)
    
    def _compute_sparsity(self) -> Dict[str, float]:
        """Compute sparsity statistics for each layer."""
        sparsity = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight
                num_zeros = (weight == 0).sum().item()
                total_params = weight.numel()
                sparsity[name] = num_zeros / total_params
        
        # Compute overall sparsity
        total_zeros = sum((p == 0).sum().item() for p in self.model.parameters())
        total_params = sum(p.numel() for p in self.model.parameters())
        sparsity["overall"] = total_zeros / total_params
        
        return sparsity
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze the experiment results."""
        analysis = {}
        
        # Compare initial vs final metrics
        if self.config.compute_initial_metrics:
            analysis["metric_changes"] = {}
            
            for metric_name in self.config.alignment_metrics:
                if metric_name in self.results["initial_metrics"] and metric_name in self.results["final_metrics"]:
                    analysis["metric_changes"][metric_name] = {}
                    
                    initial = self.results["initial_metrics"][metric_name]
                    final = self.results["final_metrics"][metric_name]
                    
                    for layer_name in initial:
                        if layer_name in final:
                            change = final[layer_name] - initial[layer_name]
                            percent_change = (change / (initial[layer_name] + 1e-8)) * 100
                            
                            analysis["metric_changes"][metric_name][layer_name] = {
                                "absolute_change": change,
                                "percent_change": percent_change
                            }
        
        # Analyze sparsity impact
        if self.config.apply_pruning:
            analysis["sparsity_impact"] = {
                "achieved_sparsity": self.results["pruning_results"]["sparsity"],
                "performance_retention": self._compute_performance_retention()
            }
        
        return analysis
    
    def _compute_performance_retention(self) -> float:
        """Compute how much performance was retained after pruning."""
        if not self.results["performance_history"]["val_acc"]:
            return 0.0
        
        initial_acc = max(self.results["performance_history"]["val_acc"][:self.config.training_config["epochs"]])
        final_acc = max(self.results["performance_history"]["val_acc"][-self.config.fine_tune_epochs:]) if self.config.fine_tune_after_pruning else 0
        
        return (final_acc / (initial_acc + 1e-8)) * 100
    
    def _training_callback(self, trainer, epoch):
        """Callback for tracking training progress."""
        if epoch % 5 == 0:
            logger.info(f"Training epoch {epoch}: LR={trainer.optimizer.param_groups[0]['lr']:.6f}")
    
    def _log_metrics(self, phase: str, metrics: Dict[str, Dict[str, float]]):
        """Log metrics for a specific phase."""
        logger.info(f"\n{phase.upper()} METRICS:")
        for metric_name, layer_results in metrics.items():
            logger.info(f"  {metric_name}:")
            for layer_name, value in layer_results.items():
                logger.info(f"    {layer_name}: {value:.4f}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GeneralAlignmentExperiment":
        """Create experiment from YAML configuration file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = GeneralAlignmentConfig(**config_dict)
        return cls(config) 