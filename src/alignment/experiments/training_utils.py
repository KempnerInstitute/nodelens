"""
Utility functions for integrating ExperimentTrainer into experiments.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from alignment.training import ExperimentTrainer, ExperimentTrainingConfig


def create_experiment_trainer(model: Union[nn.Module, List[nn.Module]], config: Dict[str, Any], device: str = "cuda") -> ExperimentTrainer:
    """
    Create an ExperimentTrainer from experiment config.

    Args:
        model: Model or list of models to train
        config: Experiment configuration dictionary
        device: Device to train on

    Returns:
        Configured ExperimentTrainer instance
    """
    # Extract training-related config
    training_config = ExperimentTrainingConfig(
        epochs=config.get("training_epochs", config.get("epochs", 10)),
        learning_rate=config.get("learning_rate", 0.001),
        batch_size=config.get("batch_size", 32),
        optimizer=config.get("optimizer", "adam"),
        optimizer_kwargs=config.get("optimizer_kwargs", {}),
        scheduler=config.get("scheduler", None),
        scheduler_kwargs=config.get("scheduler_kwargs", {}),
        device=device,
        log_interval=config.get("log_interval", 100),
        eval_interval=config.get("eval_interval", 1),
        checkpoint_dir=config.get("checkpoint_dir", None),
        early_stopping_patience=config.get("early_stopping_patience", None),
        gradient_clip_val=config.get("gradient_clip_val", None),
        # Multi-network specific
        num_networks=config.get("num_networks", 1),
        tensorized=config.get("tensorized_training", True),
        save_all_networks=config.get("save_all_networks", False),
        metric_aggregation=config.get("metric_aggregation", "mean"),
    )

    # Create trainer
    return ExperimentTrainer(model=model, config=training_config, loss_fn=nn.CrossEntropyLoss(), callbacks=[])


def train_with_metrics(
    trainer: ExperimentTrainer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    compute_accuracy: bool = True,
) -> Dict[str, Any]:
    """
    Train using ExperimentTrainer with standard metrics.

    Args:
        trainer: ExperimentTrainer instance
        train_loader: Training data loader
        val_loader: Optional validation data loader
        compute_accuracy: Whether to compute accuracy metric

    Returns:
        Training history with metrics
    """

    # Define metric function
    def metric_fn(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        if compute_accuracy:
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
            metrics["accuracy"] = 100.0 * correct / total
        return metrics

    # Train
    history = trainer.train(train_loader=train_loader, val_loader=val_loader, metric_fn=metric_fn if compute_accuracy else None)

    return history


def convert_training_history(history: Dict[str, Any], num_networks: int = 1) -> Dict[str, Any]:
    """
    Convert ExperimentTrainer history to experiment result format.

    Args:
        history: Training history from ExperimentTrainer
        num_networks: Number of networks trained

    Returns:
        Converted results dictionary
    """
    results = {
        "training_epochs": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_train_accuracy": history["train_metrics"][-1].get("accuracy", 0.0)
        if history["train_metrics"] and history["train_metrics"][-1]
        else 0.0,
        "training_history": history,
    }

    if history["val_loss"]:
        results["final_val_loss"] = history["val_loss"][-1]
        results["final_val_accuracy"] = (
            history["val_metrics"][-1].get("accuracy", 0.0) if history["val_metrics"] and history["val_metrics"][-1] else 0.0
        )

    # Add per-network results if multi-network
    if num_networks > 1 and "per_network" in history:
        results["per_network_results"] = {}
        for i in range(num_networks):
            network_history = history["per_network"][i]
            results["per_network_results"][i] = {
                "final_train_loss": network_history["train_loss"][-1] if network_history["train_loss"] else 0.0,
                "final_train_accuracy": network_history["train_metrics"][-1].get("accuracy", 0.0)
                if network_history["train_metrics"] and network_history["train_metrics"][-1]
                else 0.0,
            }
            if network_history["val_loss"]:
                results["per_network_results"][i]["final_val_loss"] = network_history["val_loss"][-1]
                results["per_network_results"][i]["final_val_accuracy"] = (
                    network_history["val_metrics"][-1].get("accuracy", 0.0)
                    if network_history["val_metrics"] and network_history["val_metrics"][-1]
                    else 0.0
                )

    return results


def evaluate_with_metrics(
    trainer: ExperimentTrainer, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str = "cuda", compute_alignment: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model and compute metrics.

    Args:
        trainer: The trainer instance
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        compute_alignment: Whether to compute alignment metrics

    Returns:
        Dictionary of evaluation metrics
    """
    # Basic evaluation
    metrics = trainer.evaluate(model, data_loader, device=device)

    # Add alignment metrics if requested
    if compute_alignment and hasattr(trainer, "compute_alignment_metrics"):
        alignment_metrics = trainer.compute_alignment_metrics(model, data_loader, device=device)
        metrics.update(alignment_metrics)

    return metrics
