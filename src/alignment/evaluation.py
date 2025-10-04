"""
Model evaluation utilities.

Provides standard evaluation functions for different model types and tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


def evaluate_classification(
    model: nn.Module,
    data_loader,
    device: str = 'cuda',
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate classification model.
    
    Args:
        model: Model to evaluate
        data_loader: Validation/test dataloader
        device: Device to use
        criterion: Loss function (default: CrossEntropyLoss)
        
    Returns:
        Dict with 'loss' and 'accuracy'
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate_perplexity(
    model: nn.Module,
    data_loader,
    device: str = 'cuda',
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate language model perplexity.
    
    Args:
        model: Language model
        data_loader: Text dataloader with 'input_ids' and 'labels'
        device: Device to use
        max_batches: Maximum batches to evaluate (None = all)
        
    Returns:
        Dict with 'perplexity' and 'loss'
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids).to(device)
            else:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            
            # Get loss
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # Compute manually
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            
            # Accumulate
            batch_tokens = (labels != -100).sum().item()  # Count non-padding tokens
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'perplexity': perplexity,
        'loss': avg_loss
    }


def evaluate_model(
    model: nn.Module,
    data_loader,
    task: str = 'classification',
    device: str = 'cuda',
    **kwargs
) -> Dict[str, float]:
    """
    General evaluation function that dispatches to task-specific evaluators.
    
    Args:
        model: Model to evaluate
        data_loader: Dataloader
        task: Task type ('classification', 'language_modeling', 'regression')
        device: Device
        **kwargs: Additional arguments for specific evaluators
        
    Returns:
        Evaluation metrics dictionary
    """
    if task == 'classification':
        return evaluate_classification(model, data_loader, device, **kwargs)
    
    elif task in ['language_modeling', 'lm', 'causal_lm']:
        return evaluate_perplexity(model, data_loader, device, **kwargs)
    
    elif task == 'regression':
        return evaluate_regression(model, data_loader, device, **kwargs)
    
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_regression(
    model: nn.Module,
    data_loader,
    device: str = 'cuda',
    criterion: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate regression model.
    
    Args:
        model: Regression model
        data_loader: Dataloader
        device: Device
        criterion: Loss function (default: MSELoss)
        
    Returns:
        Dict with 'mse' and 'mae'
    """
    if criterion is None:
        criterion = nn.MSELoss()
    
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            mse = nn.MSELoss()(outputs, targets).item()
            mae = nn.L1Loss()(outputs, targets).item()
            
            batch_size = inputs.size(0)
            total_mse += mse * batch_size
            total_mae += mae * batch_size
            total += batch_size
    
    return {
        'mse': total_mse / total,
        'mae': total_mae / total
    }


class EvaluationManager:
    """
    Manager for handling multiple evaluation metrics.
    
    Useful for tracking multiple metrics during experiments.
    """
    
    def __init__(self, task: str = 'classification'):
        """
        Initialize evaluation manager.
        
        Args:
            task: Primary task type
        """
        self.task = task
        self.history = []
    
    def evaluate(
        self,
        model: nn.Module,
        data_loader,
        device: str = 'cuda',
        step: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate and record results.
        
        Args:
            model: Model to evaluate
            data_loader: Dataloader
            device: Device
            step: Training step number (for tracking)
            **kwargs: Additional arguments
            
        Returns:
            Evaluation metrics
        """
        results = evaluate_model(model, data_loader, self.task, device, **kwargs)
        
        if step is not None:
            results['step'] = step
        
        self.history.append(results)
        
        return results
    
    def get_history(self) -> list:
        """Get evaluation history."""
        return self.history
    
    def get_best(self, metric: str = 'accuracy') -> Dict[str, float]:
        """
        Get best result based on metric.
        
        Args:
            metric: Metric to optimize ('accuracy', 'perplexity', 'loss')
            
        Returns:
            Best evaluation result
        """
        if not self.history:
            return {}
        
        if metric in ['perplexity', 'loss', 'mse', 'mae']:
            # Lower is better
            return min(self.history, key=lambda x: x.get(metric, float('inf')))
        else:
            # Higher is better (accuracy, etc.)
            return max(self.history, key=lambda x: x.get(metric, float('-inf')))

