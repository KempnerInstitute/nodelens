"""
General task-specific alignment metrics that can be customized for different downstream tasks.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, Union
from ...core.registry import register_metric
from ...core.base import BaseMetric


@register_metric("task_alignment")
class TaskAlignment(BaseMetric):
    """
    Compute alignment with respect to a specific task's target function.
    
    This metric measures how well neuron activations align with gradients
    or importance scores from a downstream task.
    """
    
    name = "task_alignment"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(
        self,
        task_loss_fn: Optional[Callable] = None,
        alignment_type: str = "gradient",
        normalize: bool = True
    ):
        """
        Initialize task alignment metric.
        
        Args:
            task_loss_fn: Loss function for the task (if None, uses MSE)
            alignment_type: Type of alignment ('gradient', 'taylor', 'integrated_gradients')
            normalize: Whether to normalize alignment scores
        """
        super().__init__()
        self.task_loss_fn = task_loss_fn or F.mse_loss
        self.alignment_type = alignment_type
        self.normalize = normalize
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute task-specific alignment scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            targets: Target values for the task
            
        Returns:
            Task alignment scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        # If no targets provided, create dummy targets
        if targets is None:
            # Use outputs themselves as targets (self-supervised)
            targets = outputs.detach()
        
        # Ensure inputs require gradients
        inputs_grad = inputs.detach().requires_grad_(True)
        outputs_grad = inputs_grad @ weights.T
        
        if self.alignment_type == "gradient":
            # Compute task loss
            loss = self.task_loss_fn(outputs_grad, targets)
            
            # Compute gradients with respect to inputs
            grads = torch.autograd.grad(loss, inputs_grad, retain_graph=True)[0]
            
            # Compute alignment as dot product between weights and input gradients
            alignment_scores = torch.abs((weights @ grads.T).mean(dim=1))
            
        elif self.alignment_type == "taylor":
            # First-order Taylor expansion importance
            loss = self.task_loss_fn(outputs_grad, targets)
            
            # Compute gradients with respect to outputs
            output_grads = torch.autograd.grad(loss, outputs_grad, retain_graph=True)[0]
            
            # Taylor importance: |grad * activation|
            importance = torch.abs(output_grads * outputs_grad)
            alignment_scores = importance.mean(dim=0)
            
        elif self.alignment_type == "integrated_gradients":
            # Integrated gradients from baseline to current inputs
            baseline = torch.zeros_like(inputs)
            n_steps = 10
            
            integrated_grads = torch.zeros_like(weights)
            
            for i in range(n_steps):
                alpha = i / n_steps
                interpolated = baseline + alpha * (inputs - baseline)
                interpolated.requires_grad_(True)
                
                outputs_interp = interpolated @ weights.T
                loss = self.task_loss_fn(outputs_interp, targets)
                
                grads = torch.autograd.grad(loss, interpolated)[0]
                integrated_grads += weights @ grads.T
            
            integrated_grads = integrated_grads / n_steps
            alignment_scores = integrated_grads.abs().mean(dim=1)
        
        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")
        
        if self.normalize:
            # Normalize scores to [0, 1]
            min_score = alignment_scores.min()
            max_score = alignment_scores.max()
            if max_score > min_score:
                alignment_scores = (alignment_scores - min_score) / (max_score - min_score)
        
        return alignment_scores


@register_metric("class_selectivity")
class ClassSelectivity(BaseMetric):
    """
    Measure how selectively neurons respond to different classes in classification tasks.
    """
    
    name = "class_selectivity"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(
        self,
        n_classes: Optional[int] = None,
        selectivity_type: str = "variance",
        temperature: float = 1.0
    ):
        """
        Initialize class selectivity metric.
        
        Args:
            n_classes: Number of classes (inferred if None)
            selectivity_type: Type of selectivity measure ('variance', 'entropy', 'gini')
            temperature: Temperature for softmax (if using entropy)
        """
        super().__init__()
        self.n_classes = n_classes
        self.selectivity_type = selectivity_type
        self.temperature = temperature
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute class selectivity scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            labels: Class labels for each sample
            
        Returns:
            Selectivity scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        n_neurons = outputs.shape[1]
        
        # If no labels provided, cluster outputs
        if labels is None:
            # Simple k-means style clustering
            n_classes = self.n_classes or min(10, outputs.shape[0] // 10)
            
            # Use random initialization
            centroids = outputs[torch.randperm(outputs.shape[0])[:n_classes]]
            
            # Assign to nearest centroid
            distances = torch.cdist(outputs, centroids)
            labels = distances.argmin(dim=1)
        
        # Infer number of classes
        unique_labels = labels.unique()
        n_classes = len(unique_labels)
        
        selectivity_scores = torch.zeros(n_neurons, device=outputs.device)
        
        for neuron_idx in range(n_neurons):
            neuron_outputs = outputs[:, neuron_idx]
            
            # Compute mean activation for each class
            class_means = []
            class_vars = []
            
            for class_idx in unique_labels:
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_activations = neuron_outputs[class_mask]
                    class_means.append(class_activations.mean())
                    # Only compute variance if we have more than one sample
                    if class_mask.sum() > 1:
                        class_vars.append(class_activations.var())
                    else:
                        class_vars.append(torch.tensor(0.0, device=outputs.device))
            
            class_means = torch.stack(class_means)
            class_vars = torch.stack(class_vars)
            
            if self.selectivity_type == "variance":
                # Ratio of between-class variance to within-class variance
                between_var = class_means.var()
                within_var = class_vars.mean()
                selectivity = between_var / (within_var + 1e-8)
                
            elif self.selectivity_type == "entropy":
                # Entropy of class response distribution
                # Normalize responses to probabilities
                probs = F.softmax(class_means / self.temperature, dim=0)
                entropy = -(probs * probs.log()).sum()
                # Lower entropy = higher selectivity
                selectivity = 1.0 / (1.0 + entropy)
                
            elif self.selectivity_type == "gini":
                # Gini coefficient of class responses
                sorted_means = class_means.sort()[0]
                n = len(sorted_means)
                index = torch.arange(1, n + 1, device=outputs.device)
                gini = (2 * (index * sorted_means).sum()) / (n * sorted_means.sum()) - (n + 1) / n
                selectivity = gini
                
            else:
                raise ValueError(f"Unknown selectivity type: {self.selectivity_type}")
            
            selectivity_scores[neuron_idx] = selectivity
        
        return selectivity_scores


@register_metric("feature_importance")
class FeatureImportance(BaseMetric):
    """
    Compute feature importance scores based on task-specific objectives.
    """
    
    name = "feature_importance"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(
        self,
        importance_method: str = "permutation",
        n_permutations: int = 10,
        task_metric: Optional[Callable] = None
    ):
        """
        Initialize feature importance metric.
        
        Args:
            importance_method: Method to compute importance ('permutation', 'shap_approximation')
            n_permutations: Number of permutations for permutation importance
            task_metric: Task-specific metric to optimize (higher is better)
        """
        super().__init__()
        self.importance_method = importance_method
        self.n_permutations = n_permutations
        self.task_metric = task_metric or self._default_metric
    
    def _default_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Default metric: negative MSE (so higher is better)."""
        return -F.mse_loss(outputs, targets).item()
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute feature importance scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations
            targets: Target values
            
        Returns:
            Importance scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        if targets is None:
            targets = outputs.detach()
        
        n_neurons = weights.shape[0]
        importance_scores = torch.zeros(n_neurons, device=outputs.device)
        
        # Baseline performance
        baseline_score = self.task_metric(outputs, targets)
        
        if self.importance_method == "permutation":
            # Permutation importance
            for neuron_idx in range(n_neurons):
                neuron_scores = []
                
                for _ in range(self.n_permutations):
                    # Create a copy of weights with permuted neuron
                    weights_perm = weights.clone()
                    perm_idx = torch.randperm(weights.shape[1])
                    weights_perm[neuron_idx] = weights[neuron_idx, perm_idx]
                    
                    # Compute outputs with permuted weights
                    outputs_perm = inputs @ weights_perm.T
                    
                    # Compute performance drop
                    perm_score = self.task_metric(outputs_perm, targets)
                    neuron_scores.append(baseline_score - perm_score)
                
                importance_scores[neuron_idx] = torch.tensor(neuron_scores).mean()
        
        elif self.importance_method == "shap_approximation":
            # Simplified SHAP-like approximation
            # Use gradient * activation as approximation
            inputs_grad = inputs.detach().requires_grad_(True)
            outputs_grad = inputs_grad @ weights.T
            
            # Use negative of task metric as loss
            loss = -self.task_metric(outputs_grad, targets)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, outputs_grad)[0]
            
            # Importance = |gradient * activation|
            importance_scores = (grads * outputs_grad).abs().mean(dim=0)
        
        else:
            raise ValueError(f"Unknown importance method: {self.importance_method}")
        
        return importance_scores


@register_metric("representation_quality")
class RepresentationQuality(BaseMetric):
    """
    Measure the quality of learned representations for downstream tasks.
    """
    
    name = "representation_quality"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False
    
    def __init__(
        self,
        quality_measure: str = "linear_probe",
        probe_type: str = "ridge",
        regularization: float = 1.0
    ):
        """
        Initialize representation quality metric.
        
        Args:
            quality_measure: Type of quality measure ('linear_probe', 'nearest_neighbor')
            probe_type: Type of linear probe ('ridge', 'logistic')
            regularization: Regularization strength for linear probe
        """
        super().__init__()
        self.quality_measure = quality_measure
        self.probe_type = probe_type
        self.regularization = regularization
    
    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute representation quality scores.
        
        Args:
            inputs: Input activations
            weights: Weight matrix
            outputs: Output activations (representations)
            targets: Target values/labels
            
        Returns:
            Quality scores for each neuron
        """
        if outputs is None:
            outputs = inputs @ weights.T
        
        if targets is None:
            # Create synthetic targets based on clustering
            kmeans_labels = self._simple_kmeans(outputs, k=5)
            targets = kmeans_labels.float()
        
        n_neurons = outputs.shape[1]
        quality_scores = torch.zeros(n_neurons, device=outputs.device)
        
        if self.quality_measure == "linear_probe":
            # Fit a linear probe on top of each neuron's representation
            for neuron_idx in range(n_neurons):
                neuron_repr = outputs[:, neuron_idx].unsqueeze(1)
                
                if self.probe_type == "ridge":
                    # Ridge regression
                    # Solve: (X^T X + λI)w = X^T y
                    XtX = neuron_repr.T @ neuron_repr
                    Xty = neuron_repr.T @ targets
                    
                    # Add regularization
                    XtX = XtX + self.regularization * torch.eye(1, device=outputs.device)
                    
                    # Solve for weights
                    probe_weights = torch.linalg.solve(XtX, Xty)
                    
                    # Compute predictions
                    predictions = neuron_repr @ probe_weights
                    
                    # Compute R^2 score
                    ss_res = ((targets - predictions) ** 2).sum()
                    ss_tot = ((targets - targets.mean()) ** 2).sum()
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                    
                    quality_scores[neuron_idx] = r2.clamp(min=0)
                
        elif self.quality_measure == "nearest_neighbor":
            # k-NN classification quality
            k = min(5, outputs.shape[0] // 10)
            
            for neuron_idx in range(n_neurons):
                neuron_repr = outputs[:, neuron_idx].unsqueeze(1)
                
                # Compute pairwise distances
                distances = torch.cdist(neuron_repr, neuron_repr)
                
                # Get k nearest neighbors (excluding self)
                _, indices = distances.topk(k + 1, largest=False, dim=1)
                neighbor_indices = indices[:, 1:]  # Exclude self
                
                # Get neighbor targets
                neighbor_targets = targets[neighbor_indices]
                
                # Predict as mean of neighbors (for regression) or mode (for classification)
                predictions = neighbor_targets.mean(dim=1)
                
                # Compute accuracy or MSE
                if targets.dtype == torch.long:
                    # Classification: accuracy
                    accuracy = (predictions.round() == targets).float().mean()
                    quality_scores[neuron_idx] = accuracy
                else:
                    # Regression: negative MSE
                    mse = F.mse_loss(predictions, targets)
                    quality_scores[neuron_idx] = 1.0 / (1.0 + mse)
        
        return quality_scores
    
    def _simple_kmeans(self, data: torch.Tensor, k: int, n_iter: int = 10) -> torch.Tensor:
        """Simple k-means clustering implementation."""
        n_samples = data.shape[0]
        
        # Random initialization
        centroids = data[torch.randperm(n_samples)[:k]]
        
        for _ in range(n_iter):
            # Assign to nearest centroid
            distances = torch.cdist(data, centroids)
            labels = distances.argmin(dim=1)
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = data[mask].mean(dim=0)
        
        return labels 