"""
Classification-specific alignment metrics.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ...core.base import BaseMetric
from ...core.registry import register_metric


@register_metric("classification_alignment")
class ClassificationAlignment(BaseMetric):
    """
    Measures alignment specifically for classification tasks.

    This metric evaluates how well neuron activations align with class boundaries
    and decision surfaces in classification problems.
    """

    name = "classification_alignment"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False

    def __init__(
        self,
        n_classes: int,
        alignment_type: str = "boundary_distance",
        temperature: float = 1.0
    ):
        """
        Initialize classification alignment metric.

        Args:
            n_classes: Number of classes in the classification task
            alignment_type: Type of alignment measure
                - 'boundary_distance': Distance to decision boundaries
                - 'class_separation': Separation between classes
                - 'confidence_alignment': Alignment with prediction confidence
            temperature: Temperature for softmax operations
        """
        super().__init__()
        self.n_classes = n_classes
        self.alignment_type = alignment_type
        self.temperature = temperature

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute classification alignment scores.

        Args:
            inputs: Input activations [batch_size, input_dim]
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations [batch_size, output_dim]
            labels: True class labels [batch_size]
            logits: Classification logits [batch_size, n_classes]

        Returns:
            Alignment scores for each neuron [output_dim]
        """
        if outputs is None:
            outputs = inputs @ weights.T

        n_neurons = outputs.shape[1]
        alignment_scores = torch.zeros(n_neurons, device=outputs.device)

        # Generate synthetic labels if not provided
        if labels is None:
            # Use k-means clustering on outputs
            labels = self._generate_labels(outputs, self.n_classes)

        # Generate logits if not provided
        if logits is None:
            # Simple linear projection from outputs to classes
            projection = torch.randn(outputs.shape[1], self.n_classes, device=outputs.device)
            logits = outputs @ projection

        if self.alignment_type == "boundary_distance":
            # Measure how neuron activations change near decision boundaries
            probabilities = F.softmax(logits / self.temperature, dim=1)
            # Add small epsilon to avoid log(0)
            entropy = -(probabilities * (probabilities + 1e-8).log()).sum(dim=1)

            # High entropy indicates proximity to decision boundary
            # Compute correlation between neuron activation and boundary proximity
            for i in range(n_neurons):
                # Check if there's enough variance for correlation
                if outputs[:, i].std() > 1e-8 and entropy.std() > 1e-8:
                    correlation = torch.corrcoef(
                        torch.stack([outputs[:, i], entropy])
                    )[0, 1]
                    # Handle NaN values
                    if torch.isnan(correlation):
                        alignment_scores[i] = 0.0
                    else:
                        alignment_scores[i] = correlation.abs()
                else:
                    alignment_scores[i] = 0.0

        elif self.alignment_type == "class_separation":
            # Measure how well neurons separate different classes
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]

                # Compute mean activation per class
                class_means = []
                class_stds = []

                for c in range(self.n_classes):
                    class_mask = labels == c
                    if class_mask.sum() > 0:
                        class_acts = neuron_activations[class_mask]
                        class_means.append(class_acts.mean())
                        class_stds.append(class_acts.std())

                if len(class_means) > 1:
                    class_means = torch.stack(class_means)
                    class_stds = torch.stack(class_stds)

                    # Fisher's discriminant ratio
                    between_class_var = class_means.var()
                    within_class_var = class_stds.mean() ** 2

                    if within_class_var > 0:
                        separation = between_class_var / within_class_var
                    else:
                        separation = between_class_var

                    alignment_scores[i] = separation

        elif self.alignment_type == "confidence_alignment":
            # Measure alignment with prediction confidence
            probabilities = F.softmax(logits / self.temperature, dim=1)

            # Get confidence (max probability) and correctness
            confidence, predictions = probabilities.max(dim=1)
            correct = (predictions == labels).float()

            # For each neuron, measure correlation with correct confident predictions
            for i in range(n_neurons):
                neuron_activations = outputs[:, i]

                # Confidence-weighted correctness
                weighted_correct = correct * confidence

                # Correlation between neuron activation and confident correct predictions
                if neuron_activations.std() > 0 and weighted_correct.std() > 0:
                    correlation = torch.corrcoef(
                        torch.stack([neuron_activations, weighted_correct])
                    )[0, 1]
                    alignment_scores[i] = correlation.abs()

        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")

        return alignment_scores

    def _generate_labels(self, outputs: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Generate synthetic labels using k-means clustering."""
        # Simple k-means
        centroids = outputs[torch.randperm(outputs.shape[0])[:n_classes]]

        for _ in range(10):  # iterations
            distances = torch.cdist(outputs, centroids)
            labels = distances.argmin(dim=1)

            # Update centroids
            for i in range(n_classes):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = outputs[mask].mean(dim=0)

        return labels
