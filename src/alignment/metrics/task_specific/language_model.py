"""
Language model-specific alignment metrics.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ...core.base import BaseMetric
from ...core.registry import register_metric


@register_metric("language_model_alignment")
class LanguageModelAlignment(BaseMetric):
    """
    Measures alignment for language modeling tasks.

    This metric evaluates how well neuron activations align with linguistic
    properties such as syntax, semantics, and prediction patterns.
    """

    name = "language_model_alignment"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False

    def __init__(self, vocab_size: int, alignment_type: str = "prediction_alignment", context_window: int = 5, semantic_dims: Optional[int] = None):
        """
        Initialize language model alignment metric.

        Args:
            vocab_size: Size of the vocabulary
            alignment_type: Type of alignment measure
                - 'prediction_alignment': Alignment with next-token predictions
                - 'attention_correlation': Correlation with attention patterns
                - 'semantic_coherence': Semantic coherence of representations
            context_window: Size of context window for analysis
            semantic_dims: Dimensionality for semantic analysis
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.alignment_type = alignment_type
        self.context_window = context_window
        self.semantic_dims = semantic_dims or 128

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute language model alignment scores.

        Args:
            inputs: Input activations [batch_size, seq_len, input_dim]
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations [batch_size, seq_len, output_dim]
            token_ids: Token IDs [batch_size, seq_len]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
            embeddings: Token embeddings [vocab_size, embedding_dim]

        Returns:
            Alignment scores for each neuron [output_dim]
        """
        # Handle 2D inputs by adding sequence dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        if outputs is None:
            # inputs: [batch, seq, input_dim], weights: [output_dim, input_dim]
            outputs = torch.matmul(inputs, weights.T)

        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)

        batch_size, seq_len, n_neurons = outputs.shape
        alignment_scores = torch.zeros(n_neurons, device=outputs.device)

        if self.alignment_type == "prediction_alignment":
            # Measure alignment with next-token prediction task
            if token_ids is None:
                # Generate synthetic token IDs
                token_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=outputs.device)

            # Create next-token targets
            if seq_len > 1:
                targets = token_ids[:, 1:]  # Next tokens
                current_outputs = outputs[:, :-1]  # Current outputs

                # Project outputs to vocabulary space
                projection = torch.randn(n_neurons, self.vocab_size, device=outputs.device)
                torch.matmul(current_outputs, projection)

                # Compute cross-entropy loss per neuron
                for i in range(n_neurons):
                    neuron_contribution = current_outputs[:, :, i : i + 1]
                    neuron_logits = neuron_contribution @ projection[i : i + 1, :]

                    # Flatten for loss computation
                    neuron_logits_flat = neuron_logits.reshape(-1, self.vocab_size)
                    targets_flat = targets.reshape(-1)

                    # Cross-entropy loss
                    loss = F.cross_entropy(neuron_logits_flat, targets_flat, reduction="mean")

                    # Lower loss = better alignment
                    alignment_scores[i] = 1.0 / (1.0 + loss.item())

        elif self.alignment_type == "attention_correlation":
            # Measure correlation with attention patterns
            if attention_weights is None:
                # Generate synthetic attention weights
                attention_weights = F.softmax(torch.randn(batch_size, 4, seq_len, seq_len, device=outputs.device), dim=-1)

            # Average attention across heads
            avg_attention = attention_weights.mean(dim=1)  # [batch, seq, seq]

            # For each neuron, measure correlation with attention-weighted representations
            for i in range(n_neurons):
                neuron_activations = outputs[:, :, i]  # [batch, seq]

                # Compute attention-weighted activations
                weighted_activations = torch.matmul(avg_attention, neuron_activations.unsqueeze(-1)).squeeze(-1)

                # Correlation between original and attention-weighted
                if neuron_activations.numel() > 1:
                    flat_original = neuron_activations.reshape(-1)
                    flat_weighted = weighted_activations.reshape(-1)

                    if flat_original.std() > 0 and flat_weighted.std() > 0:
                        correlation = torch.corrcoef(torch.stack([flat_original, flat_weighted]))[0, 1]
                        alignment_scores[i] = correlation.abs()

        elif self.alignment_type == "semantic_coherence":
            # Measure semantic coherence of representations
            if embeddings is None:
                # Generate synthetic embeddings
                embeddings = torch.randn(self.vocab_size, self.semantic_dims, device=outputs.device)

            if token_ids is None:
                token_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=outputs.device)

            # Get embeddings for tokens
            token_embeddings = embeddings[token_ids]  # [batch, seq, embed_dim]

            # Project outputs to semantic space
            projection = torch.randn(n_neurons, self.semantic_dims, device=outputs.device)
            torch.matmul(outputs, projection)

            # Measure coherence for each neuron
            for i in range(n_neurons):
                neuron_outputs = outputs[:, :, i : i + 1]  # [batch, seq, 1]
                neuron_embeddings = neuron_outputs @ projection[i : i + 1, :]  # [batch, seq, embed_dim]

                # Cosine similarity with token embeddings
                neuron_norm = F.normalize(neuron_embeddings, p=2, dim=-1)
                token_norm = F.normalize(token_embeddings, p=2, dim=-1)

                similarities = (neuron_norm * token_norm).sum(dim=-1)  # [batch, seq]

                # Average similarity as coherence measure
                alignment_scores[i] = similarities.mean()

        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")

        return alignment_scores
