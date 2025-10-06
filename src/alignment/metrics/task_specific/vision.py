"""
Vision task-specific alignment metrics.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ...core.base import BaseMetric
from ...core.registry import register_metric

logger = logging.getLogger(__name__)


@register_metric("vision_task_alignment")
class VisionTaskAlignment(BaseMetric):
    """
    Measures alignment for vision tasks.

    This metric evaluates how well neuron activations align with visual features
    such as edges, textures, objects, and spatial relationships.
    """

    name = "vision_task_alignment"
    requires_inputs = True
    requires_weights = True
    requires_outputs = False

    def __init__(
        self,
        alignment_type: str = "spatial_coherence",
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        n_orientations: int = 8,
        require_real_data: bool = False,
    ):
        """
        Initialize vision task alignment metric.

        Args:
            alignment_type: Type of alignment measure
                - 'spatial_coherence': Spatial coherence of activations
                - 'edge_detection': Alignment with edge detection
                - 'texture_response': Response to texture patterns
                - 'object_selectivity': Object-specific selectivity
            image_size: Expected image size (height, width)
            patch_size: Size of patches for patch-based analysis
            n_orientations: Number of orientations for edge detection
        """
        super().__init__()
        self.alignment_type = alignment_type
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.require_real_data = require_real_data

    def compute(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        feature_maps: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute vision task alignment scores.

        Args:
            inputs: Input activations [batch_size, channels, height, width] or flattened
            weights: Weight matrix [output_dim, input_dim]
            outputs: Output activations
            images: Original images [batch_size, channels, height, width]
            feature_maps: Intermediate feature maps
            labels: Object labels for images

        Returns:
            Alignment scores for each neuron [output_dim]
        """
        if outputs is None:
            # Handle different input shapes
            if inputs.dim() == 4:
                # Conv input: flatten spatial dimensions
                batch_size, channels, h, w = inputs.shape
                inputs_flat = inputs.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
                outputs = torch.matmul(inputs_flat, weights.T)
                outputs = outputs.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
            else:
                # Already flattened
                outputs = inputs @ weights.T
        device = outputs.device

        n_neurons = weights.shape[0]
        alignment_scores = torch.zeros(n_neurons, device=outputs.device)

        if self.alignment_type == "spatial_coherence":
            # Measure spatial coherence of neuron activations
            if outputs.dim() == 2:
                # Expect flattened spatial maps concatenated per neuron: [B, n_neurons * H * W]
                total_features = outputs.shape[1]
                if total_features % n_neurons != 0:
                    raise ValueError(f"Cannot infer spatial dims: features={total_features} not divisible by neurons={n_neurons}")
                spatial_area = total_features // n_neurons
                spatial_size = int(spatial_area**0.5)
                if spatial_size * spatial_size != spatial_area:
                    raise ValueError(f"Non-square spatial area inferred: area={spatial_area}. Provide conv outputs or proper preprocessing.")
                batch_size = outputs.shape[0]
                outputs = outputs.reshape(batch_size, n_neurons, spatial_size, spatial_size)
            elif outputs.dim() == 4:
                # Already in spatial format
                pass
            else:
                raise ValueError(f"Unexpected output dimension: {outputs.dim()}")

            # Compute spatial autocorrelation for each neuron
            for i in range(n_neurons):
                if outputs.dim() == 4:
                    neuron_maps = outputs[:, i]  # [batch, height, width]
                else:
                    neuron_maps = outputs[:, :, i]  # Alternative format

                # Compute local spatial correlation
                # Shift maps and compute correlation
                shifted_right = torch.roll(neuron_maps, shifts=1, dims=-1)
                shifted_down = torch.roll(neuron_maps, shifts=1, dims=-2)

                # Average correlation with shifted versions
                corr_right = F.cosine_similarity(
                    neuron_maps.reshape(neuron_maps.shape[0], -1), shifted_right.reshape(shifted_right.shape[0], -1), dim=1
                ).mean()

                corr_down = F.cosine_similarity(
                    neuron_maps.reshape(neuron_maps.shape[0], -1), shifted_down.reshape(shifted_down.shape[0], -1), dim=1
                ).mean()

                alignment_scores[i] = (corr_right + corr_down) / 2

        elif self.alignment_type == "edge_detection":
            # Measure alignment with edge detection filters
            # Create Sobel-like edge detection kernels

            # Horizontal and vertical edge kernels
            kernel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
            kernel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)

            if images is None:
                if self.require_real_data:
                    raise ValueError("VisionTaskAlignment(edge_detection) requires images but none were provided.")
                logger.warning("images not provided; generating synthetic edge patterns for alignment computation")
                images = self._generate_edge_patterns(outputs.shape[0], device)

            # Compute edge responses in images
            if images.dim() == 4 and images.shape[1] == 3:
                # Convert to grayscale
                images_gray = images.mean(dim=1, keepdim=True)
            else:
                images_gray = images

            # Apply edge detection
            edges_h = F.conv2d(images_gray, kernel_h.unsqueeze(0).unsqueeze(0), padding=1)
            edges_v = F.conv2d(images_gray, kernel_v.unsqueeze(0).unsqueeze(0), padding=1)
            edge_magnitude = torch.sqrt(edges_h**2 + edges_v**2)

            # Measure correlation between neuron activations and edges
            for i in range(n_neurons):
                if outputs.dim() == 4:
                    neuron_response = outputs[:, i : i + 1]
                else:
                    # Reshape if needed
                    neuron_response = outputs[:, i].reshape(outputs.shape[0], 1, -1)
                    neuron_response = neuron_response.reshape(
                        outputs.shape[0], 1, int(neuron_response.shape[2] ** 0.5), int(neuron_response.shape[2] ** 0.5)
                    )

                # Resize if necessary
                if neuron_response.shape[-2:] != edge_magnitude.shape[-2:]:
                    neuron_response = F.interpolate(neuron_response, size=edge_magnitude.shape[-2:], mode="bilinear")

                # Compute correlation
                correlation = F.cosine_similarity(
                    neuron_response.reshape(neuron_response.shape[0], -1), edge_magnitude.reshape(edge_magnitude.shape[0], -1), dim=1
                ).mean()

                alignment_scores[i] = correlation.abs()

        elif self.alignment_type == "texture_response":
            # Measure response to texture patterns
            if feature_maps is None:
                if self.require_real_data:
                    raise ValueError("VisionTaskAlignment(texture_response) requires feature_maps but none were provided.")
                logger.warning("feature_maps not provided; generating synthetic texture patterns for alignment computation")
                feature_maps = self._generate_texture_patterns(outputs.shape[0], n_neurons, device)

            # Compute texture energy for different frequencies
            for i in range(n_neurons):
                if outputs.dim() == 4:
                    neuron_response = outputs[:, i]
                else:
                    neuron_response = outputs[:, i].reshape(-1)

                # Compute frequency response using FFT
                if neuron_response.dim() >= 2:
                    fft_response = torch.fft.fft2(neuron_response)
                    power_spectrum = torch.abs(fft_response) ** 2

                    # Measure concentration of energy in different frequency bands
                    low_freq_energy = power_spectrum[..., : power_spectrum.shape[-2] // 4, : power_spectrum.shape[-1] // 4].sum()
                    high_freq_energy = power_spectrum[..., power_spectrum.shape[-2] // 4 :, power_spectrum.shape[-1] // 4 :].sum()

                    # Ratio indicates texture selectivity
                    if low_freq_energy > 0:
                        texture_selectivity = high_freq_energy / low_freq_energy
                    else:
                        texture_selectivity = high_freq_energy

                    alignment_scores[i] = texture_selectivity / (1 + texture_selectivity)
                else:
                    alignment_scores[i] = 0.0

        elif self.alignment_type == "object_selectivity":
            # Measure object-specific selectivity
            if labels is None:
                if self.require_real_data:
                    raise ValueError("VisionTaskAlignment(object_selectivity) requires labels but none were provided.")
                logger.warning("labels not provided; generating synthetic labels for alignment computation")
                # Create synthetic object labels
                n_objects = 10
                labels = torch.randint(0, n_objects, (outputs.shape[0],), device=device)

            # Compute selectivity for each neuron
            unique_labels = labels.unique()

            for i in range(n_neurons):
                if outputs.dim() == 4:
                    neuron_response = outputs[:, i].reshape(outputs.shape[0], -1).mean(dim=1)
                else:
                    neuron_response = outputs[:, i]

                # Compute mean response per object class
                class_responses = []
                for label in unique_labels:
                    mask = labels == label
                    if mask.sum() > 0:
                        class_responses.append(neuron_response[mask].mean())

                if len(class_responses) > 1:
                    class_responses = torch.stack(class_responses)

                    # Selectivity as ratio of max to mean response
                    max_response = class_responses.max()
                    mean_response = class_responses.mean()

                    if mean_response > 0:
                        selectivity = max_response / mean_response
                    else:
                        selectivity = 0.0

                    alignment_scores[i] = selectivity / (1 + selectivity)

        else:
            raise ValueError(f"Unknown alignment type: {self.alignment_type}")

        return alignment_scores

    def _generate_edge_patterns(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate synthetic images with edge patterns."""
        h, w = self.image_size
        images = torch.zeros(batch_size, 1, h, w, device=device)

        for i in range(batch_size):
            # Random edge orientation
            angle = torch.rand(1).item() * 2 * 3.14159

            # Create edge pattern
            x = torch.arange(w, device=device).float() - w / 2
            y = torch.arange(h, device=device).float() - h / 2
            xx, yy = torch.meshgrid(x, y, indexing="xy")

            # Rotated coordinates
            edge_pattern = torch.sin(xx * torch.cos(angle) + yy * torch.sin(angle))
            images[i, 0] = edge_pattern

        return images

    def _generate_texture_patterns(self, batch_size: int, n_channels: int, device: torch.device) -> torch.Tensor:
        """Generate synthetic texture patterns."""
        h, w = self.image_size
        textures = torch.randn(batch_size, n_channels, h, w, device=device)

        # Apply different frequency filters
        for i in range(n_channels):
            2 ** (i % 5)  # Different frequencies
            kernel_size = max(3, 15 - 2 * (i % 5))

            # Create Gaussian kernel for smoothing
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)

            # Apply convolution for texture effect
            textures[:, i : i + 1] = F.conv2d(textures[:, i : i + 1], kernel, padding=kernel_size // 2)

        return textures
