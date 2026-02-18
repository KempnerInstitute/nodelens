"""
Hub loaders for common pre-trained models (torchvision, timm, Hugging Face).

These are thin factory-like model classes registered into the model registry,
so they can be used from configs as regular models:

  - name: "torchvision_model", with params {model_name: str, pretrained: bool, weights: Optional[str]}
  - name: "timm_model", with params {model_name: str, pretrained: bool}
  - name: "hf_vision_model", with params {model_id: str, revision: Optional[str], trust_remote_code: bool}
  - name: "hf_causal_lm", with params {model_id: str, revision: Optional[str], dtype: Optional[str], device_map: Optional[str]}

Notes:
  - Imports are lazy and optional. If a backend is missing, a clear error is raised.
  - Returned objects are nn.Module compatible with our wrappers.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from ..core.registry import register_model

logger = logging.getLogger(__name__)


def _to_torch_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(dtype_str.lower(), None)


def adapt_model_for_dataset(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    pretrained: bool = True,
) -> nn.Module:
    """Adapt a pretrained model's stem for a target dataset resolution.

    Handles common architecture adjustments needed when transferring
    ImageNet-pretrained models to lower-resolution datasets:

    - **CIFAR (32x32)**: ResNet gets 3x3 conv1 stride=1 + Identity maxpool;
      MobileNetV2 gets stride=1 on its first conv.
    - **Tiny-ImageNet (64x64)**: ResNet gets 3x3 conv1 stride=1 (keeps maxpool
      for 64->32->16 spatial progression); MobileNetV2 gets stride=1 first conv.
    - **ImageNet (224x224)** and higher: no changes needed.

    Args:
        model: The nn.Module to adapt (modified in-place).
        model_name: Lowercase model name (e.g. "resnet18", "vgg16", "mobilenetv2").
        dataset_name: Lowercase dataset name (e.g. "cifar100", "tinyimagenet").
        pretrained: Whether the model was loaded with pretrained weights
            (used to seed new conv from old weights when possible).

    Returns:
        The same model, adapted in-place.
    """
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()

    is_cifar = "cifar" in dataset_name
    is_tinyimagenet = "tinyimagenet" in dataset_name

    if not (is_cifar or is_tinyimagenet):
        return model  # No stem changes needed for larger resolutions

    # --- ResNet stem adaptation ---
    if "resnet" in model_name and hasattr(model, "conv1"):
        conv1 = model.conv1
        needs_stem = isinstance(conv1, nn.Conv2d) and (
            tuple(conv1.kernel_size) != (3, 3) or tuple(conv1.stride) != (1, 1)
        )
        if needs_stem:
            new_conv = nn.Conv2d(
                conv1.in_channels, conv1.out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            )
            # Seed from pretrained 7x7 weights by center-cropping
            if pretrained and hasattr(conv1, "weight") and conv1.weight.shape[-1] == 7:
                with torch.no_grad():
                    new_conv.weight.copy_(conv1.weight[:, :, 2:5, 2:5])
            model.conv1 = new_conv
            # CIFAR: also remove maxpool (32->16 immediately)
            # Tiny-ImageNet: keep maxpool (64->32 is fine for 4 stages)
            if is_cifar and hasattr(model, "maxpool"):
                model.maxpool = nn.Identity()

    # --- MobileNetV2 stem adaptation ---
    if "mobilenet" in model_name:
        try:
            conv0 = model.features[0][0]
            if isinstance(conv0, nn.Conv2d) and conv0.stride != (1, 1):
                if is_cifar:
                    # In-place stride change for CIFAR (32x32)
                    conv0.stride = (1, 1)
                elif is_tinyimagenet:
                    # Replace with stride=1 conv for Tiny-ImageNet (64x64)
                    model.features[0][0] = nn.Conv2d(
                        conv0.in_channels, conv0.out_channels,
                        kernel_size=conv0.kernel_size, stride=1,
                        padding=conv0.padding, bias=False,
                    )
        except (IndexError, AttributeError):
            pass

    return model


@register_model("torchvision_model")
class TorchvisionModel(nn.Module):
    """Load a torchvision classification model by name.

    Args:
        model_name: e.g., 'resnet18', 'resnet50', 'mobilenet_v2', 'vgg16', ...
        pretrained: use pretrained ImageNet weights if available
        weights: optional weights enum/name string for newer torchvision APIs
        model_kwargs: forwarded to model constructor
    """

    def __init__(self, model_name: str, pretrained: bool = True, weights: Optional[str] = None, num_classes: int = None, **model_kwargs: Any):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as e:
            raise ImportError("torchvision is required for 'torchvision_model'") from e

        if not hasattr(tvm, model_name):
            raise ValueError(f"Unknown torchvision model: {model_name}")

        model_fn = getattr(tvm, model_name)

        # Extract num_classes from model_kwargs if present
        if num_classes is None:
            num_classes = model_kwargs.pop("num_classes", None)

        # Load model with pretrained weights (using ImageNet classes)
        if weights is not None:
            self.model = model_fn(weights=weights)
        else:
            # Fallback: many models still accept pretrained
            try:
                self.model = model_fn(pretrained=pretrained)
            except TypeError:
                self.model = model_fn()

        # Modify classifier for different number of classes if needed
        if num_classes is not None and num_classes != 1000:
            self._modify_classifier(model_name, num_classes)

    def _modify_classifier(self, model_name: str, num_classes: int):
        """Modify the classifier layer for different number of classes."""
        import torch.nn as nn

        if model_name.startswith("resnet"):
            # ResNet models have 'fc' as final layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif model_name.startswith("vgg"):
            # VGG models have classifier Sequential with final Linear layer
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name.startswith("alexnet"):
            # AlexNet has classifier Sequential with final Linear layer
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name.startswith("densenet"):
            # DenseNet models have 'classifier' as final layer
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif model_name.startswith("mobilenet"):
            # MobileNet models have classifier Sequential with final Linear layer
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif model_name.startswith("efficientnet"):
            # EfficientNet models have 'classifier' as final layer
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        else:
            logger.warning(f"Unknown model architecture '{model_name}' - classifier modification may fail")
            # Try common patterns
            if hasattr(self.model, "fc"):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(self.model, "classifier"):
                if isinstance(self.model.classifier, nn.Sequential):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = self.model.classifier.in_features
                    self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@register_model("timm_model")
class TIMMModel(nn.Module):
    """Load a timm model by name.

    Args:
        model_name: e.g., 'efficientnet_b0', 'resnet50', 'vit_base_patch16_224', ...
        pretrained: use pretrained weights if available
        model_kwargs: forwarded to timm.create_model
    """

    def __init__(self, model_name: str, pretrained: bool = True, **model_kwargs: Any):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("timm is required for 'timm_model'") from e

        self.model = timm.create_model(model_name, pretrained=pretrained, **model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@register_model("hf_vision_model")
class HFVisionModel(nn.Module):
    """Load a Hugging Face vision backbone (e.g., ViT/DeiT) as a PyTorch nn.Module.

    Args:
        model_id: e.g., 'google/vit-base-patch16-224'
        revision: optional git revision
        trust_remote_code: allow custom code in the repo
        torch_dtype: 'float32'|'float16'|'bfloat16' (maps to HF 'dtype' parameter)
        device_map: optional device_map for accelerate (None|'auto')
        model_kwargs: forwarded to from_pretrained
    """

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        **model_kwargs: Any,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:
            raise ImportError("transformers is required for 'hf_vision_model'") from e

        dtype = _to_torch_dtype(torch_dtype)
        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
            device_map=device_map,
            **model_kwargs,
        )

    def forward(self, *args, **kwargs):  # pass-through; wrapper will handle inputs
        return self.model(*args, **kwargs)


@register_model("hf_causal_lm")
class HFCausalLM(nn.Module):
    """Load a Hugging Face Causal LM (e.g., Llama/Mistral/GPT-2).

    Args:
        model_id: e.g., 'meta-llama/Meta-Llama-3-8B-Instruct'
        revision: optional git revision
        trust_remote_code: allow custom code
        torch_dtype: 'float32'|'float16'|'bfloat16' (maps to HF 'dtype' parameter)
        device_map: optional device_map for accelerate (None|'auto')
        model_kwargs: forwarded to from_pretrained
    """

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        **model_kwargs: Any,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:
            raise ImportError("transformers is required for 'hf_causal_lm'") from e

        dtype = _to_torch_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=dtype,  # Use 'dtype' instead of deprecated 'torch_dtype'
            device_map=device_map,
            **model_kwargs,
        )

    def forward(self, *args, **kwargs):  # pass-through
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generate() to the underlying HF model for text generation."""
        return self.model.generate(*args, **kwargs)

    @property
    def config(self):
        """Access the underlying model's config."""
        return self.model.config

    @property
    def device(self):
        """Get the device of the underlying model."""
        return next(self.model.parameters()).device
