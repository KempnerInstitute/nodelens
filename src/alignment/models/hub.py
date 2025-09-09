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

from typing import Optional, Any
import logging
import torch
import torch.nn as nn

from ..core.registry import register_model

logger = logging.getLogger(__name__)


def _to_torch_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    if dtype_str is None:
        return None
    mapping = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    return mapping.get(dtype_str.lower(), None)


@register_model("torchvision_model")
class TorchvisionModel(nn.Module):
    """Load a torchvision classification model by name.

    Args:
        model_name: e.g., 'resnet18', 'resnet50', 'mobilenet_v2', 'vgg16', ...
        pretrained: use pretrained ImageNet weights if available
        weights: optional weights enum/name string for newer torchvision APIs
        model_kwargs: forwarded to model constructor
    """

    def __init__(self, model_name: str, pretrained: bool = True, weights: Optional[str] = None, **model_kwargs: Any):
        super().__init__()
        try:
            import torchvision.models as tvm
        except Exception as e:
            raise ImportError("torchvision is required for 'torchvision_model'") from e

        if not hasattr(tvm, model_name):
            raise ValueError(f"Unknown torchvision model: {model_name}")

        model_fn = getattr(tvm, model_name)
        # Newer torchvision uses 'weights' over 'pretrained'
        if weights is not None:
            self.model = model_fn(weights=weights, **model_kwargs)
        else:
            # Fallback: many models still accept pretrained
            try:
                self.model = model_fn(pretrained=pretrained, **model_kwargs)
            except TypeError:
                self.model = model_fn(**model_kwargs)

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
        torch_dtype: 'float32'|'float16'|'bfloat16'
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
            torch_dtype=dtype,
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
        torch_dtype: 'float32'|'float16'|'bfloat16'
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
            torch_dtype=dtype,
            device_map=device_map,
            **model_kwargs,
        )

    def forward(self, *args, **kwargs):  # pass-through
        return self.model(*args, **kwargs)


