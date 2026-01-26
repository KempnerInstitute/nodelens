"""
Semantic / interpretation-facing analyses that can be computed from trained models.

These are intentionally model-agnostic utilities that can be
reused for:
- relating discovered channel clusters to semantic properties (e.g., class selectivity)
- sanity checks about what clusters/metrics "mean" beyond pruning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


@dataclass
class ClassSelectivityResult:
    """Per-channel class selectivity for one layer."""

    layer_name: str
    activation_point: str  # "pre_bn" or "post_bn"
    num_classes: int
    n_images: int
    selectivity: np.ndarray  # [C]
    mu_max: np.ndarray  # [C]
    mu_other: np.ndarray  # [C]


def _find_bn_for_conv(model: "nn.Module", conv_name: str) -> Optional[Tuple[str, "nn.Module"]]:
    """
    Best-effort BatchNorm lookup for a conv layer name.

    Works across common patterns:
    - ResNet: layerX.Y.convZ -> layerX.Y.bnZ
    - VGG-BN / MobileNet: features.N -> features.(N+1), or ...0.0 -> ...0.1
    """
    if not HAS_TORCH:
        return None

    modules: Dict[str, nn.Module] = dict(model.named_modules())

    candidates = []
    # Name-based conventions
    if "conv" in conv_name:
        candidates.append(conv_name.replace("conv", "bn"))
    if ".conv" in conv_name:
        candidates.append(conv_name.replace(".conv", ".bn"))
    candidates.append(conv_name + "_bn")
    if "downsample.0" in conv_name:
        candidates.append(conv_name.replace("downsample.0", "downsample.1"))

    # Index-based convention: conv at index k, bn at index k+1 in a Sequential.
    parts = conv_name.split(".")
    if parts and parts[-1].isdigit():
        try:
            candidates.append(".".join(parts[:-1] + [str(int(parts[-1]) + 1)]))
        except Exception:
            pass

    for name in candidates:
        m = modules.get(name)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            return name, m
    return None


def compute_class_selectivity(
    *,
    model,
    loader,
    layer_name: str,
    device: str = "cuda",
    activation_point: str = "post_bn",
    max_images: int = 1024,
    reduce: str = "mean_abs",
) -> ClassSelectivityResult:
    """
    Compute Morcos-style class selectivity for a layer's channels.

    We summarize each channel per image as a scalar, then compute the mean response
    per class. Selectivity is:

        sel_i = (mu_max - mu_other) / (mu_max + mu_other)

    where mu_max is the mean response for the most-responsive class and mu_other is
    the mean over the remaining classes.
    """
    if not HAS_TORCH:
        raise RuntimeError("compute_class_selectivity requires PyTorch.")

    import torch

    model = model.to(device)
    model.eval()

    modules: Dict[str, nn.Module] = dict(model.named_modules())
    src = modules.get(layer_name)
    if src is None:
        raise ValueError(f"Layer not found: {layer_name}")

    # Decide which module to hook.
    hook_module = src
    if str(activation_point) == "post_bn":
        bn = _find_bn_for_conv(model, layer_name)
        if bn is not None:
            _bn_name, bn_mod = bn
            hook_module = bn_mod

    acts: Optional["torch.Tensor"] = None

    def _hook(_m, _inp, out):
        nonlocal acts
        acts = out.detach()

    h = hook_module.register_forward_hook(_hook)

    # Infer num_classes from first batch (assume labels are ints in [0,K-1])
    num_classes = None
    sum_by_class = None
    cnt_by_class = None
    n_seen = 0

    with torch.no_grad():
        for x, y in loader:
            if n_seen >= int(max_images):
                break
            b = int(x.size(0))
            remaining = int(max_images) - n_seen
            if b > remaining:
                x = x[:remaining]
                y = y[:remaining]
                b = remaining

            x = x.to(device)
            y = y.to(device)

            acts = None
            _ = model(x)
            if acts is None:
                continue

            a = acts
            # Reduce per image to [B,C]
            if a.ndim == 4:
                if reduce == "mean_abs":
                    a = a.abs().mean(dim=(2, 3))
                elif reduce == "mean":
                    a = a.mean(dim=(2, 3))
                elif reduce == "rms":
                    a = (a * a).mean(dim=(2, 3)).sqrt()
                else:
                    raise ValueError(f"Unknown reduce: {reduce}")
            elif a.ndim == 2:
                if reduce == "mean_abs":
                    a = a.abs()
                elif reduce == "mean":
                    a = a
                elif reduce == "rms":
                    a = a.abs()
                else:
                    raise ValueError(f"Unknown reduce: {reduce}")
            else:
                raise ValueError(f"Unsupported activation shape for selectivity: {tuple(a.shape)}")

            a_cpu = a.detach().cpu().double()  # [B,C]
            y_cpu = y.detach().cpu().long()

            if num_classes is None:
                num_classes = int(y_cpu.max().item()) + 1
                c = int(a_cpu.shape[1])
                sum_by_class = torch.zeros((num_classes, c), dtype=torch.float64)
                cnt_by_class = torch.zeros((num_classes,), dtype=torch.int64)

            # Accumulate
            for cls in torch.unique(y_cpu):
                cls_i = int(cls.item())
                idx = (y_cpu == cls)
                if int(idx.sum().item()) == 0:
                    continue
                sum_by_class[cls_i] += a_cpu[idx].sum(dim=0)
                cnt_by_class[cls_i] += int(idx.sum().item())

            n_seen += b

    h.remove()

    if num_classes is None or sum_by_class is None or cnt_by_class is None:
        raise RuntimeError("No activations collected; check layer_name / loader.")

    # Compute per-class means [K,C]
    cnt = cnt_by_class.clamp_min(1).double().unsqueeze(1)  # [K,1]
    mean_by_class = (sum_by_class / cnt).numpy()  # [K,C]

    mu_max = np.max(mean_by_class, axis=0)
    mu_other = (np.sum(mean_by_class, axis=0) - mu_max) / float(max(1, num_classes - 1))
    sel = (mu_max - mu_other) / (mu_max + mu_other + 1e-12)

    return ClassSelectivityResult(
        layer_name=str(layer_name),
        activation_point=str(activation_point),
        num_classes=int(num_classes),
        n_images=int(n_seen),
        selectivity=sel.astype(np.float64),
        mu_max=mu_max.astype(np.float64),
        mu_other=mu_other.astype(np.float64),
    )

