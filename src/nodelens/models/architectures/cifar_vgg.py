"""CIFAR-style VGG with BatchNorm and width control.

Torchvision's `vgg16_bn` was designed for 224x224 ImageNet inputs; on
32x32 CIFAR images the 5 max-pool stages reduce spatial resolution to 1x1
before the deepest convs and the network typically fails to train. This
module provides a CIFAR-adapted VGG-16-BN: same conv block topology
(64-64, M, 128-128, M, 256-256-256, M, 512-512-512, M, 512-512-512) but
without the fifth pool, no spatial flattening through huge FC layers, and
a small classifier head sized to a 2x2 feature map. A `width_multiplier`
scales all channel counts uniformly.

Two-axis paper context: VGG-16 is the architecture where weight-norm
pruning remains competitive with local-axis methods on the original
benchmark suite. The CIFAR-adapted variant here is the matched testbed
for any replaceability-aware training claim that needs to demonstrate
architecture-independence.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

_VGG16_BN_CFG: List = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]


def _round_width(channels: int, multiplier: float, divisor: int = 8) -> int:
    scaled = max(divisor, int(round(channels * float(multiplier) / divisor)) * divisor)
    return scaled


def _make_features(cfg: List, width_multiplier: float) -> nn.Sequential:
    layers: list = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        out_channels = _round_width(int(v), width_multiplier)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*layers)


class CIFARVGG(nn.Module):
    """VGG-16-BN adapted for 32x32 CIFAR inputs with a width multiplier.

    Topology mirrors torchvision's `vgg16_bn` conv stack except the
    classifier head: instead of three large FC layers operating on a 7x7
    feature map, we use a single linear head on a 2x2 feature map after
    the final pool. This both reduces parameter count to a reasonable
    range for CIFAR and avoids the chance-accuracy failure mode of
    plugging 32x32 inputs into the ImageNet-sized network.
    """

    def __init__(self, *, num_classes: int = 100, width_multiplier: float = 1.0) -> None:
        super().__init__()
        self.features = _make_features(_VGG16_BN_CFG, width_multiplier)
        # After 4 max-pools the 32x32 input is 2x2; final channel count is the
        # last conv's output channels.
        final_channels = _round_width(int(_VGG16_BN_CFG[-1]), width_multiplier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def cifar_vgg16(*, num_classes: int = 100, width_multiplier: float = 1.0) -> CIFARVGG:
    return CIFARVGG(num_classes=num_classes, width_multiplier=width_multiplier)
