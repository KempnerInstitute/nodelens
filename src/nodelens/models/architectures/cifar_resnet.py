"""CIFAR-style ResNets with explicit width multipliers.

Torchvision's ResNet-18 is convenient for the full-width runs, but its
``BasicBlock`` does not support width scaling through ``width_per_group``.
These small modules provide a controlled trained-from-scratch width axis for
capacity experiments.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class CIFARBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CIFARResNet(nn.Module):
    """ResNet for 32x32 images with a controllable channel multiplier."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        width_multiplier: float = 1.0,
        base_width: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        width = max(8, int(round(float(base_width) * float(width_multiplier) / 8.0)) * 8)
        channels = [width, 2 * width, 4 * width, 8 * width]
        self.in_channels = channels[0]

        self.conv1 = _conv3x3(3, channels[0])
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], int(layers[0]), stride=1)
        self.layer2 = self._make_layer(channels[1], int(layers[1]), stride=2)
        self.layer3 = self._make_layer(channels[2], int(layers[2]), stride=2)
        self.layer4 = self._make_layer(channels[3], int(layers[3]), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)
        self._init_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [CIFARBasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(CIFARBasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def cifar_resnet18(*, num_classes: int = 100, width_multiplier: float = 1.0, base_width: int = 64) -> CIFARResNet:
    return CIFARResNet(num_classes=num_classes, width_multiplier=width_multiplier, base_width=base_width, layers=(2, 2, 2, 2))
