from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch.nn as nn
from torch import Tensor

type ND = Literal[2, 3]


def pick_layers(nd: ND):
    Conv = {2: nn.Conv2d, 3: nn.Conv3d}[nd]
    MaxPool = {2: nn.MaxPool2d, 3: nn.MaxPool3d}[nd]
    AdaptiveAvgPool = {2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[nd]
    return Conv, MaxPool, AdaptiveAvgPool


Size2 = int | tuple[int, int]
Size3 = int | tuple[int, int, int]
Padding2 = Size2
Padding3 = Size3


class BaseEncoder(nn.Module, ABC):
    """
    Dim-agnostic encoder producing a feature map from a 2D or 3D shoebox.
    """

    def __init__(
        self,
        nd: ND,
        *,
        in_channels: int,
        c1: int = 16,
        k1: Size3 | Size2 = (3, 3, 3),
        p1: Padding3 | Padding2 = 0,
        c2: int = 32,
        k2: Size3 | Size2 = (3, 3, 3),
        p2: Padding3 | Padding2 = 0,
        pool_k: Size3 | Size2 = 2,
        pool_s: Size3 | Size2 = 2,
        gn1_groups: int = 4,
        gn2_groups: int = 4,
    ):
        """

        Args:
            nd: diffraction data dimensionality
            in_channels: number of input channels
            c1:
            k1:
            p1:
            c2:
            k2:
            p2:
            pool_k:
            pool_s:
            gn1_groups:
            gn2_groups:
        """
        super().__init__()
        Conv, MaxPool, _ = pick_layers(nd)

        self.conv1 = Conv(in_channels, c1, kernel_size=k1, padding=p1)
        self.norm1 = nn.GroupNorm(gn1_groups, c1)
        self.pool = MaxPool(kernel_size=pool_k, stride=pool_s, ceil_mode=True)
        self.conv2 = Conv(c1, c2, kernel_size=k2, padding=p2)
        self.norm2 = nn.GroupNorm(gn2_groups, c2)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...
