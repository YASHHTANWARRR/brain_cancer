"""
CBAM: Convolutional Block Attention Module
ECCV 2018
Woo et al.

Original implementation adapted for Ultralytics YOLOv8.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.shared_mlp(self.avg_pool(x))
        mx = self.shared_mlp(self.max_pool(x))

        attention = self.sigmoid(avg + mx)

        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        assert kernel_size in (3, 7)

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg, mx], dim=1)

        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


class OriginalCBAM(nn.Module):
    """
    Original CBAM

    Sequential:
        Channel Attention
        Spatial Attention
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size,
        )

    def forward(self, x):

        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x