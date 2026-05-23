# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight dental lesion neck modules for YOLOv8 ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class DentalECA(nn.Module):
    """Efficient channel attention for dental lesion features."""

    def __init__(self, c1: int, k_size: int = 3):
        """Create a tiny channel-attention block that preserves input shape."""
        super().__init__()
        if k_size % 2 == 0:
            k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel reweighting without changing tensor shape."""
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class DentalEMA(nn.Module):
    """Lightweight efficient multi-scale attention for dental lesion features.

    This block preserves the input shape and is intended for a single P3 feature map. It uses grouped spatial context
    along height/width plus a small local 3x3 branch, keeping the parameter increase modest.
    """

    def __init__(self, c1: int, groups: int = 8):
        super().__init__()
        groups = max(1, min(int(groups), c1))
        while c1 % groups != 0 and groups > 1:
            groups -= 1
        self.groups = groups
        self.c_group = c1 // groups
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1x1 = nn.Conv2d(self.c_group, self.c_group, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(self.c_group, self.c_group, 3, 1, 1)
        self.gn = nn.GroupNorm(1, self.c_group)
        self.softmax = nn.Softmax(dim=-1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped EMA attention without changing feature shape."""
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, self.c_group, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).transpose(2, 3)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * self.sigmoid(x_h) * self.sigmoid(x_w.transpose(2, 3)))
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.avg_pool(x1).flatten(2).transpose(1, 2))
        x12 = x2.flatten(2)
        x21 = self.softmax(self.avg_pool(x2).flatten(2).transpose(1, 2))
        x22 = x1.flatten(2)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * self.sigmoid(weights)).reshape(b, c, h, w)


class _BiFPNAdd(nn.Module):
    """Shared implementation for normalized weighted feature addition."""

    expected_inputs = 0

    def __init__(self, c1_list: list[int], c2: int, eps: float = 1e-4, resize_mode: str = "nearest"):
        super().__init__()
        if len(c1_list) != self.expected_inputs:
            raise ValueError(f"{self.__class__.__name__} expects {self.expected_inputs} input channels, got {len(c1_list)}.")
        if resize_mode not in {"nearest", "bilinear"}:
            raise ValueError("resize_mode must be 'nearest' or 'bilinear'.")
        self.eps = eps
        self.resize_mode = resize_mode
        self.w = nn.Parameter(torch.ones(self.expected_inputs, dtype=torch.float32), requires_grad=True)
        self.proj = nn.ModuleList(Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity() for c1 in c1_list)
        self.conv = Conv(c2, c2, 3, 1)

    def _resize(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        if self.resize_mode == "bilinear":
            return F.interpolate(x, size=size, mode=self.resize_mode, align_corners=False)
        return F.interpolate(x, size=size, mode=self.resize_mode)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Resize to the first feature map, project channels, then fuse with normalized positive weights."""
        if not isinstance(xs, (list, tuple)) or len(xs) != self.expected_inputs:
            raise ValueError(f"{self.__class__.__name__} expects a list/tuple of {self.expected_inputs} tensors.")
        target_size = xs[0].shape[-2:]
        features = [proj(self._resize(x, target_size)) for x, proj in zip(xs, self.proj)]
        weights = torch.relu(self.w)
        weights = weights / (weights.sum() + self.eps)
        fused = sum(weight * feature for weight, feature in zip(weights, features))
        return self.conv(fused)


class BiFPN_Add2(_BiFPNAdd):
    """Two-input lightweight BiFPN weighted addition."""

    expected_inputs = 2


class BiFPN_Add3(_BiFPNAdd):
    """Three-input lightweight BiFPN weighted addition."""

    expected_inputs = 3
