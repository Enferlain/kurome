"""Simple CNN classifier for spatial tensor artifacts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Small residual conv block used by TensorCNNModel."""

    def __init__(self, channels: int, dropout_rate: float):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(self.dropout(F.gelu(self.norm2(x))))
        return x + residual


class DownsampleBlock(nn.Module):
    """Conv + residual stack with stride-2 downsampling."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.block = ResidualConvBlock(out_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm(self.down(x)))
        return self.block(x)


class TensorCNNModel(nn.Module):
    """Variable-resolution CNN classifier for `[C,H,W]` forensic tensors."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.1,
        output_mode: str = "linear",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.output_mode = (output_mode or "linear").lower()

        base_channels = max(32, hidden_dim // 4)
        c1 = base_channels
        c2 = min(hidden_dim // 2, 256)
        c3 = min(hidden_dim, 512)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(*[ResidualConvBlock(c1, dropout_rate=dropout_rate) for _ in range(max(1, num_res_blocks))])
        self.stage2 = DownsampleBlock(c1, c2, dropout_rate=dropout_rate)
        self.stage3 = DownsampleBlock(c2, c3, dropout_rate=dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(c3),
            nn.Dropout(dropout_rate),
            nn.Linear(c3, num_classes),
        )

    def _apply_output_mode(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "linear":
            return logits
        if self.output_mode == "sigmoid":
            return torch.sigmoid(logits)
        if self.output_mode == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.output_mode == "tanh":
            return torch.tanh(logits)
        raise ValueError(f"Unsupported output_mode '{self.output_mode}'.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.stem(pixel_values)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        logits = self.head(self.pool(x))
        return self._apply_output_mode(logits)
