"""
TinyUNet Architecture for Dense 2D Hotspot Prediction.
Stage 5b of the DFT-Aware Physical Design Hotspot Prediction framework.

Provides:
- DoubleConv: Standard 2-stage Conv2d -> BatchNorm2d -> ReLU block.
- Down: MaxPool2d downsampling followed by DoubleConv.
- Up: ConvTranspose2d upsampling with skip-connection concatenation and DoubleConv.
- TinyUNet: Compact 2D U-Net suited for (64, 64) grid-level routing density prediction.
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    nn = object  # type: ignore


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with ConvTranspose2d then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if sizes do not match exactly due to rounding
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    """
    Compact 2D U-Net baseline for dense routing congestion and wirelength density prediction.

    Default configuration for (64, 64) grids:
        in_channels: 12 (fused input channels)
        base_channels: 16
        out_channels: 1 (total_wirelength_density)
        final_activation: 'sigmoid' (bounds predictions to [0, 1])

    Parameter count: ~440K parameters (~1.8 MB).
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        base_channels: int = 16,
        bilinear: bool = False,
        final_activation: str = "sigmoid",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for TinyUNet.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.final_activation = final_activation.lower() if final_activation else "none"

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)               # 16 -> 32
        self.down2 = Down(base_channels * 2, base_channels * 4)           # 32 -> 64
        self.down3 = Down(base_channels * 4, (base_channels * 8) // factor)  # 64 -> 128 (or 64 if bilinear)
        self.up1 = Up(base_channels * 8, (base_channels * 4) // factor, bilinear)  # 128 -> 64
        self.up2 = Up(base_channels * 4, (base_channels * 2) // factor, bilinear)  # 64 -> 32
        self.up3 = Up(base_channels * 2, base_channels, bilinear)         # 32 -> 16
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        if self.final_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif self.final_activation == "relu":
            self.act = nn.ReLU()
        elif self.final_activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown final_activation: {final_activation}. Expected 'sigmoid', 'relu', or 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return self.act(logits)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
