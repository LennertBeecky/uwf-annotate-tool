"""Small UNet for Phase-0 A vs C vs G sanity.

- 4 down-blocks, base 32 channels → 5-10 M params
- bilinear upsampling + concat skip
- Output: 2-channel logits (artery, vein)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd-dim skips (shouldn't happen at 512² but defend)
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class SmallUNet(nn.Module):
    """Small UNet. Base channels 32, 4 down-blocks. Output 2 logits/pixel."""

    def __init__(self, in_channels: int = 3, out_channels: int = 2, base: int = 32):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.out_conv = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out_conv(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SmallUNet(in_channels=3, out_channels=2, base=32)
    print(f"params: {count_params(m):,}")
    x = torch.randn(2, 3, 512, 512)
    y = m(x)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
