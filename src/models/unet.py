"""Small U-Net for image denoising (PyTorch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net: encoder-decoder with skip connections. in_channels/out_channels typically 1 for grayscale."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.depth = depth
        ch = [base_channels * (2 ** i) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for i in range(depth):
            self.enc.append(DoubleConv(ch[i] if i == 0 else ch[i], ch[i + 1]))
        self.bottleneck = DoubleConv(ch[depth], ch[depth] * 2)
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        # up[0]: bottleneck out (2*ch[depth]) -> ch[depth]; dec[0]: cat -> ch[depth-1]
        # up[i]: dec[i-1] out (ch[depth-i]) -> ch[depth-i-1]; dec[i]: cat (up_out + skip) -> ch[depth-i-1]
        self.up.append(nn.ConvTranspose2d(ch[depth] * 2, ch[depth], 2, stride=2))
        self.dec.append(DoubleConv(ch[depth] * 2, ch[depth - 1]))
        for i in range(1, depth):
            self.up.append(nn.ConvTranspose2d(ch[depth - i], ch[depth - i - 1], 2, stride=2))
            self.dec.append(DoubleConv(ch[depth - i - 1] + ch[depth - i], ch[depth - i - 1]))
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        self.in_conv = DoubleConv(in_channels, base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = self.in_conv(x)
        skips.append(h)
        for i in range(self.depth):
            h = self.pool(h)
            h = self.enc[i](h)
            skips.append(h)
        h = self.bottleneck(h)
        for i in range(self.depth):
            h = self.up[i](h)
            skip = skips[self.depth - i]
            skip = F.interpolate(skip, size=h.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = self.dec[i](h)
        return self.final(h)
