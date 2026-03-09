import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """
    Residual block:
      main: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
      skip: Conv1x1 -> BN (if needed)
      out: ReLU(main + skip)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.skip is not None:
            identity = self.skip(identity)

        out = self.relu(out + identity)
        return out


class ResUNet(nn.Module):
    """
    ResUNet (UNet encoder-decoder with residual conv blocks).

    Args:
        in_channels: 3 for your projections (ILM_OPL, OPL_BM, FULL)
        n_classes: 1 for binary logits
        base: base filters (e.g., 42)
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base: int = 42):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)

        # Bottleneck
        self.bottleneck = ResidualConvBlock(c4, c5)

        # Decoder: upsample + concat skip + residual block
        self.dec4 = ResidualConvBlock(c5 + c4, c4)
        self.dec3 = ResidualConvBlock(c4 + c3, c3)
        self.dec2 = ResidualConvBlock(c3 + c2, c2)
        self.dec1 = ResidualConvBlock(c2 + c1, c1)

        self.final = nn.Conv2d(c1, n_classes, kernel_size=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # (B,c1,H,W)
        e2 = self.enc2(self.pool(e1))  # (B,c2,H/2,W/2)
        e3 = self.enc3(self.pool(e2))  # (B,c3,H/4,W/4)
        e4 = self.enc4(self.pool(e3))  # (B,c4,H/8,W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B,c5,H/16,W/16)

        # Decoder
        d4 = self._up(b, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self._up(d4, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._up(d3, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._up(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)  # logits (B,n_classes,H,W)
