import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    # Conv 3x3 + BN + ReLU
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    # (ConvBNReLU) x 2
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


def _downsample_to(x, target_hw):
    # Downsample vers target_hw par adaptive pooling (stable)
    return F.adaptive_max_pool2d(x, output_size=target_hw)


def _upsample_to(x, target_hw):
    # Upsample vers target_hw
    return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


class UNet3Plus(nn.Module):
    """
    UNet 3+ (souvent appelé UNet+++ par abus de langage).
    Full-scale skip connections : chaque décodeur agrège plusieurs échelles.

    - in_channels=3
    - n_classes=1 (logits)
    - base_channels=32 (peut être augmenté si GPU OK)
    """

    def __init__(self, in_channels=3, n_classes=1, base_channels=32, mid_channels=32):
        super().__init__()

        # Encoder (4 niveaux + bottleneck)
        self.e1 = DoubleConv(in_channels, base_channels)           # H
        self.e2 = DoubleConv(base_channels, base_channels * 2)     # H/2
        self.e3 = DoubleConv(base_channels * 2, base_channels * 4) # H/4
        self.e4 = DoubleConv(base_channels * 4, base_channels * 8) # H/8
        self.e5 = DoubleConv(base_channels * 8, base_channels * 16)# H/16

        self.pool = nn.MaxPool2d(2)

        # Pour chaque décodeur, on projette chaque source vers mid_channels, puis concat
        # d4 (H/8) agrège e1,e2,e3,e4,e5
        self.d4_e1 = ConvBNReLU(base_channels, mid_channels)
        self.d4_e2 = ConvBNReLU(base_channels * 2, mid_channels)
        self.d4_e3 = ConvBNReLU(base_channels * 4, mid_channels)
        self.d4_e4 = ConvBNReLU(base_channels * 8, mid_channels)
        self.d4_e5 = ConvBNReLU(base_channels * 16, mid_channels)
        self.d4_fuse = DoubleConv(mid_channels * 5, base_channels * 8)

        # d3 (H/4) agrège e1,e2,e3,d4,e5
        self.d3_e1 = ConvBNReLU(base_channels, mid_channels)
        self.d3_e2 = ConvBNReLU(base_channels * 2, mid_channels)
        self.d3_e3 = ConvBNReLU(base_channels * 4, mid_channels)
        self.d3_d4 = ConvBNReLU(base_channels * 8, mid_channels)
        self.d3_e5 = ConvBNReLU(base_channels * 16, mid_channels)
        self.d3_fuse = DoubleConv(mid_channels * 5, base_channels * 4)

        # d2 (H/2) agrège e1,e2,d3,d4,e5
        self.d2_e1 = ConvBNReLU(base_channels, mid_channels)
        self.d2_e2 = ConvBNReLU(base_channels * 2, mid_channels)
        self.d2_d3 = ConvBNReLU(base_channels * 4, mid_channels)
        self.d2_d4 = ConvBNReLU(base_channels * 8, mid_channels)
        self.d2_e5 = ConvBNReLU(base_channels * 16, mid_channels)
        self.d2_fuse = DoubleConv(mid_channels * 5, base_channels * 2)

        # d1 (H) agrège e1,d2,d3,d4,e5
        self.d1_e1 = ConvBNReLU(base_channels, mid_channels)
        self.d1_d2 = ConvBNReLU(base_channels * 2, mid_channels)
        self.d1_d3 = ConvBNReLU(base_channels * 4, mid_channels)
        self.d1_d4 = ConvBNReLU(base_channels * 8, mid_channels)
        self.d1_e5 = ConvBNReLU(base_channels * 16, mid_channels)
        self.d1_fuse = DoubleConv(mid_channels * 5, base_channels)

        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        e5 = self.e5(self.pool(e4))

        # Tailles cibles
        h1, w1 = e1.shape[-2:]
        h2, w2 = e2.shape[-2:]
        h3, w3 = e3.shape[-2:]
        h4, w4 = e4.shape[-2:]
        h5, w5 = e5.shape[-2:]

        # d4 (H/8 = e4)
        d4_1 = self.d4_e1(_downsample_to(e1, (h4, w4)))
        d4_2 = self.d4_e2(_downsample_to(e2, (h4, w4)))
        d4_3 = self.d4_e3(_downsample_to(e3, (h4, w4)))
        d4_4 = self.d4_e4(e4)
        d4_5 = self.d4_e5(_upsample_to(e5, (h4, w4)))
        d4 = self.d4_fuse(torch.cat([d4_1, d4_2, d4_3, d4_4, d4_5], dim=1))

        # d3 (H/4 = e3)
        d3_1 = self.d3_e1(_downsample_to(e1, (h3, w3)))
        d3_2 = self.d3_e2(_downsample_to(e2, (h3, w3)))
        d3_3 = self.d3_e3(e3)
        d3_4 = self.d3_d4(_upsample_to(d4, (h3, w3)))
        d3_5 = self.d3_e5(_upsample_to(e5, (h3, w3)))
        d3 = self.d3_fuse(torch.cat([d3_1, d3_2, d3_3, d3_4, d3_5], dim=1))

        # d2 (H/2 = e2)
        d2_1 = self.d2_e1(_downsample_to(e1, (h2, w2)))
        d2_2 = self.d2_e2(e2)
        d2_3 = self.d2_d3(_upsample_to(d3, (h2, w2)))
        d2_4 = self.d2_d4(_upsample_to(d4, (h2, w2)))
        d2_5 = self.d2_e5(_upsample_to(e5, (h2, w2)))
        d2 = self.d2_fuse(torch.cat([d2_1, d2_2, d2_3, d2_4, d2_5], dim=1))

        # d1 (H = e1)
        d1_1 = self.d1_e1(e1)
        d1_2 = self.d1_d2(_upsample_to(d2, (h1, w1)))
        d1_3 = self.d1_d3(_upsample_to(d3, (h1, w1)))
        d1_4 = self.d1_d4(_upsample_to(d4, (h1, w1)))
        d1_5 = self.d1_e5(_upsample_to(e5, (h1, w1)))
        d1 = self.d1_fuse(torch.cat([d1_1, d1_2, d1_3, d1_4, d1_5], dim=1))

        logits = self.outc(d1)  # (B,1,H,W)
        return logits
