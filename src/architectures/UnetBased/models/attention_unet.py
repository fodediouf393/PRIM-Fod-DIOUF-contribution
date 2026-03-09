# src/architectures/UnetBased/models/attention_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpConv(nn.Module):
    """Upsample + 1x1 conv (to reduce channels)"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, ref):
        x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
        x = self.relu(self.bn(self.conv(x)))
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate:
      g: gating signal from decoder (coarser)
      x: skip connection from encoder (finer)
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g and x may have different spatial sizes -> upsample g to x
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al.):
    - Encoder/Decoder UNet-like
    - Attention gates on skip connections

    Args:
        in_channels: input channels (3 for your projections)
        n_classes: output channels (1 for binary logits)
        base: base number of filters (42)
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base: int = 42):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, nb[0])
        self.enc2 = ConvBlock(nb[0], nb[1])
        self.enc3 = ConvBlock(nb[1], nb[2])
        self.enc4 = ConvBlock(nb[2], nb[3])
        self.enc5 = ConvBlock(nb[3], nb[4])

        # Decoder up + attention + conv
        self.up4 = UpConv(nb[4], nb[3])
        self.att4 = AttentionGate(F_g=nb[3], F_l=nb[3], F_int=nb[2])
        self.dec4 = ConvBlock(nb[3] + nb[3], nb[3])

        self.up3 = UpConv(nb[3], nb[2])
        self.att3 = AttentionGate(F_g=nb[2], F_l=nb[2], F_int=nb[1])
        self.dec3 = ConvBlock(nb[2] + nb[2], nb[2])

        self.up2 = UpConv(nb[2], nb[1])
        self.att2 = AttentionGate(F_g=nb[1], F_l=nb[1], F_int=nb[0])
        self.dec2 = ConvBlock(nb[1] + nb[1], nb[1])

        self.up1 = UpConv(nb[1], nb[0])
        self.att1 = AttentionGate(F_g=nb[0], F_l=nb[0], F_int=max(1, nb[0] // 2))
        self.dec1 = ConvBlock(nb[0] + nb[0], nb[0])

        # Output head
        self.out_conv = nn.Conv2d(nb[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)              # (B, nb0, H, W)
        x2 = self.enc2(self.pool(x1))  # (B, nb1, H/2, W/2)
        x3 = self.enc3(self.pool(x2))  # (B, nb2, H/4, W/4)
        x4 = self.enc4(self.pool(x3))  # (B, nb3, H/8, W/8)
        x5 = self.enc5(self.pool(x4))  # (B, nb4, H/16, W/16)

        # Decoder with attention gates on skip connections
        d4 = self.up4(x5, x4)
        x4_att = self.att4(d4, x4)
        d4 = self.dec4(torch.cat([x4_att, d4], dim=1))

        d3 = self.up3(d4, x3)
        x3_att = self.att3(d3, x3)
        d3 = self.dec3(torch.cat([x3_att, d3], dim=1))

        d2 = self.up2(d3, x2)
        x2_att = self.att2(d2, x2)
        d2 = self.dec2(torch.cat([x2_att, d2], dim=1))

        d1 = self.up1(d2, x1)
        x1_att = self.att1(d1, x1)
        d1 = self.dec1(torch.cat([x1_att, d1], dim=1))

        return self.out_conv(d1)
