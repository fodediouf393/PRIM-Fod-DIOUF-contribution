# src/architectures/UnetBased/models/unet3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """(ConvBNReLU)*2"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1),
        )

    def forward(self, x):
        return self.block(x)


class UNet3Plus(nn.Module):
    """
    UNet 3+ (full-scale skip connections) - 5 levels.
    Decoder at each level aggregates features from all encoder levels (and deeper decoder) into a fused feature.

    Args:
        in_channels: 3 for your case (ILM_OPL, OPL_BM, FULL)
        n_classes: 1 for binary logits
        base: base number of filters (e.g. 42)
        decoder_ch: number of channels used for each scale-projected feature before fusion (paper often uses 64/128).
        deep_supervision: if True, returns list of outputs at multiple decoder depths; else single output.
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 1,
        base: int = 42,
        decoder_ch: int = 128,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base = base
        self.decoder_ch = decoder_ch
        self.deep_supervision = deep_supervision

        # Encoder channels
        ch = [base, base * 2, base * 4, base * 8, base * 16]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder blocks
        self.enc1 = DoubleConv(in_channels, ch[0])  # 256
        self.enc2 = DoubleConv(ch[0], ch[1])        # 128
        self.enc3 = DoubleConv(ch[1], ch[2])        # 64
        self.enc4 = DoubleConv(ch[2], ch[3])        # 32
        self.enc5 = DoubleConv(ch[3], ch[4])        # 16

        # For each decoder stage (d4: 32, d3: 64, d2: 128, d1: 256):
        # we project each source feature to decoder_ch and resize to target scale, concatenate all, then fuse.
        def proj(in_ch):
            return ConvBNReLU(in_ch, decoder_ch, k=3, p=1)

        # Projections for each encoder feature per decoder stage (shared projections are ok; we do per-source per-stage to keep it simple/explicit).
        # Stage d4 (target 32x32)
        self.e1_d4 = proj(ch[0]); self.e2_d4 = proj(ch[1]); self.e3_d4 = proj(ch[2]); self.e4_d4 = proj(ch[3]); self.e5_d4 = proj(ch[4])
        # Stage d3 (target 64x64)
        self.e1_d3 = proj(ch[0]); self.e2_d3 = proj(ch[1]); self.e3_d3 = proj(ch[2]); self.e4_d3 = proj(ch[3]); self.e5_d3 = proj(ch[4])
        # Stage d2 (target 128x128)
        self.e1_d2 = proj(ch[0]); self.e2_d2 = proj(ch[1]); self.e3_d2 = proj(ch[2]); self.e4_d2 = proj(ch[3]); self.e5_d2 = proj(ch[4])
        # Stage d1 (target 256x256)
        self.e1_d1 = proj(ch[0]); self.e2_d1 = proj(ch[1]); self.e3_d1 = proj(ch[2]); self.e4_d1 = proj(ch[3]); self.e5_d1 = proj(ch[4])

        # Decoder fusions (concat 5 projected maps => 5*decoder_ch)
        self.fuse_d4 = ConvBNReLU(decoder_ch * 5, decoder_ch, k=3, p=1)
        self.fuse_d3 = ConvBNReLU(decoder_ch * 5, decoder_ch, k=3, p=1)
        self.fuse_d2 = ConvBNReLU(decoder_ch * 5, decoder_ch, k=3, p=1)
        self.fuse_d1 = ConvBNReLU(decoder_ch * 5, decoder_ch, k=3, p=1)

        # Output heads
        self.out = nn.Conv2d(decoder_ch, n_classes, kernel_size=1)

        # Deep supervision optional heads (same target size 256)
        if self.deep_supervision:
            self.out_d4 = nn.Conv2d(decoder_ch, n_classes, kernel_size=1)
            self.out_d3 = nn.Conv2d(decoder_ch, n_classes, kernel_size=1)
            self.out_d2 = nn.Conv2d(decoder_ch, n_classes, kernel_size=1)

    @staticmethod
    def _resize(x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)            # 256
        e2 = self.enc2(self.pool(e1))# 128
        e3 = self.enc3(self.pool(e2))# 64
        e4 = self.enc4(self.pool(e3))# 32
        e5 = self.enc5(self.pool(e4))# 16

        # Decoder stage d4 (target = e4 scale 32)
        d4_cat = torch.cat([
            self._resize(self.e1_d4(e1), e4),
            self._resize(self.e2_d4(e2), e4),
            self._resize(self.e3_d4(e3), e4),
            self.e4_d4(e4),                    # already 32
            self._resize(self.e5_d4(e5), e4),
        ], dim=1)
        d4 = self.fuse_d4(d4_cat)              # 32

        # Decoder stage d3 (target = e3 scale 64)
        d3_cat = torch.cat([
            self._resize(self.e1_d3(e1), e3),
            self._resize(self.e2_d3(e2), e3),
            self.e3_d3(e3),                    # 64
            self._resize(self.e4_d3(e4), e3),
            self._resize(self.e5_d3(e5), e3),
        ], dim=1)
        d3 = self.fuse_d3(d3_cat)              # 64

        # Decoder stage d2 (target = e2 scale 128)
        d2_cat = torch.cat([
            self._resize(self.e1_d2(e1), e2),
            self.e2_d2(e2),                    # 128
            self._resize(self.e3_d2(e3), e2),
            self._resize(self.e4_d2(e4), e2),
            self._resize(self.e5_d2(e5), e2),
        ], dim=1)
        d2 = self.fuse_d2(d2_cat)              # 128

        # Decoder stage d1 (target = e1 scale 256)
        d1_cat = torch.cat([
            self.e1_d1(e1),                    # 256
            self._resize(self.e2_d1(e2), e1),
            self._resize(self.e3_d1(e3), e1),
            self._resize(self.e4_d1(e4), e1),
            self._resize(self.e5_d1(e5), e1),
        ], dim=1)
        d1 = self.fuse_d1(d1_cat)              # 256

        out = self.out(d1)                     # logits (B, n_classes, 256, 256)

        if self.deep_supervision:
            # outputs upsampled to 256 for auxiliary losses
            o4 = self._resize(self.out_d4(d4), e1)
            o3 = self._resize(self.out_d3(d3), e1)
            o2 = self._resize(self.out_d2(d2), e1)
            return [o4, o3, o2, out]

        return out
