import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

# timm versions vary; resample_abs_pos_embed may be in different places
try:
    from timm.layers.pos_embed import resample_abs_pos_embed
except Exception:
    try:
        from timm.models.vision_transformer import resize_pos_embed as resample_abs_pos_embed  # older fallback
    except Exception:
        resample_abs_pos_embed = None


# -------------------------
# Blocks (Local DSConv U-Net encoder/decoder)
# -------------------------
class DSConv(nn.Module):
    """Depthwise-Separable Conv block: DWConv -> PWConv -> BN -> ReLU (x2)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.dw1 = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False)
        self.pw1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.dw2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, groups=out_ch, bias=False)
        self.pw2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.pw1(self.dw1(x))))
        x = self.act(self.bn2(self.pw2(self.dw2(x))))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DSConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------------
# Global ViT branch (DINO-like via timm pretrained ViT)
# -------------------------
class ViTGlobal(nn.Module):
    """
    Uses a timm ViT (e.g., vit_small_patch8_224 / vit_base_patch8_224).
    We want to feed 448x448, so we must ensure PatchEmbed accepts that size.

    This class:
      - creates the model (tries with img_size=448; fallbacks if timm doesn't accept)
      - interpolates absolute positional embeddings to match 448 token grid
      - outputs a feature map (B, C, H/P, W/P)
    """
    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        img_size: int = 448,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        freeze: bool = False,
    ):
        super().__init__()
        self.img_size = img_size

        # Try create_model with img_size
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                img_size=img_size,
                num_classes=0,           # no classifier head
                drop_rate=drop_rate,
            )
        except TypeError:
            # Older timm: no img_size arg
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                drop_rate=drop_rate,
            )
            # Patch the patch_embed expected size so it doesn't assert 224
            if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "img_size"):
                try:
                    self.vit.patch_embed.img_size = (img_size, img_size)
                except Exception:
                    pass
            if hasattr(self.vit, "patch_embed") and hasattr(self.vit.patch_embed, "grid_size"):
                # update grid size according to patch size
                ps = getattr(self.vit.patch_embed, "patch_size", 16)
                if isinstance(ps, tuple):
                    ph, pw = ps
                else:
                    ph, pw = ps, ps
                self.vit.patch_embed.grid_size = (img_size // ph, img_size // pw)

        self.embed_dim = getattr(self.vit, "embed_dim", None)
        if self.embed_dim is None:
            self.embed_dim = getattr(self.vit, "num_features", None)
        if self.embed_dim is None:
            raise ValueError("Could not infer vit embed_dim/num_features from timm model")

        self.has_cls = hasattr(self.vit, "cls_token") and (self.vit.cls_token is not None)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W) expected H=W=448 (padded).
        returns: (B,C,Hp,Wp) where Hp=H/patch, Wp=W/patch.
        """
        B, C, H, W = x.shape

        # patch embed (timm returns B,N,C)
        x = self.vit.patch_embed(x)
        N = x.shape[1]

        # patch size
        patch_size = getattr(self.vit.patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph, pw = patch_size, patch_size

        gh, gw = H // ph, W // pw

        # positional embeddings: interpolate to (gh, gw)
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None and resample_abs_pos_embed is not None:
            pos_embed = self.vit.pos_embed

            if self.has_cls:
                cls_pos = pos_embed[:, :1]
                patch_pos = pos_embed[:, 1:]
            else:
                cls_pos = None
                patch_pos = pos_embed

            # old grid
            n_old = patch_pos.shape[1]
            gs_old = int(math.sqrt(n_old))
            # if not perfect square, we still try fallback (rare)
            old_size = (gs_old, gs_old)

            try:
                patch_pos_new = resample_abs_pos_embed(
                    patch_pos,
                    new_size=(gh, gw),
                    old_size=old_size,
                    num_prefix_tokens=0,
                )
            except TypeError:
                # some timm versions use different signature
                patch_pos_new = resample_abs_pos_embed(patch_pos, (gh, gw))

            if cls_pos is not None:
                pos_embed_new = torch.cat([cls_pos, patch_pos_new], dim=1)
            else:
                pos_embed_new = patch_pos_new

            if self.has_cls:
                cls_token = self.vit.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)

            x = x + pos_embed_new
        else:
            # no pos_embed or no resampler: just add cls token if used
            if self.has_cls:
                cls_token = self.vit.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_token, x), dim=1)

        x = self.vit.pos_drop(x)

        # transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)

        # remove cls token
        if self.has_cls:
            x = x[:, 1:, :]

        # reshape tokens -> feature map
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, gh, gw)
        return x


# -------------------------
# Dual-Branch architecture
# -------------------------
class DualBranchDINO(nn.Module):
    """
    Local DSConv U-Net encoder -> L1..L4
    Global ViT -> tokens map at 56x56 (patch8) -> project -> G4 then upsample -> G3,G2,G1
    Fusion per level -> F1..F4
    Decode U-Net -> mask
    """
    def __init__(
        self,
        vit_name: str,
        base_ch: int = 48,
        in_chans: int = 3,
        out_ch: int = 1,
        pretrained_vit: bool = True,
        vit_drop: float = 0.0,
        vit_freeze: bool = False,
        use_gating: bool = True,
        img_size: int = 448,
    ):
        super().__init__()
        self.use_gating = use_gating
        self.img_size = img_size

        # Local stem
        self.stem = DSConv(in_chans, base_ch)

        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8

        self.enc1 = DSConv(base_ch, ch1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DSConv(ch1, ch2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DSConv(ch2, ch3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DSConv(ch3, ch4)

        # Global ViT branch (expects padded img_size x img_size)
        self.vit = ViTGlobal(
            model_name=vit_name,
            in_chans=in_chans,
            img_size=img_size,
            pretrained=pretrained_vit,
            drop_rate=vit_drop,
            freeze=vit_freeze,
        )
        vit_dim = self.vit.embed_dim

        # ViT patch8 -> G4 at 56x56, match local L4
        self.g4_proj = nn.Conv2d(vit_dim, ch4, 1)

        self.g3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch4, ch3, 1),
        )
        self.g2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch3, ch2, 1),
        )
        self.g1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch2, ch1, 1),
        )

        # Fusion blocks
        self.fuse1 = self._make_fuse(ch1, ch1)
        self.fuse2 = self._make_fuse(ch2, ch2)
        self.fuse3 = self._make_fuse(ch3, ch3)
        self.fuse4 = self._make_fuse(ch4, ch4)

        # Gating
        self.gate1 = nn.Conv2d(ch1, ch1, 1)
        self.gate2 = nn.Conv2d(ch2, ch2, 1)
        self.gate3 = nn.Conv2d(ch3, ch3, 1)
        self.gate4 = nn.Conv2d(ch4, ch4, 1)

        # Decoder
        self.up3 = UpBlock(ch4, ch3, ch3)
        self.up2 = UpBlock(ch3, ch2, ch2)
        self.up1 = UpBlock(ch2, ch1, ch1)

        self.head = nn.Conv2d(ch1, out_ch, kernel_size=1)

    def _make_fuse(self, lch: int, gch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(lch + gch, lch, 1, bias=False),
            nn.BatchNorm2d(lch),
            nn.ReLU(inplace=True),
            DSConv(lch, lch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local encoder
        x0 = self.stem(x)
        L1 = self.enc1(x0)                 # 448
        L2 = self.enc2(self.pool1(L1))     # 224
        L3 = self.enc3(self.pool2(L2))     # 112
        L4 = self.enc4(self.pool3(L3))     # 56

        # Global branch
        G4 = self.g4_proj(self.vit(x))     # 56
        G3 = self.g3_up(G4)                # 112
        G2 = self.g2_up(G3)                # 224
        G1 = self.g1_up(G2)                # 448

        # Gated fusion
        if self.use_gating:
            a1 = torch.sigmoid(self.gate1(G1))
            a2 = torch.sigmoid(self.gate2(G2))
            a3 = torch.sigmoid(self.gate3(G3))
            a4 = torch.sigmoid(self.gate4(G4))
            L1g = L1 * a1
            L2g = L2 * a2
            L3g = L3 * a3
            L4g = L4 * a4
        else:
            L1g, L2g, L3g, L4g = L1, L2, L3, L4

        F1 = self.fuse1(torch.cat([L1g, G1], dim=1))
        F2 = self.fuse2(torch.cat([L2g, G2], dim=1))
        F3 = self.fuse3(torch.cat([L3g, G3], dim=1))
        F4 = self.fuse4(torch.cat([L4g, G4], dim=1))

        # Decoder
        d3 = self.up3(F4, F3)
        d2 = self.up2(d3, F2)
        d1 = self.up1(d2, F1)

        return self.head(d1)
